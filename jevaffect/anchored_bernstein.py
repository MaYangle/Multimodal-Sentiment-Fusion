"""Numerical prototype: censored, median-anchored two-sided Bernstein CDF.

Integer Beta basis CDFs are exact finite binomial tails. The mathematical
components are established constructions; this module claims no novelty or
measured predictive advantage. The input point is always detached.
"""
import math
import torch


def parameters(anchor,raw_margins,logits,bound=1.,degree=6):
    if not math.isfinite(bound) or bound<=0 or not isinstance(degree,int) or not 1<=degree<=12:
        raise ValueError("Finite native bound and degree1..12 required")
    if anchor.ndim!=1 or len(anchor)==0 or raw_margins.shape!=(len(anchor),2) or logits.shape!=(len(anchor),2,degree+1):
        raise ValueError("Invalid anchored Bernstein parameter shapes")
    if any(not torch.isfinite(x).all() for x in (anchor,raw_margins,logits)) or torch.any(anchor.abs()>bound):
        raise ValueError("Finite bounded native anchors/parameters required")
    m=anchor.detach().double()
    margins=torch.nn.functional.softplus(raw_margins.double())+1e-4
    widths=torch.stack((m+bound,bound-m),1)+margins
    return {"anchor":m,"widths":widths,"weights":logits.double().softmax(-1),"bound":float(bound),"degree":degree}


def beta_basis_cdf(t,degree):
    """CDF Beta(k+1,degree-k+1) for each k, stable on closed unit interval."""
    t=t.clamp(0.,1.)
    order=degree+1
    terms=torch.stack([math.comb(order,j)*t.pow(j)*(1-t).pow(order-j) for j in range(order+1)],-1)
    return torch.stack([terms[...,k+1:].sum(-1) for k in range(degree+1)],-1)


def cdf(params,value):
    m=params["anchor"]
    x=torch.as_tensor(value,dtype=torch.float64,device=m.device)
    if x.ndim==0:x=x.expand_as(m)
    if x.ndim not in (1,2) or x.shape[0]!=len(m) or not torch.isfinite(x).all():
        raise ValueError("Finite scalar, per-sample vector or sample-by-grid required")
    if x.ndim==2:
        location=m[:,None];left_width=params["widths"][:,0,None];right_width=params["widths"][:,1,None]
        weights=params["weights"][:,None]
    else:
        location=m;left_width=params["widths"][:,0];right_width=params["widths"][:,1];weights=params["weights"]
    left_t=(location-x)/left_width;right_t=(x-location)/right_width
    left=1-(beta_basis_cdf(left_t,params["degree"])*weights[...,0,:]).sum(-1)
    right=(beta_basis_cdf(right_t,params["degree"])*weights[...,1,:]).sum(-1)
    latent=torch.where(x<location,.5*left,.5+.5*right)
    bound=params["bound"]
    return torch.where(x < -bound,0.,torch.where(x>=bound,1.,latent)).clamp(0.,1.)


def bin_probabilities(params,boundaries):
    edges=torch.as_tensor(boundaries,dtype=torch.float64,device=params["anchor"].device)
    if edges.ndim!=1 or len(edges)==0 or not torch.isfinite(edges).all() or torch.any(edges[1:]<=edges[:-1]) or edges[0]<-params["bound"] or edges[-1]>=params["bound"]:
        raise ValueError("Increasing native bin boundaries required")
    values=cdf(params,edges[None].expand(len(params["anchor"]),-1))
    cumulative=torch.cat((torch.zeros_like(values[:,:1]),values,torch.ones_like(values[:,:1])),1)
    return cumulative[:,1:]-cumulative[:,:-1]


def error_band_probability(params,tolerance):
    if not math.isfinite(tolerance) or tolerance<=0:raise ValueError("Positive native-unit error tolerance required")
    lo=params["anchor"]-tolerance;hi=params["anchor"]+tolerance
    before=torch.where(lo<=-params["bound"],0.,cdf(params,lo))
    return (cdf(params,hi)-before).clamp(0.,1.)


@torch.no_grad()
def quantile(params,probabilities):
    p=torch.as_tensor(probabilities,dtype=torch.float64,device=params["anchor"].device).reshape(-1)
    if not torch.isfinite(p).all() or torch.any(p<0) or torch.any(p>1):raise ValueError("Valid quantile levels required")
    target=p[None].expand(len(params["anchor"]),-1)
    bound=params["bound"]
    lo=torch.full_like(target,-bound);hi=torch.full_like(target,bound)
    for _ in range(64):
        mid=(lo+hi)/2;below=cdf(params,mid)<target
        lo=torch.where(below,mid,lo);hi=torch.where(below,hi,mid)
    values=(lo+hi)/2
    # The known fixed median avoids floating-point plateaus at extreme basis tails.
    values=torch.where(target==.5,params["anchor"][:,None],values)
    return torch.where(target==0,-bound,torch.where(target==1,bound,values))
