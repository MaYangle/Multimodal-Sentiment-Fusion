"""Independent implementations of known source-risk and GRL baselines."""
import math
import torch
from torch import nn
from jevaffect.native_training import PooledRegressor

CASES=("clip_erm","balanced_erm","dro_uniform","group_dro","domain_probe","dann")


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.scale=float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.scale*gradient, None


def reverse_gradient(value, scale):
    if not math.isfinite(scale) or scale<0:
        raise ValueError("Finite nonnegative GRL scale required")
    return _Reverse.apply(value,scale)


def source_group_means(losses, groups, count):
    if losses.ndim!=1 or groups.shape!=losses.shape or groups.dtype!=torch.long or count<1 or len(losses)==0 or groups.min()<0 or groups.max()>=count:
        raise ValueError("Invalid source group objective inputs")
    indicator=(torch.arange(count,device=groups.device)[:,None]==groups[None]).to(losses.dtype)
    counts=indicator.sum(1)
    return (indicator@losses)/counts.clamp_min(1),counts


class SourceRiskModel(nn.Module):
    def __init__(self,dims,source_count,hidden=256,dropout=.1):
        super().__init__()
        if source_count<1:
            raise ValueError("Optimization source groups required")
        self.regressor=PooledRegressor(dims,hidden,dropout)
        # Construct this CPU-only head without advancing point/dropout RNG.
        with torch.random.fork_rng(devices=[]):
            self.domain_classifier=nn.Linear(hidden,source_count)
        self.register_buffer("log_source_probability",torch.zeros(source_count,dtype=torch.float64))

    def forward(self,features):
        return self.regressor(features)

    def point_context(self,features):
        x=torch.cat([features[m] for m in self.regressor.modalities],-1)
        for layer in self.regressor.head[:3]:
            x=layer(x)
        return self.regressor.head[3](x).flatten(),x

    def source_probability(self):
        return self.log_source_probability.softmax(0)

    def training_objective(self,features,target,groups,case,step_fraction,dro_step_size=.01,dann_gradient_scale=.1):
        if case not in CASES or not 0<=step_fraction<=1 or dro_step_size<0 or dann_gradient_scale<0:
            raise ValueError("Invalid source training case/config")
        prediction,context=self.point_context(features)
        losses=(prediction-target).abs()
        point_loss=losses.mean();domain_loss=point_loss.new_zeros(())
        scale=0.
        if case in ("dro_uniform","group_dro"):
            means,_=source_group_means(losses,groups,len(self.log_source_probability))
            if case=="group_dro":
                with torch.no_grad():
                    updated=self.log_source_probability+dro_step_size*means.detach().double()
                    self.log_source_probability.copy_(updated-torch.logsumexp(updated,0))
            point_loss=(means*self.source_probability().to(means.dtype)).sum()
        if case in ("domain_probe","dann"):
            if case=="dann":
                scale=dann_gradient_scale*(2/(1+math.exp(-10*step_fraction))-1)
                source_context=reverse_gradient(context,scale)
            else:
                source_context=context.detach()
            domain_loss=nn.functional.cross_entropy(self.domain_classifier(source_context),groups)
        return point_loss+domain_loss,{"clip_L1":losses.mean().detach(),"risk_objective":point_loss.detach(),
            "domain_CE":domain_loss.detach(),"reversal_scale":scale}
