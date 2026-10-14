"""Masked multi-key sequence controls for scalar gates and normalization.

Independent frozen-feature adaptation of KuDA's branch-ratio idea, with
learned latent queries instead of its learned sequence-length alignment.
No original formula, trained KuDA reproduction or calibrated reliability claim.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

MODALITIES=("text","audio","vision")
CASES=("static","pre_norm","post_norm","distilled_static","distilled_pre_norm","distilled_post_norm","learned_post_norm")


def valid_mask(features, modality):
    x=features[modality]
    if x.ndim!=3 or x.shape[1]<1:
        raise ValueError("Nonempty padded sequence axis required")
    if "masks" in features:
        mask=features["masks"][modality]
        if mask.dtype!=torch.bool or mask.shape!=x.shape[:2]:
            raise ValueError("Invalid explicit sequence mask")
        if "lengths" in features and not torch.equal(mask.sum(1),features["lengths"][modality]):
            raise ValueError("Explicit mask and valid-step count disagree")
        return mask
    lengths=features["lengths"][modality]
    if x.ndim!=3 or lengths.ndim!=1 or lengths.shape[0]!=x.shape[0] or lengths.dtype not in (torch.int32,torch.int64) or torch.any(lengths<0) or torch.any(lengths>x.shape[1]):
        raise ValueError("Invalid sequence shape/lengths")
    return torch.arange(x.shape[1],device=x.device)[None,:]<lengths[:,None]


def masked_mean(x, mask):
    return (x*mask[:,:,None]).sum(1)/mask.sum(1).clamp_min(1)[:,None]


class SequenceBranch(nn.Module):
    def __init__(self, input_dim, dim=128):
        super().__init__()
        self.projector=nn.Sequential(nn.Linear(input_dim,dim),nn.LayerNorm(dim))
        self.adapter=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim//2),nn.GELU(),nn.Linear(dim//2,dim))
        self.decoder=nn.Sequential(nn.Linear(2*dim,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))

    def forward(self,x,mask):
        # Restore padding after each affine path. Invalid input bytes cannot
        # enter the sentiment head or attention through projection biases.
        x=torch.where(mask[:,:,None],x,torch.zeros_like(x))
        h=self.projector(x)
        k=h+self.adapter(h)
        u=torch.cat((h,k),dim=-1)
        u=torch.where(mask[:,:,None],u,torch.zeros_like(u))
        score=self.decoder(masked_mean(u,mask)).flatten()
        return u,score


class UnimodalSequenceHeads(nn.Module):
    def __init__(self,dims,dim=128):
        super().__init__()
        if set(dims)!=set(MODALITIES):
            raise ValueError("T/A/V dimensions required")
        self.branches=nn.ModuleDict({m:SequenceBranch(dims[m],dim) for m in MODALITIES})

    def forward(self,features):
        return torch.stack([self.branches[m](features[m],valid_mask(features,m))[1] for m in MODALITIES],dim=1)


class CrossBranch(nn.Module):
    def __init__(self,dim,heads,ff_dim,dropout):
        super().__init__()
        self.attention=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,ff_dim),nn.GELU(),nn.Dropout(dropout),nn.Linear(ff_dim,dim),nn.Dropout(dropout))

    def forward(self,query,source,mask):
        present=mask.any(1)
        safe_mask=mask.clone()
        # MHA cannot attend to an entirely masked row. The dummy key never
        # contributes: its entire branch is zeroed after attention/normalization.
        safe_mask[~present,0]=True
        source=torch.where(mask[:,:,None],source,torch.zeros_like(source))
        update,_=self.attention(query,source,source,key_padding_mask=~safe_mask,need_weights=False)
        z=self.norm1(query+update)
        z=self.norm2(z+self.ff(z))
        return torch.where(present[:,None,None],z,torch.zeros_like(z))


class GateBlock(nn.Module):
    def __init__(self,dim,heads,ff_dim,dropout):
        super().__init__()
        self.cross=nn.ModuleDict({m:CrossBranch(dim,heads,ff_dim,dropout) for m in MODALITIES})
        self.branch_norm=nn.ModuleDict({m:nn.LayerNorm(dim) for m in MODALITIES})
        self.mix_norm=nn.LayerNorm(dim)
        self.self_attention=nn.MultiheadAttention(dim,heads,dropout=0,batch_first=True)
        self.norm1=nn.LayerNorm(dim,eps=1e-6)
        self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.norm3=nn.LayerNorm(dim,eps=1e-6)
        self.ff=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,ff_dim),nn.GELU(),nn.Linear(ff_dim,dim))

    def forward(self,query,tokens,masks,ratios,case):
        routed=[]
        variances=[]
        for i,m in enumerate(MODALITIES):
            z=self.cross[m](query,tokens[m],masks[m])
            present=masks[m].any(1)
            variances.append(z.var(dim=-1,unbiased=False).mean(1))
            scale=(1+ratios[:,i])[:,None,None]
            if case=="source_none":
                transformed=z
            else:
                transformed=scale*self.branch_norm[m](z) if case=="post_norm" else self.branch_norm[m](scale*z)
            routed.append(torch.where(present[:,None,None],transformed,torch.zeros_like(transformed)))
        mixed=self.mix_norm(torch.stack(routed).sum(0))
        query=self.norm1(query+mixed)
        update,_=self.self_attention(query,query,query,need_weights=False)
        query=self.norm2(query+update)
        query=self.norm3(query+self.ff(query))
        return query,torch.stack(variances,dim=1)


def correlation_loss(tokens,masks,fused):
    losses=[]
    for m in MODALITIES:
        present=masks[m].any(1)
        if int(present.sum())<2:
            losses.append(tokens[m].sum()*0)
            continue
        a=F.normalize(masked_mean(tokens[m],masks[m])[present],dim=-1,eps=1e-8)
        b=F.normalize(fused.mean(1)[present],dim=-1,eps=1e-8)
        similarity=a@b.T
        losses.append((torch.logsumexp(similarity,dim=1)-similarity.diagonal()).mean())
    return torch.stack(losses).sum()


class SequenceGateRegressor(nn.Module):
    def __init__(self,dims,case,dim=128,queries=8,depth=3,heads=4,ff_dim=256,dropout=.3):
        super().__init__()
        if case not in CASES or queries<2 or depth<1 or dim%heads:
            raise ValueError("Registered case and genuine multi-query architecture required")
        self.case=case
        self.branches=UnimodalSequenceHeads(dims,dim).branches
        self.align=nn.ModuleDict({m:nn.Linear(2*dim,dim) for m in MODALITIES})
        self.queries=nn.Parameter(torch.randn(queries,dim)*.02)
        self.blocks=nn.ModuleList([GateBlock(dim,heads,ff_dim,dropout) for _ in range(depth)])
        self.head=nn.Sequential(nn.Linear(dim,64),nn.GELU(),nn.Linear(64,32),nn.GELU(),nn.Linear(32,1))
        # A deployment policy learns a TRAIN-only teacher ratio. All arms keep
        # this same head; only distilled arms train it. Distilled-static trains
        # the same auxiliary task but does not execute the policy in fusion.
        self.policy=nn.Sequential(nn.Linear(3*dim+3,64),nn.GELU(),nn.Linear(64,3))

    def load_unimodal_state(self,state):
        prefix="branches."
        if not state or any(not k.startswith(prefix) for k in state):
            raise ValueError("UnimodalSequenceHeads state required")
        self.branches.load_state_dict({k[len(prefix):]:v for k,v in state.items()},strict=True)

    def freeze_knowledge(self):
        for branch in self.branches.values():
            for module in (branch.adapter,branch.decoder):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)

    def _extract(self,features):
        masks={m:valid_mask(features,m) for m in MODALITIES}
        tokens,scores={},[]
        for m in MODALITIES:
            u,p=self.branches[m](features[m],masks[m])
            aligned=self.align[m](u)
            tokens[m]=torch.where(masks[m][:,:,None],aligned,torch.zeros_like(aligned))
            scores.append(p)
        return tokens,masks,torch.stack(scores,dim=1)

    def _fuse(self,tokens,masks,scores,ratios,placement=None):
        query=self.queries[None]+torch.stack([masked_mean(tokens[m],masks[m]) for m in MODALITIES]).sum(0)[:,None,:]
        variance=None
        for block in self.blocks:
            operation=placement or ("post_norm" if self.case.endswith("post_norm") else "pre_norm")
            query,variance=block(query,tokens,masks,ratios,operation)
        return {"prediction":self.head(query.mean(1)).flatten(),"contrastive":correlation_loss(tokens,masks,query),
                "ratios":ratios,"unimodal_scores":scores,"branch_variances":variance}

    def _policy_distribution(self,tokens,masks,scores,present):
        context=torch.cat([*(masked_mean(tokens[m],masks[m]) for m in MODALITIES),scores.detach()],dim=-1)
        logits=self.policy(context)
        safe=present|(~present.any(1))[:,None]
        log_policy=F.log_softmax(logits.masked_fill(~safe,-torch.inf),dim=1)
        return log_policy.exp()*present,log_policy.masked_fill(~present,0)

    def forward(self,features,target=None):
        if not self.training and target is not None:
            raise ValueError("Evaluation/inference must not receive a target")
        tokens,masks,scores=self._extract(features)
        present=torch.stack([masks[m].any(1) for m in MODALITIES],dim=1)
        distilled=self.case.startswith("distilled_")
        policy,log_policy=self._policy_distribution(tokens,masks,scores,present)
        distillation=scores.sum()*0
        if distilled and self.training:
            if target is None or target.shape!=(scores.shape[0],) or not torch.isfinite(target).all():
                raise ValueError("Policy distillation requires TRAIN-only native M targets")
            teacher_logits=-.3*(scores.detach()-target.detach()[:,None]).square()
            safe=present|(~present.any(1))[:,None]
            teacher=torch.softmax(teacher_logits.masked_fill(~safe,-torch.inf),dim=1)*present
            row_loss=(teacher*(teacher.clamp_min(1e-8).log()-log_policy)).sum(1)
            observed=present.any(1)
            distillation=(row_loss*observed).sum()/observed.sum().clamp_min(1)
        if self.case in ("distilled_pre_norm","distilled_post_norm","learned_post_norm"):
            ratios=policy  # Target-free deployed policy; same train/eval rule.
        elif not self.training:
            ratios=present.to(scores.dtype)  # Paper target-free boundary: R=1.
        elif self.case in ("static","distilled_static"):
            ratios=present.to(scores.dtype)/present.sum(1).clamp_min(1)[:,None]
        else:
            if target is None or target.shape!=(scores.shape[0],) or not torch.isfinite(target).all():
                raise ValueError("Dynamic training requires finite native M targets")
            logits=-.3*(scores-target.detach()[:,None]).square()
            safe=present|(~present.any(1))[:,None]
            ratios=torch.softmax(logits.masked_fill(~safe,-torch.inf),dim=1)*present
        output=self._fuse(tokens,masks,scores,ratios)
        output.update(distillation=distillation,policy_ratios=policy)
        return output

    def probe(self,features,ratios):
        """Explicit target-free evaluation intervention; not the public predictor."""
        if self.training:
            raise ValueError("Diagnostic probe requires evaluation mode")
        tokens,masks,scores=self._extract(features)
        if ratios.shape!=scores.shape or not torch.isfinite(ratios).all() or torch.any(ratios<0):
            raise ValueError("Finite nonnegative diagnostic ratios required")
        return self._fuse(tokens,masks,scores,ratios)

    def probe_source_none(self,features):
        """Target-free pinned-code inference sensitivity, omitting branch LN."""
        if self.training:
            raise ValueError("Diagnostic probe requires evaluation mode")
        tokens,masks,scores=self._extract(features)
        ratios=torch.stack([masks[m].any(1) for m in MODALITIES],dim=1).to(scores.dtype)
        return self._fuse(tokens,masks,scores,ratios,placement="source_none")
