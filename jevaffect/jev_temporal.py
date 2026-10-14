"""Masked temporal ordinal fusion with optional intervention-utility targets.

This is an experimental Jev-inspired model, not Jev weights or a novelty claim.
Text tokens have no timestamps, so text supplies utterance context only. Audio
and video windows use the timestamps recorded by the raw-video extractor.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class TemporalInputError(ValueError):
    pass


def _masked_mean(features: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(features.dtype).unsqueeze(-1)
    return (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def _windows(features: Tensor, valid: Tensor, times: Tensor, duration: Tensor, count: int) -> tuple[Tensor, Tensor]:
    """Pool only real frames into physical-time windows; padding cannot vote."""
    b, t, d = features.shape
    if times.shape != (b, t) or valid.shape != (b, t):
        raise TemporalInputError("timestamps/masks must match [B,T]")
    if not torch.isfinite(times).all() or torch.any(times[valid] < 0):
        raise TemporalInputError("valid timestamps must be finite and nonnegative")
    position = ((times / duration.unsqueeze(1)) * count).floor().long().clamp(0, count - 1)
    positions = position.unsqueeze(-1).expand(-1, -1, d)
    result = torch.zeros((b, count, d), device=features.device, dtype=features.dtype)
    num = torch.zeros((b, count), device=features.device, dtype=features.dtype)
    result.scatter_add_(1, positions, features * valid.unsqueeze(-1))
    num.scatter_add_(1, position, valid.to(features.dtype))
    return result / num.clamp_min(1).unsqueeze(-1), num > 0


class JevTemporalOrdinal(nn.Module):
    """Independent five-class T/A/V model with explicit missing-modality masks.

    `utility_targets`, when supplied, must be training-only OOF differences in
    per-record loss after masking a modality-window. This module never creates
    targets from its own labels or from validation/test data.
    """

    def __init__(self, *, text_dim:int=768, audio_dim:int=768, vision_dim:int=384,
                 hidden:int=96, windows:int=8, dropout:float=0.1,
                 gate_kind:str="learned", temporal:bool=True) -> None:
        super().__init__()
        if hidden < 16 or windows < 1 or gate_kind not in ("learned", "entropy", "global_jsd", "uniform"):
            raise TemporalInputError("invalid hidden width, window count or gate kind")
        self.windows=windows;self.gate_kind=gate_kind;self.temporal=temporal
        self.text_projection=nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim,hidden),nn.GELU())
        self.audio_projection=nn.Sequential(nn.LayerNorm(audio_dim),nn.Linear(audio_dim,hidden),nn.GELU())
        self.vision_projection=nn.Sequential(nn.LayerNorm(vision_dim),nn.Linear(vision_dim,hidden),nn.GELU())
        self.audio_temporal=nn.Conv1d(hidden,hidden,3,padding=1,groups=hidden,bias=False)
        self.vision_temporal=nn.Conv1d(hidden,hidden,3,padding=1,groups=hidden,bias=False)
        self.text_head=nn.Linear(hidden,5)
        self.audio_head=nn.Linear(hidden,5)
        self.vision_head=nn.Linear(hidden,5)
        self.prior_logits=nn.Parameter(torch.zeros(5))
        # Four cumulative ordinal differences plus three entropies and time.
        self.utility_head=nn.Sequential(nn.Linear(2*hidden+hidden+4+4+3+1,hidden),
                                        nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,2))
        self.fused_delta=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,hidden),
                                       nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,5))
        self.correction_scale=nn.Parameter(torch.tensor(0.1))

    @staticmethod
    def _validate(batch:Mapping[str,Tensor])->None:
        required=("text","audio","vision","text_mask","audio_mask","vision_mask","audio_times","vision_times")
        if any(key not in batch for key in required):
            raise TemporalInputError("all feature, mask and timestamp keys are required")
        b=None
        for kind in ("text","audio","vision"):
            array=batch[kind];mask=batch[kind+"_mask"]
            if not isinstance(array,Tensor) or array.ndim!=3 or not array.is_floating_point():
                raise TemporalInputError(kind+" must be float [B,T,D]")
            if not isinstance(mask,Tensor) or mask.shape!=array.shape[:2] or mask.dtype!=torch.bool:
                raise TemporalInputError(kind+"_mask must be bool [B,T]")
            if b is None:b=array.shape[0]
            elif b!=array.shape[0]:raise TemporalInputError("modalities have different batch sizes")
            if not torch.isfinite(array[mask]).all():raise TemporalInputError(kind+" has nonfinite valid features")
        for kind in ("audio","vision"):
            if batch[kind+"_times"].shape!=batch[kind].shape[:2]:
                raise TemporalInputError(kind+" timestamps have wrong shape")

    def forward(self,batch:Mapping[str,Tensor],*,utility_targets:Tensor|None=None,
                utility_weight:float=0.0,duration_override:Tensor|None=None)->dict[str,Tensor]:
        self._validate(batch)
        if utility_weight<0 or not math.isfinite(utility_weight):
            raise TemporalInputError("utility_weight must be finite and nonnegative")
        masks={m:batch[m+"_mask"] for m in ("text","audio","vision")}
        # Padded values are removed before projection or convolution.
        xt=self.text_projection(batch["text"].masked_fill(~masks["text"].unsqueeze(-1),0))
        xa=self.audio_projection(batch["audio"].masked_fill(~masks["audio"].unsqueeze(-1),0))
        xv=self.vision_projection(batch["vision"].masked_fill(~masks["vision"].unsqueeze(-1),0))
        xa=xa*masks["audio"].unsqueeze(-1)
        xv=xv*masks["vision"].unsqueeze(-1)
        if self.temporal:
            xa=self.audio_temporal(xa.transpose(1,2)).transpose(1,2)*masks["audio"].unsqueeze(-1)
            xv=self.vision_temporal(xv.transpose(1,2)).transpose(1,2)*masks["vision"].unsqueeze(-1)
        valid_t=masks["text"].any(1)
        tvec=_masked_mean(xt,masks["text"])
        tlogits=torch.where(valid_t.unsqueeze(-1),self.text_head(tvec),self.prior_logits)
        # A vision frame at time zero is still a valid observation. Endpoints
        # include half the 2 fps frame interval, avoiding zero duration.
        at=batch["audio_times"].masked_fill(~masks["audio"],0)
        vt=batch["vision_times"].masked_fill(~masks["vision"],0)
        duration=torch.maximum(at.max(1).values,vt.max(1).values+0.25*masks["vision"].any(1)).clamp_min(0.25)
        if duration_override is not None:
            if (duration_override.shape!=(batch["text"].shape[0],) or not torch.isfinite(duration_override).all() or torch.any(duration_override<0.25)):
                raise TemporalInputError("duration_override must be finite [B] and >=0.25 seconds")
            duration=duration_override.to(duration.dtype)
        if self.temporal:
            a,am=_windows(xa,masks["audio"],batch["audio_times"],duration,self.windows)
            v,vm=_windows(xv,masks["vision"],batch["vision_times"],duration,self.windows)
        else:
            # True no-time control: no timestamps, temporal order, or window
            # position can influence the classifier or utility descriptor.
            a=_masked_mean(xa,masks["audio"])[:,None,:].expand(-1,self.windows,-1)
            v=_masked_mean(xv,masks["vision"])[:,None,:].expand(-1,self.windows,-1)
            am=masks["audio"].any(1)[:,None].expand(-1,self.windows)
            vm=masks["vision"].any(1)[:,None].expand(-1,self.windows)
        alogits=self.audio_head(a);vlogits=self.vision_head(v)
        tp=tlogits.softmax(-1);ap=alogits.softmax(-1);vp=vlogits.softmax(-1)
        t_cdf=tp.cumsum(-1)[:,:4].unsqueeze(1)
        a_cdf=ap.cumsum(-1)[...,:4];v_cdf=vp.cumsum(-1)[...,:4]
        def entropy(prob:Tensor)->Tensor:
            return -(prob*prob.clamp_min(1e-10).log()).sum(-1,keepdim=True)/math.log(5)
        time_fraction=(torch.arange(self.windows,device=tvec.device,dtype=tvec.dtype)+0.5)/self.windows
        time_fraction=time_fraction.view(1,self.windows,1).expand(tvec.shape[0],-1,-1)
        if not self.temporal:time_fraction=torch.zeros_like(time_fraction)
        descriptor=torch.cat((tvec[:,None,:].expand(-1,self.windows,-1),a,v,
                              (t_cdf-a_cdf).abs(),(t_cdf-v_cdf).abs(),
                              entropy(tp).unsqueeze(1).expand(-1,self.windows,-1),
                              entropy(ap),entropy(vp),time_fraction),dim=-1)
        utility=self.utility_head(descriptor)
        validity=torch.stack((am,vm),dim=-1)
        if self.gate_kind=="learned":gate=utility.sigmoid()
        elif self.gate_kind=="entropy":gate=torch.cat((1-entropy(ap),1-entropy(vp)),dim=-1).clamp(0,1)
        elif self.gate_kind=="global_jsd":
            ap_global=(ap*am.unsqueeze(-1)).sum(1)/am.sum(1).clamp_min(1).unsqueeze(-1)
            vp_global=(vp*vm.unsqueeze(-1)).sum(1)/vm.sum(1).clamp_min(1).unsqueeze(-1)
            def js_reliability(q:Tensor)->Tensor:
                middle=(tp+q)/2
                kl_p=(tp*(tp.clamp_min(1e-10)/middle.clamp_min(1e-10)).log()).sum(-1)
                kl_q=(q*(q.clamp_min(1e-10)/middle.clamp_min(1e-10)).log()).sum(-1)
                return (1-(kl_p+kl_q)/(2*math.log(2))).clamp(0,1)
            ga=js_reliability(ap_global)[:,None,None].expand(-1,self.windows,1)
            gv=js_reliability(vp_global)[:,None,None].expand(-1,self.windows,1)
            gate=torch.cat((ga,gv),dim=-1)
        else:gate=torch.ones_like(utility)
        gate=gate*validity.to(gate.dtype)
        correction=(a*gate[...,0:1]+v*gate[...,1:2]).sum(1)/gate.sum((1,2)).clamp_min(1).unsqueeze(-1)
        correction_logits=self.correction_scale*self.fused_delta(correction)
        logits=tlogits+torch.where(gate.sum((1,2)).unsqueeze(-1)>0,
                                   correction_logits,torch.zeros_like(correction_logits))
        interval=torch.stack((torch.arange(self.windows,device=duration.device,dtype=duration.dtype)/self.windows,
                              (torch.arange(self.windows,device=duration.device,dtype=duration.dtype)+1)/self.windows),dim=-1)
        intervals=duration[:,None,None]*interval[None,:,:]
        output={"logits":logits,"text_logits":tlogits,"audio_window_logits":alogits,
                "vision_window_logits":vlogits,"utility_pred":utility,"gate":gate,
                "valid_windows":validity,"time_windows_seconds":intervals}
        if utility_targets is not None:
            if utility_targets.shape!=utility.shape:
                raise TemporalInputError("utility targets must be [B,W,2]")
            supervised=torch.isfinite(utility_targets)&validity
            clipped=utility_targets.nan_to_num(nan=0).clamp(-3,3)
            loss=((utility.clamp(-3,3)-clipped)**2*supervised).sum()/supervised.sum().clamp_min(1)
            output["utility_loss"]=loss*utility_weight
        elif utility_weight:
            raise TemporalInputError("positive utility weight requires training-only OOF targets")
        return output
