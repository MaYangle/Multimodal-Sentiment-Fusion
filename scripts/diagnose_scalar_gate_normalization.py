"""Mechanistic checks for scalar gates inside vs after branch LayerNorm."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256
from jevaffect.native_training import digest,save_json


def layernorm(x, eps=1e-5):
    x=np.asarray(x,np.float64)
    if eps<0 or not np.isfinite(x).all():
        raise ValueError("Finite inputs and nonnegative epsilon required")
    centered=x-x.mean(-1,keepdims=True)
    variance=np.mean(centered**2,axis=-1,keepdims=True)
    if np.any(variance+eps<=0):
        raise ValueError("Zero-variance input with zero epsilon")
    return centered/np.sqrt(variance+eps)


def fuse(z, ratios, placement, eps=1e-5):
    z=np.asarray(z,np.float64)
    ratios=np.asarray(ratios,np.float64)
    if z.ndim!=3 or z.shape[1]!=3 or ratios.shape not in ((3,),z.shape[:2]) or not np.isfinite(ratios).all() or np.any(ratios<0):
        raise ValueError("Expected three branches and nonnegative per-example scalar ratios")
    if placement not in ("inside_norm","after_norm"):
        raise ValueError("Unknown placement")
    scale=(1+ratios)[...,None]
    branches=layernorm(scale*z,eps) if placement=="inside_norm" else scale*layernorm(z,eps)
    return layernorm(branches.sum(1),eps)


def singleton_attention_gradient(key_count):
    torch.manual_seed(20261014)
    qlayer=torch.nn.Linear(5,4,bias=False,dtype=torch.float64)
    klayer=torch.nn.Linear(5,4,bias=False,dtype=torch.float64)
    vlayer=torch.nn.Linear(5,4,bias=False,dtype=torch.float64)
    q=torch.randn(3,2,5,dtype=torch.float64)
    kv=torch.randn(3,key_count,5,dtype=torch.float64)
    attention=torch.softmax(qlayer(q)@klayer(kv).transpose(-1,-2)/2,dim=-1)
    output=attention@vlayer(kv)
    output.square().sum().backward()
    return {"key_count":key_count,"Q_gradient_norm":float(qlayer.weight.grad.norm()),
        "K_gradient_norm":float(klayer.weight.grad.norm()),"V_gradient_norm":float(vlayer.weight.grad.norm())}


def sensitivity(z, eps):
    ratio_a=np.array([.1,.2,.7])
    ratio_b=np.array([.7,.2,.1])
    result={}
    for placement in ("inside_norm","after_norm"):
        first,second=fuse(z,ratio_a,placement,eps),fuse(z,ratio_b,placement,eps)
        delta=np.linalg.norm(second-first,axis=1)/np.maximum(np.linalg.norm(first,axis=1),1e-12)
        result[placement]={"max_abs_output_change":float(np.abs(first-second).max()),
            "relative_l2_change_quantiles":np.quantile(delta,[0,.25,.5,.75,1]).tolist()}
    return result


def diagnose(cache, output):
    cache=Path(cache)
    meta=json.loads((cache/"metadata.json").read_text())
    if meta.get("author_csv_sha256")!=SOURCE_CSV_SHA256 or meta.get("counts")!={"train":2722,"valid":647} or meta.get("normalization",{}).get("fit_split")!="train":
        raise ValueError("Pinned official TRAIN pooled-standardized cache required")
    dims={"text":768,"audio":768,"vision":384}
    rng=np.random.default_rng(20261014)
    branches=[]
    files={"metadata.json":digest(cache/"metadata.json")}
    for name in ("ids.json",*(m+".npy" for m in dims)):
        path=cache/"train"/name
        sha=digest(path)
        if sha!=meta["artifact_sha256"]["train/"+name]:
            raise ValueError("Changed TRAIN payload: "+name)
        files["train/"+name]=sha
    for m,dim in dims.items():
        values=np.load(cache/"train"/(m+".npy"),allow_pickle=False)
        if values.shape!=(2722,1,dim) or not np.isfinite(values).all():
            raise ValueError("Invalid native pooled features")
        projection=rng.normal(size=(dim,128))/np.sqrt(dim)
        # Proxy for an INITIAL branch ending in LayerNorm with gamma=1/beta=0.
        # These are not trained KuDA states or a reproduced KuDA forward pass.
        branches.append(layernorm(values[:,0].astype(np.float64)@projection,1e-5))
    z=np.stack(branches,axis=1)
    report={"implementation_sha256":digest(__file__),"source_hashes":files,
        "source_revision":"e60ef6bf070f23855e186c9fd7563902e9a8c056",
        "source_file":"https://github.com/MKMaS-GUET/KuDA/blob/e60ef6bf070f23855e186c9fd7563902e9a8c056/models/DyRoutFusion_CLS.py",
        "sample_count":2722,"projection_seed":20261014,"projection_width":128,
        "gate":"1+R_m; R changes from [.1,.2,.7] to [.7,.2,.1] in both placements",
        "epsilon_zero":sensitivity(z,0),"epsilon_1e_5":sensitivity(z,1e-5),
        "attention_gradients":[singleton_attention_gradient(n) for n in (1,2)],
        "model_fits":0,"VALID_features_read":False,"sentiment_scores_parsed":False,"TEST_accessed":False,
        "scope":"Analytical and numerical mechanism diagnosis on normalized random projections of actual TRAIN vectors, not learned KuDA states, paper-performance reproduction, original formula or improvement evidence. For positive scalar c and eps=0, LN(c*z)=LN(z). Finite epsilon, low variance and learned affine parameters qualify practical sensitivity. Singleton-key softmax gives zero Q/K gradients; multi-key attention is distinct."}
    save_json(output,report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=diagnose(a.cache_dir,a.output)
    print(json.dumps({k:r[k] for k in ("sample_count","epsilon_zero","epsilon_1e_5","attention_gradients","model_fits")},indent=2))
