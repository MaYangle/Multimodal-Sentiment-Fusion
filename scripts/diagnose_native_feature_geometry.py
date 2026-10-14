"""TRAIN-only pooled feature geometry, without causal nuisance interpretation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import prepare
from jevaffect.native_training import digest,save_json


def geometry(features,ids):
    x=np.asarray(features,np.float64)
    groups=np.array([s.split("/")[0] for s in ids])
    center=x.mean(0);variance=x.var(0);std=np.sqrt(variance)
    between=np.zeros(x.shape[1]);within=np.zeros(x.shape[1])
    for source in sorted(set(groups)):
        values=x[groups==source];mean=values.mean(0);fraction=len(values)/len(x)
        between+=fraction*(mean-center)**2
        within+=fraction*values.var(0)
    if not np.allclose(variance,between+within,atol=1e-12,rtol=1e-10):
        raise ValueError("Feature variance decomposition failed")
    active=variance>1e-12
    return {"dimension":x.shape[1],"count":len(x),"zero_variance_coordinates":int((~active).sum()),
        "sample_L2_norm_quantiles":np.quantile(np.linalg.norm(x,axis=1),[0,.25,.5,.75,1]).tolist(),
        "coordinate_std_quantiles":np.quantile(std,[0,.25,.5,.75,1]).tolist(),
        "total_variance_trace":float(variance.sum()),"within_source_variance_trace":float(within.sum()),
        "between_source_variance_trace":float(between.sum()),
        "between_fraction_trace_weighted":float(between.sum()/variance.sum()) if variance.sum()>0 else None,
        "between_fraction_coordinate_equal":float((between[active]/variance[active]).mean()) if active.any() else None}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text())
    if cfg["predict_valid"]:
        raise ValueError("Feature geometry is registered for TRAIN-only input")
    data=prepare(cfg)
    report={"config_sha256":digest(a.config),"author_csv_sha256":data["author_csv_sha256"],"feature_sha256":data["train"].hashes,
        "modalities":{m:geometry(x,data["train"].ids) for m,x in data["train"].features.items()},
        "scope":"Descriptive source-video geometry only; between-source variance includes real content/sentiment and is not identified causal nuisance.",
        "official_VALID_used":False,"test_evaluated":False}
    save_json(a.output,report);print(json.dumps(report["modalities"],indent=2))
