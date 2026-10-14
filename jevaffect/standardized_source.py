"""Fit-source normalization wrapper around the frozen source point trainer."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from jevaffect.anchor_calibration import normalizer,normalize
from jevaffect.anchor_crossfit import prepare,train_teacher,collect_oof
from jevaffect.native_training import digest,save_json


def transform_fold(config,data,fold):
    if fold not in (0,1,2) or config["predict_valid"] or "preprocessing" in config:
        raise ValueError("Untransformed TRAIN-only source config required")
    fit_ids=data["fold_manifest"]["folds"][fold]["fit_ids"]
    base=data["train"];index={sid:i for i,sid in enumerate(base.ids)}
    fit_indices=[index[sid] for sid in fit_ids]
    stats={m:normalizer(x,fit_indices) for m,x in base.features.items()}
    features={m:normalize(x,stats[m]) for m,x in base.features.items()}
    own=Path(__file__)
    preprocessing={"method":"coordinate mean/population STD of pooled optimization sources only; small STD<1e-6 uses1",
        "fit_ids":fit_ids,"fit_indices_in_cache_order":fit_indices,"statistics":stats,
        "raw_feature_sha256":base.hashes,"implementation":{own.name:digest(own),"anchor_calibration.py":digest(own.with_name("anchor_calibration.py"))}}
    transformed={**data,"train":SimpleNamespace(ids=base.ids,targets=base.targets,features=features,hashes=base.hashes)}
    return {**config,"preprocessing":preprocessing},transformed


def run(config,output,seed=None,fold=None,smoke_groups=None,resume=False,stop_after=None,allow_fixture=False):
    data=prepare(config,allow_fixture=allow_fixture,smoke_groups=smoke_groups)
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    smoke=bool(data["fixture_smoke"] or smoke_groups is not None)
    results=[]
    save_json(out.with_suffix(out.suffix+".manifest.json"),{"config":config,"ids":data["ids"],"source_csv_sha256":data["author_csv_sha256"],
        "fold_manifest":data["fold_manifest"],"raw_feature_sha256":data["train"].hashes,"smoke_only":smoke,"test_evaluated":False})
    prepared={f:transform_fold(config,data,f) for f in ([fold] if fold is not None else (0,1,2))}
    for s in ([seed] if seed else (13,42,2026)):
        for f,(actual_config,actual_data) in prepared.items():
            results.append(train_teacher(actual_config,actual_data,view="fusion",seed=s,fold=f,resume=resume,stop_after=stop_after))
            save_json(out,{"results":results,"suite_complete":False,"smoke_only":smoke,"test_evaluated":False})
    oof={}
    for s in ([seed] if seed else (13,42,2026)):
        value=collect_oof(config["run_root"],"smoke" if smoke else "formal","fusion",s,data["ids"])
        if value is not None:
            oof[f"fusion/seed_{s}"]=value
    complete=seed is None and fold is None and not smoke and stop_after is None and len(results)==9 and all(r["complete"] for r in results) and len(oof)==3
    report={"results":results,"oof":oof,"suite_complete":complete,"smoke_only":smoke,"test_evaluated":False,"official_VALID_used":False}
    save_json(out,report);return report
