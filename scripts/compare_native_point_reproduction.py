"""Exact full text/fusion retraining comparison in independent run directories."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest,save_json
from scripts.prepare_anchor_calibration_tables import _normalized_feature_hashes


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference",type=Path,required=True);p.add_argument("--reproduced",type=Path,required=True)
    p.add_argument("--reference-root",type=Path,required=True);p.add_argument("--reproduced-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if a.reference_root.resolve()==a.reproduced_root.resolve():raise ValueError("Independent run roots required")
    suites=[json.loads(s.read_text()) for s in (a.reference,a.reproduced)]
    expected={(m,s) for m in ("text","early_fusion") for s in (13,42,2026)};maps=[]
    for suite in suites:
        rows=suite["results"]
        if not suite["suite_complete"] or len(rows)!=6 or {(r["model"],r["seed"]) for r in rows}!=expected or any(not r["complete"] or r["smoke_only"] or r["test_evaluated"] for r in rows):raise ValueError("Complete six-run native point suites required")
        maps.append({(r["model"],r["seed"]):r for r in rows})
    evidence=[]
    for model,seed in sorted(expected):
        roots=[r/model/f"seed_{seed}" for r in (a.reference_root,a.reproduced_root)]
        results=[m[(model,seed)] for m in maps];ids=[json.loads((r/"identity.json").read_text()) for r in roots]
        configs=[{k:v for k,v in i["config"].items() if k not in ("cache_dir","run_root")} for i in ids]
        if configs[0]!=configs[1] or any(ids[0][k]!=ids[1][k] for k in ("author_csv_sha256","implementation","torch","numpy","device","gpu","selection","loss")):raise ValueError("Native reproduction identity changed beyond data/output paths")
        for split in ("train","valid"):
            if _normalized_feature_hashes(ids[0]["features"][split])!=_normalized_feature_hashes(ids[1]["features"][split]):raise ValueError("Re-prepared native cache bytes changed")
        for result,root in zip(results,roots):
            if digest(root/"best.pt")!=result["checkpoint_sha256"] or digest(root/"valid_predictions.json")!=result["predictions_sha256"]:raise ValueError("Reproduced checkpoint/prediction artifact changed")
        predictions=[json.loads((r/"valid_predictions.json").read_text()) for r in roots]
        if predictions[0]!=predictions[1] or results[0]["valid_metrics"]!=results[1]["valid_metrics"] or results[0]["best_epoch"]!=results[1]["best_epoch"]:raise ValueError("Native reproduced forecasts/metrics/selection differ")
        history=[[{k:v for k,v in row.items() if k!="seconds"} for row in json.loads((r/"history.json").read_text())] for r in roots]
        if history[0]!=history[1]:raise ValueError("Native training losses/epoch metrics differ")
        states=[torch.load(r/"best.pt",weights_only=False,map_location="cpu",mmap=True) for r in roots]
        if any(not torch.equal(value,states[1]["model"][name]) for name,value in states[0]["model"].items()):raise ValueError("Native trained parameter tensors differ")
        evidence.append({"model":model,"seed":seed,"647_prediction_records_and_metrics_exact":True,"parameters_and_history_exact":True,
            "epochs":len(history[0]),"updates":len(history[0])*int(np.ceil(results[0]["train_count"]/configs[0]["batch_size"]))})
    report={"reference_suite_sha256":digest(a.reference),"reproduced_suite_sha256":digest(a.reproduced),"evidence":evidence,
        "passed":True,"test_evaluated":False,"scope":"Six official CH pooled-point full retrainings and cache-byte verification; not encoder/original-method/final-test reproduction"}
    save_json(a.output,report);print(json.dumps({"passed":True,"verified_runs":len(evidence)}))
