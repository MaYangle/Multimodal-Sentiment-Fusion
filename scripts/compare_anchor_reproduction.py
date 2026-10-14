"""Compare every calibration run against an isolated source-copy retraining."""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest, save_json


def read(path):
    return json.loads(Path(path).read_text())


def compare(original, reproduction, original_base, reproduction_base):
    a,b=read(original),read(reproduction)
    expected={(c,s) for c in ("global","scale","mixture","views","disagreement","direct_hit","mdn") for s in (13,42,2026)}
    by=[]
    for suite in (a,b):
        if not suite["suite_complete"] or suite["smoke_only"] or len(suite["results"])!=21 or {(r["case"],r["seed"]) for r in suite["results"]}!=expected:
            raise ValueError("Every registered formal case and seed must be reproduced")
        by.append({(r["case"],r["seed"]):r for r in suite["results"]})
    evidence=[]
    for key in sorted(expected):
        first,second=by[0][key],by[1][key]
        roots=[Path(base)/result["root"] for base,result in ((original_base,first),(reproduction_base,second))]
        identities=[read(r/"identity.json") for r in roots]
        allowed={"run_root","table_dir"}
        normalized=[]
        for identity,base in zip(identities,(original_base,reproduction_base)):
            cfg=dict(identity["config"])
            table=Path(base)/cfg["table_dir"]
            if digest(table/"metadata.json")!=identity["metadata_sha256"]:
                raise ValueError("Reproduction table metadata does not match training identity")
            cfg={k:v for k,v in cfg.items() if k not in allowed}
            normalized.append({**identity,"config":cfg})
        if normalized[0]!=normalized[1]:
            raise ValueError(f"Reproduction identity differs beyond output/input path: {key}")
        metrics=("valid","complete","smoke_only","parameters","optimizer_calls","scale","best_epoch","epochs_executed","optimizer_updates","first_batch_active_gradient_parameters","test_evaluated")
        if any(first.get(k)!=second.get(k) for k in metrics):
            raise ValueError(f"Calibration reproduction metrics/selection/budget differ: {key}")
        predictions=[read(r/"valid_predictions.json") for r in roots]
        if predictions[0]!=predictions[1] or any(digest(r/"valid_predictions.json")!=result["predictions_sha256"] for r,result in zip(roots,(first,second))):
            raise ValueError(f"Every per-sample probability/interval/point must match: {key}")
        if key[0]=="global":
            if read(roots[0]/"global_scale.json")!=read(roots[1]/"global_scale.json"):
                raise ValueError("Global scalar fit differs")
        else:
            histories=[[{k:v for k,v in row.items() if k!="seconds"} for row in read(r/"history.json")] for r in roots]
            if histories[0]!=histories[1]:
                raise ValueError("Calibration epoch objectives/update counts differ")
            states=[torch.load(r/"best.pt",map_location="cpu",weights_only=False,mmap=True) for r in roots]
            if any(digest(r/"best.pt")!=result["checkpoint_sha256"] for r,result in zip(roots,(first,second))):
                raise ValueError("Calibration checkpoint changed")
            if states[0]["model"].keys()!=states[1]["model"].keys() or any(not torch.equal(states[0]["model"][name],states[1]["model"][name]) for name in states[0]["model"]):
                raise ValueError("Calibration best model tensors differ")
        evidence.append({"case":key[0],"seed":key[1],"all_647_prediction_records_exact":True,
            "selected_parameters_objectives_metrics_and_budget_exact":True})
    return {"original_suite_sha256":digest(original),"reproduction_suite_sha256":digest(reproduction),
        "verified_runs":len(evidence),"evidence":evidence,"passed":True,"test_evaluated":False,
        "scope":"Calibration full retraining only; reused audited OOF tables, no point-teacher/encoder re-extraction claim"}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original",type=Path,required=True);p.add_argument("--reproduction",type=Path,required=True)
    p.add_argument("--original-base",type=Path,required=True);p.add_argument("--reproduction-base",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();report=compare(a.original,a.reproduction,a.original_base,a.reproduction_base)
    save_json(a.output,report);print(json.dumps({"passed":True,"verified_runs":report["verified_runs"]}))
