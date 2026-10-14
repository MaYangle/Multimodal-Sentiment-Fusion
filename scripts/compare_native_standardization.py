"""Paired official-VALID development comparison for raw vs standardized CH."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import digest,save_json
from scripts.audit_source_text_capacity import bootstrap


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-suite",type=Path,required=True);p.add_argument("--normalized-suite",type=Path,required=True)
    p.add_argument("--raw-run-root",type=Path,required=True);p.add_argument("--normalized-run-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=("text","early_fusion","almt"),default=("text","early_fusion"))
    p.add_argument("--reference-model",choices=("text","early_fusion","almt"),help="Optional different model in the first suite")
    a=p.parse_args();suites=[json.loads(path.read_text()) for path in (a.raw_suite,a.normalized_suite)]
    if not all(s["suite_complete"] for s in suites):raise ValueError("Complete three-seed native suites required")
    comparisons={}
    for model in a.models:
        differences=[];canonical=None;results_by_form=[]
        member_models=(a.reference_model or model,model)
        for suite,member_model in zip(suites,member_models):
            rows={r["seed"]:r for r in suite["results"] if r["model"]==member_model}
            if set(rows)!={13,42,2026}:raise ValueError("Three complete model seeds required")
            results_by_form.append(rows)
        for seed in (13,42,2026):
            records=[]
            for base,results,member_model in zip((a.raw_run_root,a.normalized_run_root),results_by_form,member_models):
                path=base/member_model/f"seed_{seed}"/"valid_predictions.json"
                if digest(path)!=results[seed]["predictions_sha256"]:raise ValueError("Frozen native prediction changed")
                records.append(json.loads(path.read_text()))
            ids=[r["id"] for r in records[0]]
            if len(ids)!=647 or ids!=[r["id"] for r in records[1]] or (canonical is not None and ids!=canonical):raise ValueError("Native comparison cohort/order mismatch")
            canonical=ids
            y=np.array([r["target"] for r in records[0]],np.float32)
            if not np.array_equal(y,np.array([r["target"] for r in records[1]],np.float32)):raise ValueError("Native comparison target mismatch")
            errors=[np.abs(np.clip(np.array([r["prediction"] for r in rr],np.float32),-1,1)-y) for rr in records]
            differences.append(errors[1]-errors[0])
        comparisons[model]=bootstrap(np.stack(differences),canonical)
        comparisons[model]["scope"]="Conditional official VALID whole-source resampling of already-selected models; no selection-adjusted or final-test significance claim"
        comparisons[model]["reference_model"],comparisons[model]["candidate_model"]=member_models
    report={"raw_suite_sha256":digest(a.raw_suite),"normalized_suite_sha256":digest(a.normalized_suite),"comparisons":comparisons,
        "test_evaluated":False,"scope":"Official VALID development/selection comparison; conditional source-bootstrap intervals do not account for model selection or repeated development use"}
    save_json(a.output,report);print(json.dumps(comparisons,indent=2))
