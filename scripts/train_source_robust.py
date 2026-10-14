"""Train registered source-CV native GroupDRO/DANN controls serially."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.source_robust import CASES
from jevaffect.source_robust_training import run_suite

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--case",choices=CASES);p.add_argument("--seed",type=int,choices=(13,42,2026));p.add_argument("--fold",type=int,choices=(0,1,2))
    p.add_argument("--smoke-groups",type=int);p.add_argument("--stop-after",type=int);p.add_argument("--resume",action="store_true")
    a=p.parse_args()
    if a.smoke_groups is not None and a.smoke_groups<8:
        p.error("At least eight complete source groups required")
    if a.stop_after is not None and a.stop_after<1:
        p.error("Positive epoch limit required")
    cfg=json.loads(a.config.read_text());r=run_suite(cfg,a.output,a.case,a.seed,a.fold,a.smoke_groups,a.resume,a.stop_after)
    print(json.dumps({"suite_complete":r["suite_complete"],"runs":len(r["results"]),"smoke_only":r["smoke_only"]}))
