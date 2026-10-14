"""Export only bound descriptors/statistics for a native normalized point model."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_predict import load_model
from jevaffect.native_raw import read_contract
from jevaffect.native_training import digest,save_json


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--raw-source-cache",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise ValueError("New descriptor output required; refusing overwrite")
    spec=load_model(a.checkpoint,a.cache_dir,"cpu");contract=read_contract(a.cache_dir,spec,a.raw_source_cache)
    if contract["normalization"] is None:raise ValueError("This export supports normalized pooled native points")
    a.output_dir.mkdir(parents=True)
    for name in ("metadata.json","normalization.json"):
        shutil.copyfile(a.cache_dir/name,a.output_dir/name)
    raw=a.raw_source_cache/"metadata.json"
    if not raw.exists():raw=a.raw_source_cache/"meta.json"
    raw_out=a.output_dir/"raw_encoder_descriptor";raw_out.mkdir();shutil.copyfile(raw,raw_out/raw.name)
    exported=load_model(a.checkpoint,a.output_dir,"cpu");read_contract(a.output_dir,exported,raw_out)
    report={"checkpoint_sha256":digest(a.checkpoint),"metadata_sha256":spec["metadata_sha256"],
        "files":{path.relative_to(a.output_dir).as_posix():digest(path) for path in a.output_dir.rglob("*") if path.is_file()},
        "train_arrays_exported":False,"labels_exported":False,"weights_exported":False,
        "scope":"Local inference-only descriptors and TRAIN-derived normalization statistics; redistribution conditions remain subject to final review"}
    save_json(a.output_dir/"export_audit.json",report);print(json.dumps(report,indent=2))
