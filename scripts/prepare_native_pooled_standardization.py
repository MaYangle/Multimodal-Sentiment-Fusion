"""Prepare a new native CH pooled-coordinate standardized cache, TRAIN-fit only."""
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.native_training import NativeSplit,author_rows,digest,save_json
from jevaffect.anchor_calibration import normalizer,normalize
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256


def prepare_cache(source,author_csv,output,allow_fixture=False):
    source,author_csv,output=map(Path,(source,author_csv,output))
    if output.exists():raise ValueError("Use a new output directory; refusing overwrite")
    if not allow_fixture and digest(author_csv)!=SOURCE_CSV_SHA256:raise ValueError("Pinned author CSV required")
    rows=author_rows(author_csv,"simsv2");mods=("text","audio","vision")
    datasets={s:NativeSplit(source,s,rows,mods,pooled=True) for s in ("train","valid")}
    if not allow_fixture and (len(datasets["train"])!=2722 or len(datasets["valid"])!=647):raise ValueError("Complete native TRAIN/VALID required")
    train=datasets["train"]
    stats={m:normalizer(x,np.arange(len(train))) for m,x in train.features.items()}
    metadata={"dataset":"simsv2","input_form":"Project-derived pinned MacBERT CLS/WavLM masked-mean/DINOv2 masked-mean vectors; TRAIN-only coordinate standardization; stored as one-step sequences",
        "source_form":"author-ID-matched competition-acquired frozen-encoder cache; full author-original Chinese media equivalence unverified",
        "source_cache":str(source.resolve()),"source_feature_sha256":{s:d.hashes for s,d in datasets.items()},
        "author_csv_sha256":digest(author_csv),"preparation_sha256":digest(__file__),"normalization_implementation_sha256":digest(Path(__file__).resolve().parents[1]/"jevaffect/anchor_calibration.py"),
        "normalization":{"kind":"TRAIN clip-pooled coordinate mean/population STD; tiny STD<1e-6 uses1","fit_split":"train","fit_count":len(train)},
        "counts":{s:len(d) for s,d in datasets.items()},"dimensions":{m:x.shape[-1] for m,x in train.features.items()},
        "artifact_sha256":{},"test_exported":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-normalized-",dir=output.parent) as temporary:
        stage=Path(temporary)
        save_json(stage/"normalization.json",stats)
        for split,ds in datasets.items():
            directory=stage/split;directory.mkdir()
            save_json(directory/"ids.json",ds.ids);save_json(directory/"raw_labels.json",ds.targets.tolist())
            for modality in mods:
                values=normalize(ds.features[modality],stats[modality])
                present=np.asarray(ds.lengths[modality])>0
                values[~present]=0
                np.save(directory/f"{modality}.npy",values[:,None,:])
                np.save(directory/f"{modality}_lengths.npy",present.astype(np.int64))
        for path in stage.rglob("*"):
            if path.is_file():metadata["artifact_sha256"][path.relative_to(stage).as_posix()]=digest(path)
        metadata["normalization"]["fit_ids_sha256"]=metadata["artifact_sha256"]["train/ids.json"]
        save_json(stage/"metadata.json",metadata);stage.rename(output)
    for ds in datasets.values():ds.lengths.clear()
    return metadata


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cache-dir",type=Path,required=True);p.add_argument("--author-csv",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args();m=prepare_cache(a.cache_dir,a.author_csv,a.output_dir)
    print(json.dumps({"counts":m["counts"],"dimensions":m["dimensions"],"artifacts":len(m["artifact_sha256"]),"test_exported":False},indent=2))
