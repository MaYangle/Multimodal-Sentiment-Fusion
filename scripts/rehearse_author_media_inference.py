"""Rehearse label-free native inference on verified author TRAIN media."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256
from jevaffect.native_training import digest,save_json
from scripts.audit_standardized_media_inference import audit


def rehearse(author_csv, transfer_report, checkpoint, cache, macbert, model_root, output):
    output=Path(output).resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    if digest(author_csv)!=SOURCE_CSV_SHA256:
        raise ValueError("Pinned author CSV required")
    receipt=json.loads(Path(transfer_report).read_text(encoding="utf-8"))
    transfers=receipt["records"]
    if not transfers or any(r.get("mode")!="train" or not r.get("transfer_complete") or not r.get("decoded_cpu") or not r.get("sha256_matches_competition_source") for r in transfers):
        raise ValueError("Completed, decoded, source-matched author TRAIN media required")
    wanted={r["canonical_id"]:r for r in transfers}
    if len(wanted)!=len(transfers):
        raise ValueError("Duplicate author media ID")
    transcripts={}
    with Path(author_csv).open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            if row["mode"]!="train":
                continue
            sid=row["video_id"]+"/"+row["clip_id"]
            if sid in wanted:
                transcripts[sid]=row["text"]
                if len(transcripts)==len(wanted):
                    break
    if set(transcripts)!=set(wanted):
        raise ValueError("Author media IDs lack official TRAIN transcripts")
    inputs=[]
    for sid,record in wanted.items():
        path=Path(record["payload_path"]).resolve()
        if path.stat().st_size!=record["actual_payload_bytes"] or digest(path)!=record["sha256"] or record["sha256"]!=record["prior_competition_source_sha256"]:
            raise ValueError("Author media receipt/source identity mismatch")
        inputs.append({"id":sid,"video_path":str(path),"transcript":transcripts[sid]})
    # Verify the cached reference vectors before reusing the existing numerical
    # auditor. Its historical scope string is replaced only after this separate
    # author-source verification; the comparison and tolerances are unchanged.
    meta=json.loads((Path(cache)/"metadata.json").read_text())
    for name in ("ids.json","text.npy","audio.npy","vision.npy"):
        path=Path(cache)/"train"/name
        if digest(path)!=meta["artifact_sha256"]["train/"+name]:
            raise ValueError("Changed cached reference: "+name)
    output.mkdir(parents=True)
    source_jsonl=output/"input.jsonl"
    source_jsonl.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in inputs),encoding="utf-8")
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/"scripts/predict_native_media.py"),"--checkpoint",str(checkpoint),"--cache-dir",str(cache),"--macbert-dir",str(macbert),"--model-root",str(model_root),"--input-jsonl",str(source_jsonl),"--output-dir",str(output/"inference"),"--device","cpu"]
    completed=subprocess.run(command,check=False)
    if completed.returncode:
        raise RuntimeError("Author-media encoder/inference failed; partial artifacts preserved")
    report=audit(output/"inference",checkpoint,cache)
    report.update(scope="Verified author-original TRAIN MP4 and supplied-transcript CPU encoder/inference rehearsal; not held-out performance, full-corpus equivalence or final-method reproduction",
        author_transfer_report_sha256=digest(transfer_report),author_csv_sha256=digest(author_csv),
        rehearsal_implementation_sha256=digest(__file__),author_identity_verified_for=[r["canonical_id"] for r in transfers],
        sentiment_scores_supplied_to_inference=False,TEST_scores_parsed=False)
    save_json(output/"audit.json",report)
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ("author-csv","transfer-report","checkpoint","cache-dir","macbert-dir","model-root","output-dir"):
        p.add_argument("--"+flag,type=Path,required=True)
    a=p.parse_args()
    r=rehearse(a.author_csv,a.transfer_report,a.checkpoint,a.cache_dir,a.macbert_dir,a.model_root,a.output_dir)
    print(json.dumps(r,indent=2))
