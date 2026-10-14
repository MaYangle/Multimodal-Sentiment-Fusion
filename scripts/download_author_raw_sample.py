"""Exact author-ID video retrieval, without parsing any sentiment labels."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--meta-csv", type=Path, required=True)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--split", choices=("train", "valid", "test"), required=True)
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    if a.limit < 1:
        raise ValueError("Sample limit must be positive")
    with a.meta_csv.open(encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == a.split][:a.limit]
    files = json.loads(a.inventory.read_text())
    lookup = {r["path"].replace("\\", "/").removeprefix("Raw/").removesuffix(".mp4"): r
              for r in files if r["path"].replace("\\", "/").startswith("Raw/") and r["path"].endswith(".mp4")}
    a.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"split": a.split, "selection": "first requested CSV rows; not representative", "sentiment_labels_parsed": False,
              "test_scored": False, "csv_sha256": hashlib.sha256(a.meta_csv.read_bytes()).hexdigest(),
              "inventory_sha256": hashlib.sha256(a.inventory.read_bytes()).hexdigest(), "records": []}
    for row in rows:
        sid = row["video_id"] + "/" + row["clip_id"]
        item = lookup.get(sid)
        if item is None:
            raise ValueError("Author raw directory lacks exact sample ID")
        directory = (a.output_dir / row["video_id"]).resolve()
        directory.relative_to(a.output_dir.resolve())
        if Path(row["clip_id"]).name != row["clip_id"]:
            raise ValueError("Unsafe author clip ID")
        directory.mkdir(parents=True, exist_ok=True)
        name = row["clip_id"] + ".mp4"
        source = {"publisher": "https://github.com/thuiar/ch-sims-v2", "folder_id": "1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA",
                  "input_form": "author original MP4 media", "files": [{"id": item["id"], "path": name,
                                                                            "published_path": item["path"], "canonical_id": sid}]}
        inventory_path = directory / (row["clip_id"] + ".source.json")
        inventory_path.write_text(json.dumps(source, indent=2))
        command = [sys.executable, str(Path(__file__).with_name("download_author_payload.py")), "--inventory", str(inventory_path),
                   "--name", name, "--output-dir", str(directory), "--direct", "--read-timeout", "60", "--retries", "1"]
        result = subprocess.run(command)
        report["records"].append({"id": sid, "publisher_file_id": item["id"], "exit_code": result.returncode,
                                  "path": str(directory / name), "transfer_complete": result.returncode == 0})
        (a.output_dir / ("sample_" + a.split + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return 0 if all(r["transfer_complete"] for r in report["records"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
