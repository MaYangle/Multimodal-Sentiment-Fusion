"""Fetch author-linked supervised metadata without touching existing caches."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

SOURCES = {
    "simsv2": {
        "file_id": "1OINbBnCat988BNPV0RnS5iphRlHd6SW2",
        "publisher": "https://github.com/thuiar/ch-sims-v2",
        "folder": "https://drive.google.com/drive/folders/1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA",
        "expected_sha256": "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150",
    },
    "mosei": {
        "file_id": "17_IhQLXHqON78XX5i0h9c0ywXXpt60PX",
        "publisher": "https://github.com/thuiar/MMSA",
        "folder": "https://drive.google.com/drive/folders/1A2S4pqCHryGmiqnNSPLv7rEg63WvjCSk",
        "expected_sha256": "52d885bd1e8b29a4c06a69e2b1dd67dd02f7bcbdfa38ad918144fd5673a83521",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cached-csv", type=Path, help="Reuse previously downloaded exact author bytes")
    args = parser.parse_args()
    import gdown
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "meta.csv"
    source = SOURCES[args.dataset]
    if args.cached_csv is not None and not path.exists():
        if hashlib.sha256(args.cached_csv.read_bytes()).hexdigest() != source["expected_sha256"]:
            raise ValueError("Cached metadata differs from pinned author file")
        shutil.copyfile(args.cached_csv, path)
    if not path.exists():
        result = gdown.download(id=source["file_id"], output=str(path),
                                use_cookies=False, resume=True, timeout=40,
                                retries=1, quiet=True)
        if result is None:
            raise RuntimeError("Author metadata download did not finish")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != source["expected_sha256"]:
        raise ValueError("Metadata version/hash changed; review author publication before accepting")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"video_id", "clip_id", "label", "mode"} <= rows[0].keys():
        raise ValueError("Unexpected author CSV schema")
    ids = [(r["video_id"], r["clip_id"]) for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate author sample IDs")
    report = {"dataset": args.dataset, **source, "bytes": path.stat().st_size,
              "sha256": actual, "cached_reuse": args.cached_csv is not None,
              "rows": len(rows), "split_counts": dict(Counter(r["mode"] for r in rows)),
              "columns": list(rows[0]), "input_form": "author continuous annotations",
              "test_scored": False}
    (args.output_dir / "provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
