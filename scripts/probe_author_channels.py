"""Probe public SDK endpoints and validate MOSEI source-video fold membership."""
from __future__ import annotations
import ast
import csv
import hashlib
import json
from pathlib import Path
import requests


def main():
    report = {"channels": [], "paid_resources": False, "raw_media_downloaded": False}
    repo = "CMU-MultiComp-Lab/CMU-MultimodalSDK"
    branch = requests.get(f"https://api.github.com/repos/{repo}/commits/main", timeout=30)
    branch.raise_for_status()
    revision = branch.json()["sha"]
    recipe_file = "mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei.py"
    recipe_url = f"https://raw.githubusercontent.com/{repo}/{revision}/{recipe_file}"
    recipe = requests.get(recipe_url, timeout=30)
    recipe.raise_for_status()
    published = {}
    for node in ast.parse(recipe.text).body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    published[target.slice.value] = node.value.value
    urls = [published[name] for name in ("All Labels", "words", "glove_vectors", "COVAREP", "OpenFace_2")]
    report["sdk_recipe"] = {"revision": revision, "source_url": recipe_url,
                             "source_sha256": hashlib.sha256(recipe.content).hexdigest(), "published_urls": urls}
    output = Path("docs/author_channel_probe.json")
    previous = json.loads(output.read_text()) if output.exists() else {}
    valid_urls = set(urls + [url.replace("http:", "https:", 1) for url in urls])
    prior = {r["url"]: r for r in previous.get("channels", [])}
    report["rejected_previous_paths"] = [url for url in prior if url not in valid_urls]
    for protocol in ("https", "http"):
        for published_url in urls:
            url = published_url.replace("http:", protocol + ":", 1)
            if url in prior:
                report["channels"].append(prior[url])
                continue
            row = {"url": url}
            try:
                with requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=(8, 12)) as response:
                    row.update(status=response.status_code, content_type=response.headers.get("Content-Type"),
                               content_length=response.headers.get("Content-Length"), content_range=response.headers.get("Content-Range"))
                    prefix = next(response.iter_content(1024), b"")
                    row["prefix_bytes"] = len(prefix)
                    row["hdf5_header"] = prefix[:8] == b"\x89HDF\r\n\x1a\n"
            except requests.RequestException as exc:
                row["error"] = type(exc).__name__
            report["channels"].append(row)
            print(json.dumps(row), flush=True)
    file = "mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py"
    url = f"https://raw.githubusercontent.com/{repo}/{revision}/{file}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    tree = ast.parse(response.text)
    folds = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("standard_"):
                    folds[target.id] = set(ast.literal_eval(node.value))
    mapping = {"train": "standard_train_fold", "valid": "standard_valid_fold", "test": "standard_test_fold"}
    with Path("data/mosei/mmsa_label.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    errors = [{"video_id": r["video_id"], "split": r["mode"]} for r in rows if r["video_id"] not in folds[mapping[r["mode"]]]]
    report["mosei_folds"] = {"sdk_revision": revision, "source_url": url, "source_sha256": hashlib.sha256(response.content).hexdigest(),
                             "rows_checked": len(rows), "membership_errors": len(errors), "first_errors": errors[:10],
                             "cohort_csv_sha256": hashlib.sha256(Path("data/mosei/mmsa_label.csv").read_bytes()).hexdigest()}
    Path("docs/author_channel_probe.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
