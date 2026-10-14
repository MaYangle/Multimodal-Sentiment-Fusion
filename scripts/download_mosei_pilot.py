"""Fetch a small, class-stratified real CMU-MOSEI train/valid media pilot.

This is a pipeline check, not a performance sample. It includes missing-video
records with separate WAV and supplied transcript. No test assets are read.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import random
import threading
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/mosei/pilot_media"
COMPETITION = "msa-challenge-at-the-4th-pazhou-ai-competition"
CLASSES = ("SNEG", "WNEG", "NEUT", "WPOS", "SPOS")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = json.loads((ROOT / "data/competition_metadata/files.json").read_text(encoding="utf-8"))
    sizes = {row["name"]: row["totalBytes"] for row in inventory["files"]}
    transcript = {split: json.loads((ROOT / f"data/mosei/competition/transcript_{split}.json").read_text(encoding="utf-8"))
                  for split in ("train", "valid")}
    selected = []
    rng = random.Random(13)
    for split in ("train", "valid"):
        rows = [json.loads(line) for line in (ROOT / f"data/mosei/manifests/mosei_{split}_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line]
        per_class: dict[tuple[str, bool], list[dict]] = defaultdict(list)
        for row in rows:
            if row["audio_path"] not in sizes or row["ID"] not in transcript[split] or not transcript[split][row["ID"]].strip():
                continue
            per_class[(row["class"], bool(row["video_present_in_inventory"]))].append(row)
        for cls in CLASSES:
            present = per_class[(cls, True)]
            missing = per_class[(cls, False)]
            rng.shuffle(present)
            rng.shuffle(missing)
            selected.extend(present[:10 if split == "train" else 4])
            selected.extend(missing[:1 if split == "train" else 1])
    if len({row["ID"] for row in selected}) != len(selected) or len(selected) < 65:
        raise ValueError("Pilot selection is unexpectedly small or non-unique")
    selection = {"scope": "real train/valid media flow only; not a performance benchmark",
                 "selection_seed": 13, "rows": [{"ID": row["ID"], "split": row["split"],
                                               "class": row["class"],
                                               "video_present": row["video_present_in_inventory"]}
                                              for row in selected], "test_assets_used": False}
    selection_path = OUT / "selection.json"
    if selection_path.exists():
        if json.loads(selection_path.read_text(encoding="utf-8")) != selection:
            raise ValueError("Existing pilot selection changed")
    else:
        atomic_json(selection_path, selection)
    jobs = []
    for row in selected:
        for modality, field in (("audio", "audio_path"), ("video", "video_path")):
            relative = row[field]
            if modality == "video" and not row["video_present_in_inventory"]:
                continue
            if relative not in sizes or ".." in Path(relative).parts:
                raise ValueError("Official media path missing/unsafe")
            target = OUT / row["split"] / modality / Path(relative).name
            jobs.append({"ID": row["ID"], "split": row["split"], "modality": modality,
                         "relative": relative, "target": target, "bytes": sizes[relative]})
    token_path = Path.home() / ".kaggle/access_token"
    if "KAGGLE_API_TOKEN" not in os.environ and token_path.is_file():
        os.environ["KAGGLE_API_TOKEN"] = token_path.read_text(encoding="utf-8").strip()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    if not api.config_values.get("username"):
        raise RuntimeError("Kaggle authentication has no account identity")
    original_send = requests.Session.send
    def bounded(self, request, **kwargs):
        kwargs["timeout"] = (30, 90)
        return original_send(self, request, **kwargs)
    requests.Session.send = bounded
    manifest_path = OUT / "download_manifest.json"
    prior = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    results = dict(prior.get("files", {}))
    lock = threading.Lock()
    def fetch(job: dict) -> dict:
        target = job["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        old = results.get(job["relative"])
        if old and target.is_file() and target.stat().st_size == job["bytes"] and sha(target) == old["sha256"]:
            return old
        for attempt in range(3):
            try:
                api.competition_download_file(COMPETITION, job["relative"], path=str(target.parent),
                                              force=target.exists(), quiet=True)
                if not target.is_file() or target.stat().st_size != job["bytes"]:
                    raise RuntimeError("Competition inventory size mismatch")
                return {"ID": job["ID"], "split": job["split"], "modality": job["modality"],
                        "relative": job["relative"], "bytes": job["bytes"], "sha256": sha(target)}
            except (requests.RequestException, RuntimeError):
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch, job): job for job in jobs}
        for future in concurrent.futures.as_completed(futures):
            item = future.result()
            with lock:
                results[item["relative"]] = item
                atomic_json(manifest_path, {"source": "authenticated competition file API",
                                            "selection_sha256": sha(selection_path),
                                            "files": results, "complete_count": len(results),
                                            "expected_count": len(jobs), "test_assets_used": False})
            print(json.dumps({"downloaded": len(results), "expected": len(jobs), "ID": item["ID"]}), flush=True)
    inputs = []
    for row in selected:
        inputs.append({"id": row["ID"], "split": row["split"],
                       "video_path": str(OUT / row["split"] / "video" / Path(row["video_path"]).name),
                       "audio_path": str(OUT / row["split"] / "audio" / Path(row["audio_path"]).name),
                       "allow_missing_video": not row["video_present_in_inventory"],
                       "transcript": transcript[row["split"]][row["ID"]]})
    (OUT / "input_records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in inputs), encoding="utf-8")
    print(json.dumps({"rows": len(inputs), "files": len(jobs),
                      "missing_video_rows": sum(row["allow_missing_video"] for row in inputs),
                      "selection_sha256": sha(selection_path), "test_assets_used": False}), flush=True)


if __name__ == "__main__":
    main()
