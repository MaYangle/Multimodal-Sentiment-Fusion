"""Build label-free, pinned CMU-MOSEI train/valid extraction inputs.

Uses the challenge's frozen ID/media inventory and official transcript JSON.
Only a source manifest reaches the private Kaggle feature extractor; no labels,
test records, or author-ID renumbering hypothesis are packaged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mosei"


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    inventory_path = ROOT / "data/competition_metadata/files.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    files = {row["name"]: int(row["totalBytes"]) for row in inventory["files"]}
    records = []
    inputs = {}
    for split, count in (("train", 16327), ("valid", 1871)):
        manifest_path = DATA / "manifests" / f"mosei_{split}_manifest.jsonl"
        transcript_path = DATA / "competition" / f"transcript_{split}.json"
        labels = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line]
        transcripts = json.loads(transcript_path.read_text(encoding="utf-8"))
        if len(labels) != count:
            raise ValueError(split + " label-manifest count changed")
        inputs[split] = {"label_manifest_sha256": sha(manifest_path),
                         "transcript_json_sha256": sha(transcript_path),
                         "rows": count}
        for row in labels:
            sample_id = row["ID"]
            if row["split"] != split or sample_id not in transcripts:
                raise ValueError(split + " transcript/ID mismatch")
            audio_path = row["audio_path"]
            video_path = row["video_path"]
            if audio_path not in files or row["audio_present_in_inventory"] is not True:
                raise ValueError("Training/validation audio is missing: " + sample_id)
            present = bool(row["video_present_in_inventory"])
            if present != (video_path in files):
                raise ValueError("Video presence differs from authenticated inventory")
            for relative in (audio_path, video_path):
                parts = PurePosixPath(relative)
                if (parts.is_absolute() or ".." in parts.parts or "\\" in relative or
                        not relative.startswith("CMU-MOSEI/cmumosei/" + split + "/")):
                    raise ValueError("Unsafe competition media path")
            text = transcripts[sample_id]
            if not isinstance(text, str):
                raise ValueError("Transcript is not a string")
            records.append({"id": sample_id, "split": split,
                            "video_relative_path": video_path,
                            "audio_relative_path": audio_path,
                            "video_present": present,
                            "video_expected_bytes": files.get(video_path),
                            "audio_expected_bytes": files[audio_path],
                            "transcript": text,
                            "transcript_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    if len(records) != 18198 or len({row["id"] for row in records}) != 18198:
        raise ValueError("Frozen train/valid cohort is not 18,198 unique IDs")
    output = DATA / "cloud_input_records.jsonl"
    payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)
    if output.exists() and output.read_text(encoding="utf-8") != payload:
        raise ValueError("Existing label-free cloud input manifest differs")
    output.write_text(payload, encoding="utf-8")
    report = {"scope": "label-free official CMU-MOSEI train/valid cloud extraction inputs",
              "input_manifest_sha256": sha(output), "input_count": len(records),
              "split_counts": {split: sum(row["split"] == split for row in records) for split in ("train", "valid")},
              "video_missing_count": sum(not row["video_present"] for row in records),
              "audio_missing_count": 0,
              "video_bytes_if_present": sum(row["video_expected_bytes"] or 0 for row in records),
              "audio_bytes": sum(row["audio_expected_bytes"] for row in records),
              "inventory_sha256": sha(inventory_path), "source_inputs": inputs,
              "competition_labels_passed_to_feature_extractor": False,
              "test_records_passed_to_feature_extractor": False}
    document = ROOT / "docs/mosei_cloud_input_preflight.json"
    document.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
