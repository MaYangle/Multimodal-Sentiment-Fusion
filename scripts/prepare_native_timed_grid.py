"""Make four decoder-time queries per retained A/V overlap, without encoding."""
from __future__ import annotations
import argparse
import json
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256
from jevaffect.native_training import NativeSplit, author_rows, digest, save_json

MODALITIES = ("text", "audio", "vision")
FORMAL_SHAPES = {"text": (64, 768), "audio": (400, 768), "vision": (55, 384)}


def interpolate(values, times, queries):
    """Linear interpolation on strictly increasing seconds; never extrapolate."""
    times = np.asarray(times, dtype=np.float64)
    queries = np.asarray(queries, dtype=np.float64)
    values = np.asarray(values)
    if values.ndim != 2 or times.shape != (len(values),) or len(times) < 2 or not np.isfinite(values).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Interpolation requires finite strictly increasing valid source rows")
    if queries.ndim != 1 or not np.isfinite(queries).all() or np.any(queries < times[0]) or np.any(queries > times[-1]):
        raise ValueError("Queries must lie within retained timestamps")
    right = np.clip(np.searchsorted(times, queries, side="right"), 1, len(times)-1)
    left = right - 1
    weight = (queries - times[left]) / (times[right] - times[left])
    return (values[left] * (1-weight[:, None]) + values[right] * weight[:, None]).astype(np.float32)


def paired_grid(audio, audio_times, vision, vision_times):
    """Invalid pairs return zero grids; bad valid timestamps are errors."""
    empty = (np.zeros((4, audio.shape[-1]), np.float32), np.zeros((4, vision.shape[-1]), np.float32), np.zeros(4, np.float64), False)
    for values, times in ((audio, audio_times), (vision, vision_times)):
        if values.ndim != 2 or np.asarray(times).shape != (len(values),) or not np.isfinite(values).all() or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Invalid valid-prefix sequence/timestamps")
    if len(audio) < 2 or len(vision) < 2:
        return empty
    low = max(float(audio_times[0]), float(vision_times[0]))
    high = min(float(audio_times[-1]), float(vision_times[-1]))
    if high <= low:
        return empty
    queries = np.linspace(low, high, 6, dtype=np.float64)[1:-1]
    return interpolate(audio, audio_times, queries), interpolate(vision, vision_times, queries), queries, True


def _verify(cache, meta, path):
    key = path.relative_to(cache.resolve()).as_posix()
    record = meta.get("artifact_sha256", {}).get(key, meta.get("files", {}).get(key))
    expected = record.get("sha256") if isinstance(record, dict) else record
    if expected is None or digest(path) != expected:
        raise ValueError("Missing/mismatched source SHA: " + key)
    if isinstance(record, dict) and record.get("size_bytes", path.stat().st_size) != path.stat().st_size:
        raise ValueError("Mismatched source size: " + key)
    return digest(path)


def prepare_cache(source, author_csv, output, allow_fixture=False):
    source, author_csv, output = map(lambda p: Path(p).resolve(), (source, author_csv, output))
    if output.exists():
        raise ValueError("Use a new output directory; refusing overwrite")
    csv_sha = digest(author_csv)
    if not allow_fixture and csv_sha != SOURCE_CSV_SHA256:
        raise ValueError("Pinned author CSV required")
    rows = author_rows(author_csv, "simsv2")
    meta = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    if meta.get("official_csv_sha256", meta.get("source_descriptor", {}).get("official_csv_sha256")) != csv_sha:
        raise ValueError("Raw cache author CSV provenance mismatch")
    datasets = {}
    opened = []
    metadata = {"format": "native-timed-grid-v1", "dataset": "simsv2", "test_exported": False,
        "author_csv_sha256": csv_sha, "preparation_sha256": digest(__file__),
        "native_reader_sha256": digest(Path(__file__).resolve().parents[1]/"jevaffect/native_training.py"),
        "source_cache": str(source), "source_descriptor": meta.get("source_descriptor"),
        "source_truncation": meta.get("truncation"), "source_feature_sha256": {}, "counts": {},
        "input_form": "Provided-transcript MacBERT CLS and frozen WavLM/DINOv2 features; decoder-reported-time interpolation, no new encoding",
        "source_form": "author-ID-matched competition-acquired cache; author-original media byte equivalence unverified",
        "time_contract": {"queries": "linspace(max(first A,V), min(last retained A,V), 6)[1:-1]",
            "interpolation": "linear in retained seconds; no extrapolation or index stretching",
            "pair_eligibility": "at least two valid rows in each stream and positive reported-time overlap",
            "audio_origin": "relative to independently decoded waveform; first source audio PTS not retained",
            "vision_origin": "ffmpeg fps=2:start_time=0 output PTS",
            "corpus_physical_synchrony": "UNVERIFIED", "wavlm_context": "final transformer states; time anchors do not imply local-only content"},
        "coverage": {}, "artifact_sha256": {}}
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix=".native-time-grid-", dir=output.parent) as temporary:
            stage = Path(temporary)
            for split in ("train", "valid"):
                ds = NativeSplit(source, split, rows, MODALITIES, pooled=False)
                datasets[split] = ds
                if not allow_fixture and (len(ds) != {"train":2722,"valid":647}[split] or {m: x.shape[1:] for m,x in ds.features.items()} != FORMAL_SHAPES):
                    raise ValueError("Complete native TRAIN/VALID and pinned sequence shapes required")
                raw_dir = source / meta.get("split_paths", {}).get(split, "splits/"+split if (source/"splits").exists() else split)
                raw_dir = raw_dir.resolve()
                raw_dir.relative_to(source)
                extra_hashes = {"ids.json": _verify(source, meta, raw_dir/"ids.json")}
                if (raw_dir/"raw_labels.json").is_file():
                    extra_hashes["raw_labels.json"] = _verify(source, meta, raw_dir/"raw_labels.json")
                times = {}
                for modality in ("audio", "vision"):
                    path = raw_dir/(modality+"_times.npy")
                    extra_hashes[path.name] = _verify(source, meta, path)
                    times[modality] = np.load(path, mmap_mode="r", allow_pickle=False)
                    opened.append(times[modality])
                    if times[modality].shape != ds.features[modality].shape[:2]:
                        raise ValueError("Timestamp shape differs from source sequence")
                n = len(ds)
                presence = np.stack([np.asarray(ds.lengths[m])>0 for m in MODALITIES], axis=1)
                arrays = {"text": np.zeros((n, ds.features["text"].shape[-1]), np.float32),
                    "audio_mean": np.zeros((n, ds.features["audio"].shape[-1]), np.float32),
                    "vision_mean": np.zeros((n, ds.features["vision"].shape[-1]), np.float32),
                    "audio_grid": np.zeros((n,4,ds.features["audio"].shape[-1]), np.float32),
                    "vision_grid": np.zeros((n,4,ds.features["vision"].shape[-1]), np.float32),
                    "pair_valid": np.zeros(n, bool), "presence": presence, "grid_times": np.zeros((n,4), np.float64)}
                overlap_spans = []
                for i in range(n):
                    lengths = {m: int(ds.lengths[m][i]) for m in MODALITIES}
                    if lengths["text"]:
                        arrays["text"][i] = ds.features["text"][i,0]
                    for m in ("audio", "vision"):
                        if lengths[m]:
                            arrays[m+"_mean"][i] = ds.features[m][i,:lengths[m]].mean(0)
                    a = np.asarray(ds.features["audio"][i,:lengths["audio"]])
                    v = np.asarray(ds.features["vision"][i,:lengths["vision"]])
                    at = np.asarray(times["audio"][i,:lengths["audio"]])
                    vt = np.asarray(times["vision"][i,:lengths["vision"]])
                    ag, vg, q, valid = paired_grid(a, at, v, vt)
                    arrays["audio_grid"][i], arrays["vision_grid"][i], arrays["grid_times"][i], arrays["pair_valid"][i] = ag, vg, q, valid
                    if valid:
                        overlap_spans.append(min(at[-1],vt[-1])-max(at[0],vt[0]))
                directory = stage/split
                directory.mkdir()
                save_json(directory/"ids.json", ds.ids)
                save_json(directory/"raw_labels.json", ds.targets.tolist())
                for name, array in arrays.items():
                    if not np.isfinite(array).all():
                        raise ValueError("Nonfinite output: " + name)
                    np.save(directory/(name+".npy"), array)
                metadata["counts"][split] = n
                metadata["source_feature_sha256"][split] = {**ds.hashes, **extra_hashes}
                metadata["coverage"][split] = {"valid_pairs": int(arrays["pair_valid"].sum()), "invalid_pairs": int((~arrays["pair_valid"]).sum()),
                    "retained_overlap_span_quantiles_seconds": np.quantile(overlap_spans,[0,.25,.5,.75,1]).tolist() if overlap_spans else [],
                    "queries_per_valid_pair": 4, "prefix_truncation_recovered": False}
            if set(datasets["train"].ids) & set(datasets["valid"].ids):
                raise ValueError("TRAIN/VALID ID overlap")
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    metadata["artifact_sha256"][path.relative_to(stage).as_posix()] = digest(path)
            save_json(stage/"metadata.json", metadata)
            stage.rename(output)
    finally:
        for ds in datasets.values():
            opened.extend((*ds.features.values(), *ds.lengths.values()))
        for array in opened:
            mmap = getattr(array,"_mmap",None)
            if mmap is not None:
                mmap.close()
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir",type=Path,required=True)
    p.add_argument("--author-csv",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    a = p.parse_args()
    result = prepare_cache(a.cache_dir,a.author_csv,a.output_dir)
    print(json.dumps({"counts":result["counts"],"coverage":result["coverage"],"artifacts":len(result["artifact_sha256"]),"test_exported":False},indent=2))
