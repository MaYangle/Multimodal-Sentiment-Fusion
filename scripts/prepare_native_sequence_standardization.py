"""Apply the verified pooled TRAIN scaler to valid tokens in native sequences."""
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_crossfit import SOURCE_CSV_SHA256
from jevaffect.native_training import NativeSplit, author_rows, digest, save_json
from jevaffect.raw_features import ENCODER_PINS


MODALITIES = ("text", "audio", "vision")
TIME_ARTIFACTS = ("text_times.npy", "audio_times.npy", "vision_times.npy")
NORMALIZATION_KIND = "TRAIN clip-pooled coordinate mean/population STD; tiny STD<1e-6 uses1"
POOLING_TOLERANCE = 1e-4
FORMAL_COUNTS = {"train": 2722, "valid": 647}
FORMAL_SHAPES = {"text": (64, 768), "audio": (400, 768), "vision": (55, 384)}
PREPROCESSING = {
    "audio_sample_rate": 16000,
    "max_seconds": 120.0,
    "mbert_max_tokens": 64,
    "video_fps": 2.0,
    "video_frame_size": 224,
    "video_frame_timestamps_kind": "ffmpeg_resampled_output_timeline",
    "audio_feature_times": "centers of WavLM convolutional receptive fields",
}


def _metadata_path(cache):
    path = Path(cache) / "metadata.json"
    if not path.is_file():
        raise ValueError("Cache requires metadata.json: " + str(cache))
    return path


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Could not read cache metadata: " + str(path)) from exc


def _split_directory(cache, metadata, split):
    cache = Path(cache).resolve()
    default = "splits/" + split if (cache / "splits").is_dir() else split
    directory = (cache / metadata.get("split_paths", {}).get(split, default)).resolve()
    try:
        directory.relative_to(cache)
    except ValueError as exc:
        raise ValueError("Cache split path escapes its root") from exc
    return directory


def _normalized_key(key):
    return str(key).replace("\\", "/").rstrip("/").lower()


def _dataset_hash(dataset, suffix, by_name=False):
    target = _normalized_key(suffix)
    matches = []
    for key, value in dataset.hashes.items():
        normalized = _normalized_key(key)
        if (normalized.rsplit("/", 1)[-1] == target if by_name else normalized.endswith("/" + target) or normalized == target):
            matches.append(value)
    if len(matches) != 1:
        raise ValueError("Native cache is missing an unambiguous source hash: " + suffix)
    return matches[0]


def _source_feature_hashes(dataset, split):
    result = {
        "metadata.json": _dataset_hash(dataset, "metadata.json", by_name=True),
        "ids.json": _dataset_hash(dataset, split + "/ids.json"),
    }
    for modality in MODALITIES:
        result[modality + ".npy"] = _dataset_hash(dataset, split + "/" + modality + ".npy")
        result[modality + "_lengths.npy"] = _dataset_hash(dataset, split + "/" + modality + "_lengths.npy")
    return result


def _recorded_source_hashes(record, split):
    if not isinstance(record, dict):
        raise ValueError("Pooled cache is missing raw-source hashes for " + split)
    result = {}
    for name in ("metadata.json", "ids.json", *(m + ".npy" for m in MODALITIES),
                 *(m + "_lengths.npy" for m in MODALITIES)):
        matches = []
        for key, value in record.items():
            normalized = _normalized_key(key)
            if name == "metadata.json":
                match = normalized.rsplit("/", 1)[-1] == name
            elif name == "ids.json":
                match = normalized.endswith("/" + split + "/ids.json")
            else:
                match = normalized.endswith("/" + split + "/" + name)
            if match:
                matches.append(value)
        if len(matches) != 1:
            raise ValueError("Pooled cache has a missing or ambiguous raw-source hash: " + split + "/" + name)
        result[name] = matches[0]
    if len(record) != len(result):
        raise ValueError("Pooled cache raw-source hash inventory is unexpected")
    return result


def _verify_recorded_artifact(cache, metadata, relative):
    key = Path(relative).as_posix()
    expected = metadata.get("artifact_sha256", {}).get(key)
    path = Path(cache) / Path(relative)
    if not expected or not path.is_file() or digest(path) != expected:
        raise ValueError("Missing or mismatched cached artifact: " + key)
    return path


def _timestamp_records(metadata, relative):
    """Collect any raw-cache files/artifact SHA declarations for one time array."""
    target = _normalized_key(relative)
    records = []
    for container_name in ("files", "artifact_sha256"):
        container = metadata.get(container_name, {})
        if not isinstance(container, dict):
            continue
        matches = []
        for key, value in container.items():
            normalized = _normalized_key(key)
            if normalized == target or normalized.endswith("/" + target):
                matches.append(value)
        if len(matches) > 1:
            raise ValueError("Raw timestamp metadata has ambiguous entries: " + relative)
        if matches:
            value = matches[0]
            if isinstance(value, dict):
                records.append((value.get("sha256"), value.get("size_bytes")))
            else:
                records.append((value, None))
    return records


def _discover_timestamp_artifacts(cache, metadata, split):
    directory = _split_directory(cache, metadata, split)
    result = {}
    for name in TIME_ARTIFACTS:
        path = directory / name
        relative = path.relative_to(Path(cache).resolve()).as_posix()
        records = _timestamp_records(metadata, relative)
        if not path.exists() and not records:
            continue
        if not path.is_file() or not records:
            raise ValueError("Raw timestamp artifact is missing or unbound: " + relative)
        actual_sha = digest(path)
        actual_size = path.stat().st_size
        for expected_sha, expected_size in records:
            if expected_sha != actual_sha or (expected_size is not None and expected_size != actual_size):
                raise ValueError("Raw timestamp artifact failed metadata SHA/size validation: " + relative)
        result[name] = {
            "source_relative_path": relative,
            "source_sha256": actual_sha,
            "source_size_bytes": actual_size,
            "output_relative_path": split + "/" + name,
            "output_sha256": actual_sha,
        }
    return result


def _close_dataset(dataset):
    for value in (*dataset.features.values(), *dataset.lengths.values()):
        mmap = getattr(value, "_mmap", None)
        if mmap is not None:
            mmap.close()
    dataset.features.clear()
    dataset.lengths.clear()


def _raw_contract(metadata, actual_shapes, allow_fixture):
    descriptor = metadata.get("source_descriptor")
    if not isinstance(descriptor, dict) or descriptor.get("text_condition") != "provided transcripts, not ASR":
        raise ValueError("Raw cache must bind the provided-transcript MacBERT condition")
    base = descriptor.get("base_raw_cache_descriptor", {})
    if base.get("encoder_preprocessing") != PREPROCESSING or base.get("truncation_policy") != "keep first target steps in original order; no resampling/interpolation":
        raise ValueError("Unsupported raw encoder preprocessing or truncation contract")
    target_shapes = {key: tuple(value) for key, value in base.get("target_shapes", {}).items()}
    if target_shapes != actual_shapes:
        raise ValueError("Raw sequence shapes differ from their source descriptor")
    encoder = descriptor.get("replacement_text_encoder", {})
    if (encoder.get("model") != "hfl/chinese-macbert-base"
            or encoder.get("revision") != "a986e004d2a7f2a1c2f5a3edef4e20604a974ed1"
            or encoder.get("files", {}).get("pytorch_model.bin", {}).get("sha256") != "db0506d985574b80c33eec1cf13bd4c130585568753175871095ca13dbad9e23"):
        raise ValueError("Raw cache does not bind the pinned MacBERT sequence representation")
    if descriptor.get("test_data_encoded") is not False:
        raise ValueError("Raw source metadata does not exclude TEST encodings")
    encoders = base.get("encoder_provenance", {}).get("encoders", {})
    for alias in ("wavlm", "dinov2-small"):
        pin = ENCODER_PINS[alias]
        actual = encoders.get(alias, {})
        if (actual.get("repo") != pin["repo"] or actual.get("revision") != pin["revision"]
                or actual.get("weight_sha256") != pin["sha256"]):
            raise ValueError("Raw cache does not bind the pinned " + alias + " representation")
    if not allow_fixture and actual_shapes != FORMAL_SHAPES:
        raise ValueError("Formal raw cache must have the native T64/A400/V55 shapes")
    return descriptor


def _validate_stats(statistics, metadata, pooled_cache, train_count, dimensions):
    norm = metadata.get("normalization", {})
    if (norm.get("kind") != NORMALIZATION_KIND or norm.get("fit_split") != "train"
            or norm.get("fit_count") != train_count or metadata.get("test_exported") is not False):
        raise ValueError("Pooled cache is not the verified TRAIN-only normalization artifact")
    if set(statistics) != set(MODALITIES):
        raise ValueError("Normalization statistics have an unexpected modality set")
    train_ids = _verify_recorded_artifact(pooled_cache, metadata, "train/ids.json")
    if norm.get("fit_ids_sha256") != digest(train_ids):
        raise ValueError("Normalization fit IDs do not match pooled TRAIN IDs")
    for modality in MODALITIES:
        row = statistics[modality]
        if set(row) != {"mean", "scale", "fit_indices"}:
            raise ValueError("Unexpected saved scaler schema for " + modality)
        mean = np.asarray(row["mean"], dtype=np.float32)
        scale = np.asarray(row["scale"], dtype=np.float32)
        indices = row["fit_indices"]
        if (mean.shape != (dimensions[modality],) or scale.shape != mean.shape
                or not np.isfinite(mean).all() or not np.isfinite(scale).all()
                or np.any(scale <= 0) or indices != list(range(train_count))):
            raise ValueError("Invalid TRAIN-only saved scaler for " + modality)


def _check_complete_existing(output, expected_bindings):
    output = Path(output)
    metadata_path = output / "metadata.json"
    if not output.is_dir() or not metadata_path.is_file():
        raise ValueError("Output path exists but is not a complete matching cache; refusing overwrite")
    metadata = _load_json(metadata_path)
    for key, value in expected_bindings.items():
        if metadata.get(key) != value:
            raise ValueError("Existing output cache does not match this source: " + key)
    artifacts = metadata.get("artifact_sha256")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Existing output cache has no artifact manifest")
    actual = {}
    for path in output.rglob("*"):
        if path.is_file() and path != metadata_path:
            actual[path.relative_to(output).as_posix()] = digest(path)
    if actual != artifacts:
        raise ValueError("Existing output cache is incomplete, modified, or contains unbound files")
    if metadata.get("test_exported") is not False or any(name.startswith("test/") for name in actual):
        raise ValueError("Existing output cache violates the no-TEST-export contract")
    result = dict(metadata)
    result["cache_reused"] = True
    return result


def _transform_split(raw, pooled, raw_cache, split, stage, statistics):
    raw_directory = _split_directory(raw_cache, raw.meta, split)
    output_directory = Path(stage) / split
    output_directory.mkdir(parents=True)
    shutil.copyfile(raw_directory / "ids.json", output_directory / "ids.json")
    shutil.copyfile(raw_directory / "raw_labels.json", output_directory / "raw_labels.json")

    max_errors = {}
    padded_entries = {}
    nonzero_padding = {}
    for modality in MODALITIES:
        x = raw.features[modality]
        lengths = raw.lengths[modality]
        if x.dtype != np.float32 or x.ndim != 3:
            raise ValueError("Raw sequence features must be float32 [N,T,D]")
        n_rows, steps, width = x.shape
        if pooled.features[modality].shape != (n_rows, width):
            raise ValueError("Pooled feature dimensions differ from raw sequences")
        stats = statistics[modality]
        mean = np.asarray(stats["mean"], dtype=np.float32)
        scale = np.asarray(stats["scale"], dtype=np.float32)
        if mean.shape != (width,) or scale.shape != (width,):
            raise ValueError("Scaler width differs from raw feature width")
        output_path = output_directory / (modality + ".npy")
        target = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.float32, shape=x.shape)
        max_error = 0.0
        padded_count = 0
        nonzero_count = 0
        block_size = 64
        for first in range(0, n_rows, block_size):
            last = min(n_rows, first + block_size)
            block = np.zeros((last - first, steps, width), dtype=np.float32)
            for local, row_index in enumerate(range(first, last)):
                length = int(lengths[row_index])
                tail = np.asarray(x[row_index, length:])
                padded_count += int(tail.size)
                nonzero_count += int(np.count_nonzero(tail))
                if length:
                    transformed = (np.asarray(x[row_index, :length], dtype=np.float32) - mean) / scale
                    if not np.isfinite(transformed).all():
                        raise ValueError("Nonfinite standardized sequence values")
                    block[local, :length] = transformed
                    pooled_value = (block[local, 0] if modality == "text"
                                    else block[local, :length].mean(axis=0, dtype=np.float32))
                else:
                    pooled_value = np.zeros(width, dtype=np.float32)
                expected = np.asarray(pooled.features[modality][row_index], dtype=np.float32)
                error = float(np.max(np.abs(pooled_value - expected))) if width else 0.0
                max_error = max(max_error, error)
                if not np.allclose(pooled_value, expected, atol=POOLING_TOLERANCE, rtol=0):
                    raise ValueError("Sequence pooling differs from bound pooled scaler for " + modality + " row " + str(row_index))
            target[first:last] = block
        target.flush()
        del target
        np.save(output_directory / (modality + "_lengths.npy"), np.asarray(lengths))
        max_errors[modality] = max_error
        padded_entries[modality] = padded_count
        nonzero_padding[modality] = nonzero_count
    return max_errors, padded_entries, nonzero_padding


def prepare_cache(cache_dir, pooled_standardized_cache, output_dir, author_csv=None, allow_fixture=False):
    """Write a sequence cache using saved pooled TRAIN stats; fixtures require explicit API opt-in."""
    source = Path(cache_dir)
    pooled_cache = Path(pooled_standardized_cache)
    output = Path(output_dir)
    author_csv = Path(author_csv) if author_csv is not None else Path(__file__).resolve().parents[1] / "data/author/simsv2/meta.csv"
    source_metadata_path = _metadata_path(source)
    pooled_metadata_path = _metadata_path(pooled_cache)
    source_metadata = _load_json(source_metadata_path)
    pooled_metadata = _load_json(pooled_metadata_path)
    author_csv_sha = digest(author_csv)
    if not allow_fixture and author_csv_sha != SOURCE_CSV_SHA256:
        raise ValueError("Pinned author CSV required")
    if (source_metadata.get("official_csv_sha256") != author_csv_sha
            or source_metadata.get("source_descriptor", {}).get("official_csv_sha256") != author_csv_sha
            or pooled_metadata.get("author_csv_sha256") != author_csv_sha):
        raise ValueError("Author CSV is not bound to both verified caches")
    if pooled_metadata.get("dataset") != "simsv2" or pooled_metadata.get("test_exported") is not False:
        raise ValueError("Unsupported pooled standardized cache contract")
    if Path(pooled_metadata.get("source_cache", "")).resolve() != source.resolve():
        raise ValueError("Pooled scaler cache is bound to a different raw source cache")
    implementation_root = Path(__file__).resolve().parents[1]
    pooled_preparer = implementation_root / "scripts/prepare_native_pooled_standardization.py"
    normalization_implementation = implementation_root / "jevaffect/anchor_calibration.py"
    if (pooled_metadata.get("preparation_sha256") != digest(pooled_preparer)
            or pooled_metadata.get("normalization_implementation_sha256") != digest(normalization_implementation)):
        raise ValueError("Pooled normalization metadata is not bound to the current verified implementation")

    rows = author_rows(author_csv, "simsv2")
    raw_datasets = {split: NativeSplit(source, split, rows, MODALITIES, pooled=False)
                    for split in ("train", "valid")}
    pooled_datasets = {split: NativeSplit(pooled_cache, split, rows, MODALITIES, pooled=True)
                       for split in ("train", "valid")}
    preserved_time_artifacts = {
        split: _discover_timestamp_artifacts(source, source_metadata, split)
        for split in ("train", "valid")
    }
    counts = {split: len(raw_datasets[split]) for split in ("train", "valid")}
    if (source_metadata.get("split_counts") != counts or pooled_metadata.get("counts") != counts
            or any(raw_datasets[s].ids != pooled_datasets[s].ids for s in counts)
            or any(not np.array_equal(raw_datasets[s].targets, pooled_datasets[s].targets) for s in counts)):
        raise ValueError("Raw and pooled caches do not have identical TRAIN/VALID IDs and M targets")
    if not allow_fixture and (counts != FORMAL_COUNTS or counts != {s: len([r for r in rows.values() if r["split"] == s]) for s in counts}):
        raise ValueError("Formal native cache requires exactly 2,722 TRAIN and 647 VALID clips")

    actual_shapes = {}
    dimensions = {}
    for modality in MODALITIES:
        shape = raw_datasets["train"].features[modality].shape[1:]
        if len(shape) != 2:
            raise ValueError("Raw sequences must preserve [T,D] modality shapes")
        actual_shapes[modality] = tuple(int(v) for v in shape)
        dimensions[modality] = actual_shapes[modality][1]
        if any(raw_datasets[split].features[modality].shape[1:] != actual_shapes[modality] for split in counts):
            raise ValueError("TRAIN and VALID raw sequence shapes differ")
        if any(pooled_datasets[split].features[modality].shape[1] != dimensions[modality] for split in counts):
            raise ValueError("Pooled cache feature dimensions differ from raw sequences")
    descriptor = _raw_contract(source_metadata, actual_shapes, allow_fixture)
    if not allow_fixture and dimensions != {"text": 768, "audio": 768, "vision": 384}:
        raise ValueError("Formal cache dimensions differ from pinned native encoders")
    descriptor_dimensions = descriptor["base_raw_cache_descriptor"]["encoder_provenance"].get("dimensions", {})
    if {m: int(descriptor_dimensions.get(m, -1)) for m in MODALITIES} != dimensions:
        raise ValueError("Raw encoder provenance dimensions differ from feature arrays")
    if pooled_metadata.get("dimensions") != dimensions:
        raise ValueError("Pooled scaler cache dimensions differ from raw sequences")
    for split in counts:
        for modality in MODALITIES:
            lengths = raw_datasets[split].lengths[modality]
            if not np.array_equal(pooled_datasets[split].lengths[modality], (lengths > 0).astype(np.int64)):
                raise ValueError("Pooled modality presence differs from raw sequence lengths")

    pooled_stats_path = _verify_recorded_artifact(pooled_cache, pooled_metadata, "normalization.json")
    statistics = _load_json(pooled_stats_path)
    _validate_stats(statistics, pooled_metadata, pooled_cache, counts["train"], dimensions)
    raw_metadata_sha = digest(source_metadata_path)
    pooled_metadata_sha = digest(pooled_metadata_path)
    stats_sha = digest(pooled_stats_path)
    preparation_sha = digest(__file__)
    input_form = "Original MacBERT CLS/WavLM/DINOv2 sequences; exact saved TRAIN clip-pooled coordinate scaler applied to valid steps"
    raw_feature_hashes = {split: _source_feature_hashes(raw_datasets[split], split) for split in counts}
    pooled_feature_hashes = {split: _source_feature_hashes(pooled_datasets[split], split) for split in counts}
    for split in counts:
        recorded = _recorded_source_hashes(pooled_metadata.get("source_feature_sha256", {}).get(split), split)
        if recorded != raw_feature_hashes[split]:
            raise ValueError("Pooled normalization cache is not bound to the current raw feature bytes: " + split)

    raw_split_hashes = {}
    pooled_split_label_hashes = {}
    for split in counts:
        raw_dir = _split_directory(source, source_metadata, split)
        pooled_dir = _split_directory(pooled_cache, pooled_metadata, split)
        raw_relative = (raw_dir / "raw_labels.json").relative_to(source.resolve()).as_posix()
        raw_label_sha = digest(raw_dir / "raw_labels.json")
        raw_record = source_metadata.get("files", {}).get(raw_relative, {}).get("sha256")
        if raw_record != raw_label_sha:
            raise ValueError("Raw M target labels failed their source metadata hash: " + split)
        pooled_relative = split + "/raw_labels.json"
        pooled_label_sha = digest(_verify_recorded_artifact(pooled_cache, pooled_metadata, pooled_relative))
        raw_split_hashes[split] = raw_label_sha
        pooled_split_label_hashes[split] = pooled_label_sha

    normalization_record = {
        **pooled_metadata["normalization"],
        "source_metadata_sha256": pooled_metadata_sha,
        "statistics_sha256": stats_sha,
    }
    expected_bindings = {
        "source_cache": str(source.resolve()),
        "source_metadata_sha256": raw_metadata_sha,
        "pooled_standardized_cache": str(pooled_cache.resolve()),
        "pooled_standardized_metadata_sha256": pooled_metadata_sha,
        "normalization_statistics_sha256": stats_sha,
        "normalization_implementation_sha256": pooled_metadata["normalization_implementation_sha256"],
        "author_csv_sha256": author_csv_sha,
        "preparation_sha256": preparation_sha,
        "source_form": pooled_metadata.get("source_form"),
        "input_form": input_form,
        "source_feature_sha256": raw_feature_hashes,
        "source_raw_labels_sha256": raw_split_hashes,
        "pooled_source_feature_sha256": pooled_feature_hashes,
        "pooled_source_raw_labels_sha256": pooled_split_label_hashes,
        "preserved_time_artifacts": preserved_time_artifacts,
        "normalization": normalization_record,
        "source_descriptor": descriptor,
        "raw_encoder_provenance": source_metadata.get("raw_bundle_encoder_provenance"),
        "counts": counts,
        "sequence_shapes": {m: list(actual_shapes[m]) for m in MODALITIES},
        "dimensions": dimensions,
    }
    if output.exists():
        existing = _check_complete_existing(output, expected_bindings)
        for ds in (*raw_datasets.values(), *pooled_datasets.values()):
            _close_dataset(ds)
        return existing

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".native-sequence-standardized-", dir=output.parent) as temporary:
        stage = Path(temporary)
        shutil.copyfile(pooled_stats_path, stage / "normalization.json")
        pooling_errors = {split: {} for split in counts}
        padded_entries = {m: 0 for m in MODALITIES}
        nonzero_padding = {m: 0 for m in MODALITIES}
        for split in counts:
            errors, padded, nonzero = _transform_split(
                raw_datasets[split], pooled_datasets[split], source, split, stage, statistics,
            )
            pooling_errors[split] = errors
            for modality in MODALITIES:
                padded_entries[modality] += padded[modality]
                nonzero_padding[modality] += nonzero[modality]
            source_directory = _split_directory(source, source_metadata, split)
            for name, record in preserved_time_artifacts[split].items():
                destination = stage / record["output_relative_path"]
                shutil.copyfile(source_directory / name, destination)
                output_sha = digest(destination)
                if output_sha != record["source_sha256"]:
                    raise ValueError("Copied timestamp artifact differs from its raw source bytes: " + split + "/" + name)
                record["output_sha256"] = output_sha

        metadata = {
            "dataset": "simsv2",
            **expected_bindings,
            "padding_audit": {
                "policy": "All entries at positions >= saved valid length are written as zero",
                "padded_entries": padded_entries,
                "original_nonzero_padding_entries": nonzero_padding,
                "extra_padding_cleanup_intervention": any(nonzero_padding.values()),
            },
            "pooling_identity": {
                "rule": {"text": "first valid CLS token", "audio": "mean over valid frames", "vision": "mean over valid frames"},
                "tolerance_atol": POOLING_TOLERANCE,
                "tolerance_rtol": 0.0,
                "maximum_absolute_error": pooling_errors,
            },
            "artifact_sha256": {},
            "test_exported": False,
        }
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                metadata["artifact_sha256"][path.relative_to(stage).as_posix()] = digest(path)
        save_json(stage / "metadata.json", metadata)
        stage.rename(output)
    for ds in (*raw_datasets.values(), *pooled_datasets.values()):
        _close_dataset(ds)
    result = dict(metadata)
    result["cache_reused"] = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True, help="Verified full raw sequence cache")
    parser.add_argument("--pooled-standardized-cache", type=Path, required=True,
                        help="Verified pooled cache that contains the TRAIN-only normalization statistics")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_cache(args.cache_dir, args.pooled_standardized_cache, args.output_dir)
    print(json.dumps({
        "counts": result["counts"],
        "sequence_shapes": result["sequence_shapes"],
        "artifacts": len(result["artifact_sha256"]),
        "padding_audit": result["padding_audit"],
        "pooling_identity_max_abs_error": result["pooling_identity"]["maximum_absolute_error"],
        "preserved_time_artifacts": {
            split: sorted(artifacts) for split, artifacts in result["preserved_time_artifacts"].items()
        },
        "cache_reused": result["cache_reused"],
        "test_exported": result["test_exported"],
    }, indent=2))
