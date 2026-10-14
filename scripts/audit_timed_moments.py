"""Independent CPU/NumPy audit for the registered native timed-moment matrix."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf


CASES = ("static_product", "marginal_variance", "paired_cov", "unpaired_cov")
SEEDS = (13, 42, 2026)
FORMAL_KEYS = {(case, seed) for case in CASES for seed in SEEDS}
EXPECTED_DIMS = {"text": 768, "audio": 768, "vision": 384}
EXPECTED_COUNTS = {"train": 2722, "valid": 647}
EXPECTED_CSV_SHA256 = "be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150"
PARAMETER_COUNT = 494101
REPLAY_TOLERANCE = 1e-5
BOOTSTRAP_SEED = 20261014
BOOTSTRAP_REPLICATES = 5000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _expected_hash(record: Any) -> tuple[str | None, int | None]:
    if isinstance(record, dict):
        value = record.get("sha256")
        size = record.get("size_bytes", record.get("bytes"))
        return (value if isinstance(value, str) else None,
                int(size) if isinstance(size, int) else None)
    return (record if isinstance(record, str) else None, None)


class DigestCache:
    """Hash each large cache file at most once during this audit."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def digest(self, path: Path) -> str:
        key = str(Path(path).resolve())
        if key not in self.values:
            self.values[key] = sha256_file(Path(key))
        return self.values[key]


class Recorder:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []
        self.findings: list[dict[str, str]] = []

    def must(self, name: str, passed: bool, detail: str) -> bool:
        passed = bool(passed)
        self.checks.append({"name": name, "passed": passed, "detail": str(detail)})
        if not passed:
            self.findings.append({"check": name, "issue": str(detail)})
        return passed


def _resolve_manifest_path(key: str, root: Path, *, preferred: Path | None = None,
                           repo_root: Path | None = None) -> Path:
    candidate = Path(key)
    if candidate.is_absolute():
        return candidate
    choices: list[Path] = []
    if preferred is not None:
        choices.append(preferred / candidate)
    choices.append(root / candidate)
    if repo_root is not None:
        choices.append(repo_root / candidate)
    for path in choices:
        if path.is_file():
            return path
    return choices[0]


def _verify_hash_map(mapping: Any, root: Path, label: str, digests: DigestCache,
                     *, rec: Recorder, preferred: Path | None = None,
                     repo_root: Path | None = None) -> tuple[int, list[str]]:
    if not isinstance(mapping, dict):
        rec.must(label, False, "Hash manifest is missing or malformed")
        return 0, ["manifest missing or malformed"]
    failures: list[str] = []
    for key, record in mapping.items():
        expected, expected_size = _expected_hash(record)
        if expected is None:
            failures.append(f"{key}: expected SHA256 missing")
            continue
        path = _resolve_manifest_path(str(key), root, preferred=preferred, repo_root=repo_root)
        if not path.is_file():
            failures.append(f"{key}: file not found at {path}")
            continue
        try:
            actual = digests.digest(path)
        except OSError as exc:
            failures.append(f"{key}: unreadable ({exc})")
            continue
        if actual != expected:
            failures.append(f"{key}: SHA256 differs")
            continue
        if expected_size is not None and path.stat().st_size != expected_size:
            failures.append(f"{key}: byte count differs")
    detail = (f"verified {len(mapping) - len(failures)}/{len(mapping)} declared hashes"
              if not failures else "; ".join(failures[:8]))
    rec.must(label, not failures, detail)
    return len(mapping) - len(failures), failures


def _read_author_csv(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    rows: dict[str, dict[str, float]] = {"train": {}, "valid": {}}
    ordered: dict[str, list[str]] = {"train": [], "valid": []}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "clip_id", "label", "mode"}
        if not reader.fieldnames or not required <= set(reader.fieldnames):
            raise ValueError("Author CSV is missing video_id, clip_id, label, or mode")
        for row in reader:
            split = (row.get("mode") or "").strip().lower()
            if split not in rows:
                continue
            sample_id = f"{row['video_id'].strip()}/{row['clip_id'].strip()}"
            if sample_id in rows[split]:
                raise ValueError(f"Duplicate author ID: {sample_id}")
            score = float(row["label"])
            if not math.isfinite(score):
                raise ValueError(f"Nonfinite author label for {sample_id}")
            rows[split][sample_id] = score
            ordered[split].append(sample_id)
    return rows, ordered


def _load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_metrics(prediction: Any, target: Any) -> dict[str, float | int | None]:
    """Independent CH-SIMS native clipped MAE, Corr and positive-only Acc2."""
    p = np.asarray(prediction, dtype=np.float32).reshape(-1)
    y = np.asarray(target, dtype=np.float32).reshape(-1)
    if p.shape != y.shape or not len(y) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("Native metrics need finite, nonempty aligned arrays")
    p, y = np.clip(p, -1, 1), np.clip(y, -1, 1)
    corr = None if len(y) < 2 or np.std(y) == 0 or np.std(p) == 0 else float(np.corrcoef(p, y)[0, 1])
    # CH-SIMS Acc2 treats zero as negative: its author threshold is strictly > 0.
    acc2 = float(np.mean((p > 0) == (y > 0)))
    return {"count": int(len(y)), "MAE": float(np.mean(np.abs(p - y))),
            "Corr": corr, "Acc2_positive_strict": acc2}


def _close(a: Any, b: Any, atol: float = 1e-7) -> bool:
    if a is None or b is None:
        return a is None and b is None
    try:
        return bool(np.isclose(float(a), float(b), atol=atol, rtol=0, equal_nan=False))
    except (TypeError, ValueError):
        return False


def _validate_stats_and_normalize(raw: dict[str, np.ndarray], presence: np.ndarray,
                                  pair_valid: np.ndarray, stats: dict[str, Any],
                                  *, rec: Recorder, context: str,
                                  verify_fit: bool) -> dict[str, np.ndarray]:
    required_stats = {"text": "text", "audio": "audio_mean", "vision": "vision_mean"}
    fitted_ok = True
    fit_indices: list[int] | None = None
    normalized: dict[str, np.ndarray] = {}
    for modality, raw_name in required_stats.items():
        entry = stats.get(modality)
        if not isinstance(entry, dict):
            fitted_ok = False
            continue
        values = np.asarray(raw[raw_name], dtype=np.float32)
        mean64 = values.astype(np.float64).mean(axis=0)
        scale64 = values.astype(np.float64).std(axis=0)
        scale64[scale64 < 1e-6] = 1.
        recomputed_mean = mean64.astype(np.float32)
        recomputed_scale = scale64.astype(np.float32)
        saved_mean = np.asarray(entry.get("mean"), dtype=np.float32)
        saved_scale = np.asarray(entry.get("scale"), dtype=np.float32)
        fitted_ok &= saved_mean.shape == recomputed_mean.shape and np.array_equal(saved_mean, recomputed_mean)
        fitted_ok &= saved_scale.shape == recomputed_scale.shape and np.array_equal(saved_scale, recomputed_scale)
        indices = entry.get("fit_indices")
        if not isinstance(indices, list):
            fitted_ok = False
        elif fit_indices is None:
            fit_indices = indices
        elif indices != fit_indices:
            fitted_ok = False
        if saved_mean.shape != (values.shape[-1],) or saved_scale.shape != (values.shape[-1],):
            raise ValueError(f"Invalid {modality} normalization coordinate shape")
        if not np.isfinite(saved_mean).all() or not np.isfinite(saved_scale).all() or np.any(saved_scale <= 0):
            raise ValueError(f"Invalid {modality} normalization values")
        normalized[raw_name] = ((values - saved_mean) / saved_scale).astype(np.float32)
    if verify_fit:
        rec.must(f"{context}: TRAIN-only mean and population-STD recomputation",
                 fitted_ok and fit_indices == list(range(len(raw["text"]))),
                 "Saved mean/scale must exactly match independent float64 TRAIN fit with float32 output and fit indices")

    for name, mask_index in (("text", 0), ("audio_mean", 1), ("vision_mean", 2)):
        normalized[name][~presence[:, mask_index]] = 0
    n = len(raw["text"])
    normalized["audio_grid"] = ((raw["audio_grid"].reshape(-1, EXPECTED_DIMS["audio"]).astype(np.float32)
                                 - np.asarray(stats["audio"]["mean"], dtype=np.float32))
                                / np.asarray(stats["audio"]["scale"], dtype=np.float32)).astype(np.float32).reshape(
                                    n, 4, EXPECTED_DIMS["audio"])
    normalized["vision_grid"] = ((raw["vision_grid"].reshape(-1, EXPECTED_DIMS["vision"]).astype(np.float32)
                                   - np.asarray(stats["vision"]["mean"], dtype=np.float32))
                                  / np.asarray(stats["vision"]["scale"], dtype=np.float32)).astype(np.float32).reshape(
                                      n, 4, EXPECTED_DIMS["vision"])
    normalized["audio_grid"][~pair_valid] = 0
    normalized["vision_grid"][~pair_valid] = 0
    normalized["pair_valid"] = pair_valid.astype(bool, copy=False)
    if not all(np.isfinite(values).all() for values in normalized.values()):
        raise ValueError("Independent normalization produced nonfinite features")
    return normalized


def _numpy_moment(features: dict[str, np.ndarray], weights: dict[str, np.ndarray], case: str,
                  *, vision_grid: np.ndarray | None = None) -> np.ndarray:
    audio_projection = np.asarray(weights["audio_projection.weight"], dtype=np.float64)
    vision_projection = np.asarray(weights["vision_projection.weight"], dtype=np.float64)
    valid = np.asarray(features["pair_valid"], dtype=bool).reshape(-1)
    if case == "static_product":
        a = np.asarray(features["audio_mean"], dtype=np.float64) @ audio_projection.T
        v = np.asarray(features["vision_mean"], dtype=np.float64) @ vision_projection.T
        moment = a * v
    else:
        a_grid = np.asarray(features["audio_grid"], dtype=np.float64)
        v_grid = np.asarray(features["vision_grid"] if vision_grid is None else vision_grid, dtype=np.float64)
        if case == "unpaired_cov":
            v_grid = np.roll(v_grid, 1, axis=1)
        a = a_grid @ audio_projection.T
        v = v_grid @ vision_projection.T
        a = a - a.mean(axis=1, keepdims=True)
        v = v - v.mean(axis=1, keepdims=True)
        if case == "marginal_variance":
            moment = .5 * (np.mean(a * a, axis=1) + np.mean(v * v, axis=1))
        elif case in ("paired_cov", "unpaired_cov"):
            moment = np.mean(a * v, axis=1)
        else:
            raise ValueError(f"Unknown moment case: {case}")
    moment[~valid] = 0
    return moment


def numpy_forward(features: dict[str, np.ndarray], weights: dict[str, np.ndarray], case: str,
                  *, vision_grid: np.ndarray | None = None,
                  audio_grid: np.ndarray | None = None) -> np.ndarray:
    """Independent float64 forward; GELU is the exact erf definition."""
    if case not in CASES:
        raise ValueError(f"Unknown moment case: {case}")
    local = dict(features)
    if audio_grid is not None:
        local["audio_grid"] = audio_grid
    if vision_grid is not None:
        local["vision_grid"] = vision_grid
    moment = _numpy_moment(local, weights, case)
    globals_ = np.concatenate((np.asarray(local["text"], dtype=np.float64),
                               np.asarray(local["audio_mean"], dtype=np.float64),
                               np.asarray(local["vision_mean"], dtype=np.float64)), axis=1)
    joined = np.concatenate((globals_, moment), axis=1)
    hidden = joined @ np.asarray(weights["head.0.weight"], dtype=np.float64).T
    hidden += np.asarray(weights["head.0.bias"], dtype=np.float64)
    hidden = .5 * hidden * (1. + erf(hidden / math.sqrt(2.)))
    output = hidden @ np.asarray(weights["head.3.weight"], dtype=np.float64).T
    output += np.asarray(weights["head.3.bias"], dtype=np.float64)
    return output.reshape(-1)


def torch_cpu_forward(features: dict[str, np.ndarray], weights: dict[str, Any], case: str) -> np.ndarray:
    """Functional CPU replay of the saved model without importing its forward code."""
    import torch
    import torch.nn.functional as F

    tensor = {key: torch.from_numpy(np.array(value, dtype=np.float32, copy=True))
              for key, value in features.items() if key != "pair_valid"}
    valid = torch.from_numpy(np.array(features["pair_valid"], dtype=bool, copy=True))
    wa = weights["audio_projection.weight"].detach().cpu().float()
    wv = weights["vision_projection.weight"].detach().cpu().float()
    if case == "static_product":
        moment = F.linear(tensor["audio_mean"], wa) * F.linear(tensor["vision_mean"], wv)
    else:
        a = F.linear(tensor["audio_grid"], wa)
        v = F.linear(tensor["vision_grid"], wv)
        if case == "unpaired_cov":
            v = torch.roll(v, shifts=1, dims=1)
        a = a - a.mean(dim=1, keepdim=True)
        v = v - v.mean(dim=1, keepdim=True)
        if case == "marginal_variance":
            moment = .5 * (a.square().mean(dim=1) + v.square().mean(dim=1))
        elif case in ("paired_cov", "unpaired_cov"):
            moment = (a * v).mean(dim=1)
        else:
            raise ValueError(f"Unknown moment case: {case}")
    moment = torch.where(valid[:, None], moment, torch.zeros_like(moment))
    globals_ = torch.cat((tensor["text"], tensor["audio_mean"], tensor["vision_mean"]), dim=1)
    joined = torch.cat((globals_, moment), dim=1)
    hidden = F.linear(joined, weights["head.0.weight"].detach().cpu().float(),
                      weights["head.0.bias"].detach().cpu().float())
    hidden = F.gelu(hidden, approximate="none")
    output = F.linear(hidden, weights["head.3.weight"].detach().cpu().float(),
                      weights["head.3.bias"].detach().cpu().float())
    return output.flatten().detach().cpu().numpy()


def _state_numpy(state: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() if hasattr(value, "detach") else np.asarray(value)
            for key, value in state.items()}


def _model_parameter_count(state: dict[str, Any]) -> int:
    return int(sum(value.numel() if hasattr(value, "numel") else np.asarray(value).size
                   for value in state.values()))


def _expected_model_shapes(weights: dict[str, np.ndarray]) -> bool:
    return (weights.get("audio_projection.weight", np.empty(0)).shape == (32, 768)
            and weights.get("vision_projection.weight", np.empty(0)).shape == (32, 384)
            and weights.get("head.0.weight", np.empty(0)).shape == (234, 1952)
            and weights.get("head.0.bias", np.empty(0)).shape == (234,)
            and weights.get("head.3.weight", np.empty(0)).shape == (1, 234)
            and weights.get("head.3.bias", np.empty(0)).shape == (1,))


def _replay_scheduler(history: list[dict[str, Any]], base_lr: float) -> dict[str, Any]:
    """ReduceLROnPlateau defaults in the registered mode=min/patience=3/factor=.5."""
    lr = float(base_lr)
    best = math.inf
    bad = 0
    expected: list[float] = []
    for row in history:
        score = float(row["valid"]["MAE"])
        if score < best * (1. - 1e-4):
            best, bad = score, 0
        else:
            bad += 1
        if bad > 3:
            lowered = max(lr * .5, 0.)
            if lr - lowered > 1e-8:
                lr = lowered
            bad = 0
        expected.append(lr)
    return {"learning_rates": expected, "best": best, "num_bad_epochs": bad,
            "last_epoch": len(history)}


def _check_prediction_records(records: Any, *, ids: list[str], target: np.ndarray,
                              label: str, rec: Recorder) -> np.ndarray:
    valid = isinstance(records, list) and len(records) == len(ids)
    if not valid:
        rec.must(label, False, f"Expected {len(ids)} saved VALID prediction rows")
        raise ValueError(f"Malformed prediction rows for {label}")
    found_ids: list[str] = []
    found_target: list[float] = []
    found_prediction: list[float] = []
    try:
        for row in records:
            found_ids.append(str(row["id"]))
            found_target.append(float(row["target"]))
            found_prediction.append(float(row["prediction"]))
    except (KeyError, TypeError, ValueError) as exc:
        rec.must(label, False, f"Invalid VALID row schema: {exc}")
        raise ValueError(f"Malformed prediction row for {label}") from exc
    ok = (found_ids == ids and np.allclose(np.asarray(found_target, np.float32), target,
                                            atol=1e-6, rtol=0)
          and np.isfinite(found_prediction).all())
    rec.must(label, ok, "Ordered IDs, original author M targets and finite predictions must match the canonical VALID split")
    return np.asarray(found_prediction, dtype=np.float32)


def _verify_identity_files(identity: dict[str, Any], repo_root: Path, digests: DigestCache,
                           rec: Recorder, label: str) -> tuple[int, list[str]]:
    implementation = identity.get("implementation")
    return _verify_hash_map(implementation, repo_root, f"{label}: implementation hashes", digests,
                            rec=rec, repo_root=repo_root)


def _seeded_initialization_audit(repo_root: Path, rec: Recorder) -> dict[str, Any]:
    training_path = repo_root / "jevaffect" / "timed_moment_training.py"
    source = training_path.read_text(encoding="utf-8")
    rng_tokens = ("random.seed(seed)", "np.random.seed(seed)", "torch.manual_seed(seed)",
                  "torch.cuda.manual_seed_all(seed)", "torch.use_deterministic_algorithms(True)",
                  "torch.Generator().manual_seed(seed)")
    positions = [source.find(token) for token in rng_tokens]
    model_position = source.find("model = TimedMomentRegressor(")
    source_ok = all(position >= 0 for position in positions) and model_position > max(positions[:3])
    rec.must("registered seeded initialization policy", source_ok,
             "Hashed trainer must reseed Python, NumPy and Torch before constructing the same case-independent module, and seed the train loader")
    if not source_ok:
        return {"source_policy_verified": False, "per_seed_initial_parameter_sha256": {}}
    import torch
    sys.path.insert(0, str(repo_root))
    from jevaffect.timed_moments import TimedMomentRegressor

    hashes: dict[str, dict[str, str]] = {}
    all_equal = True
    for seed in SEEDS:
        hashes[str(seed)] = {}
        for case in CASES:
            torch.manual_seed(seed)
            model = TimedMomentRegressor(EXPECTED_DIMS, case, rank=32, hidden=234, dropout=.1)
            data = b"".join(name.encode("utf-8") + b"\0" + tensor.detach().cpu().contiguous().numpy().tobytes()
                            for name, tensor in sorted(model.state_dict().items()))
            hashes[str(seed)][case] = hashlib.sha256(data).hexdigest()
        all_equal &= len(set(hashes[str(seed)].values())) == 1
    rec.must("case-matched seeded model initialization", all_equal,
             "Reconstructing each registered case under the same seed yields identical projection and head initial weights")
    return {"source_policy_verified": source_ok, "same_initial_parameters_across_cases": all_equal,
            "per_seed_initial_parameter_sha256": hashes,
            "artifact_limitation": "The run stores no separate initial-weight file; equality is established from the hashed trainer policy and reconstructed module initialization."}


def _audit_fit(repo_root: Path, fit: dict[str, Any], config: dict[str, Any], config_sha: str,
               author: dict[str, dict[str, float]], ids: list[str], targets: np.ndarray,
               features: dict[str, np.ndarray], digests: DigestCache, rec: Recorder) -> dict[str, Any]:
    case, seed = str(fit["case"]), int(fit["seed"])
    label = f"{case}/seed_{seed}"
    run_dir = repo_root / "runs" / "native_timed_moments_v1" / "formal" / case / f"seed_{seed}"
    result_path, identity_path = run_dir / "result.json", run_dir / "identity.json"
    history_path, prediction_path = run_dir / "history.json", run_dir / "valid_predictions.json"
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    for path in (result_path, identity_path, history_path, prediction_path, best_path, last_path):
        if not path.is_file():
            rec.must(f"{label}: required formal artifact", False, f"Missing {path.relative_to(repo_root)}")
            raise ValueError(f"Missing formal artifact {path}")
    result = _load_json(result_path)
    identity = _load_json(identity_path)
    history = _load_json(history_path)
    saved_records = _load_json(prediction_path)
    expected_config = dict(config)
    for key in ("author_csv", "cache_dir", "normalization_cache_dir", "run_root"):
        expected_config[key] = str((repo_root / expected_config[key]).resolve())
    expected_config.update({"case": case, "seed": seed, "smoke32": False,
                            "output_dir": str((repo_root / config["run_root"]).resolve())})
    identity_ok = (identity.get("case") == case and identity.get("seed") == seed
                   and identity.get("smoke_limit") is None and identity.get("config") == expected_config
                   and identity.get("config_sha256") == config_sha
                   and identity.get("author_csv_sha256") == sha256_file(repo_root / config["author_csv"])
                   and identity.get("device") == "cuda" and identity.get("parameter_count") == PARAMETER_COUNT
                   and identity.get("test_evaluated") is False)
    rec.must(f"{label}: run identity binds registered config, seed, author CSV, CUDA and parameter budget",
             identity_ok, "Checkpoint identity must match the frozen formal config and full native run")
    rec.must(f"{label}: complete non-smoke result", result.get("complete") is True
             and result.get("smoke_only") is False and result.get("dataset") == "simsv2"
             and result.get("case") == case and int(result.get("seed", -1)) == seed
             and result.get("train_count") == EXPECTED_COUNTS["train"]
             and result.get("valid_count") == EXPECTED_COUNTS["valid"]
             and result.get("full_train_count") == EXPECTED_COUNTS["train"]
             and result.get("full_valid_count") == EXPECTED_COUNTS["valid"]
             and result.get("test_evaluated") is False,
             "Formal fit must use the complete TRAIN/VALID splits and no TEST export")
    result_file_hashes = {
        "best_checkpoint_sha256": digests.digest(best_path),
        "last_checkpoint_sha256": digests.digest(last_path),
        "predictions_sha256": digests.digest(prediction_path),
        "identity_sha256": digests.digest(identity_path),
    }
    path_hash_ok = (result.get("checkpoint_sha256") == result_file_hashes["best_checkpoint_sha256"]
                    and result.get("last_checkpoint_sha256") == result_file_hashes["last_checkpoint_sha256"]
                    and result.get("predictions_sha256") == result_file_hashes["predictions_sha256"]
                    and result.get("identity_sha256") == result_file_hashes["identity_sha256"])
    rec.must(f"{label}: result artifact SHA256 bindings", path_hash_ok,
             "Reported best, last, predictions and identity hashes must match disk")

    # Verify all candidate per-split feature and provenance hashes.
    feature_hash_count = 0
    feature_hash_failures: list[str] = []
    for split, manifest in identity.get("features", {}).items():
        count, failures = _verify_hash_map(manifest, Path(identity["timed_cache"]),
                                           f"{label}: {split} timed-cache artifact hashes", digests,
                                           rec=rec, repo_root=repo_root)
        feature_hash_count += count
        feature_hash_failures.extend(failures)
    implementation_count, implementation_failures = _verify_identity_files(identity, repo_root, digests, rec, label)

    timed_meta = _load_json(Path(identity["timed_cache"]) / "metadata.json")
    expected_source_provenance = {
        key: timed_meta.get(key)
        for key in ("preparation_sha256", "native_reader_sha256", "source_cache", "source_descriptor",
                    "source_truncation", "source_feature_sha256", "time_contract", "coverage")
    }
    normal_meta = _load_json(repo_root / config["normalization_cache_dir"] / "metadata.json")
    normal_expected = {key: normal_meta.get(key) for key in (
        "author_csv_sha256", "source_cache", "source_feature_sha256", "normalization", "counts", "dimensions")}
    provenance_ok = identity.get("timed_cache_provenance") == expected_source_provenance
    norm_identity = identity.get("normalization", {})
    norm_dir = (repo_root / config["normalization_cache_dir"]).resolve()
    provenance_ok &= norm_identity.get("path") == str((norm_dir / "normalization.json").resolve())
    provenance_ok &= norm_identity.get("metadata_path") == str((norm_dir / "metadata.json").resolve())
    provenance_ok &= norm_identity.get("sha256") == digests.digest(norm_dir / "normalization.json")
    provenance_ok &= norm_identity.get("metadata_sha256") == digests.digest(norm_dir / "metadata.json")
    provenance_ok &= identity.get("normalization_provenance") == normal_expected
    rec.must(f"{label}: timed-cache and normalizer identity provenance", provenance_ok,
             "Identity must bind current cache metadata, pooled normalization metadata and their source provenance")

    record_predictions = _check_prediction_records(saved_records, ids=ids, target=targets,
                                                    label=f"{label}: saved VALID IDs and targets", rec=rec)
    rec.must(f"{label}: result and saved history agree", result.get("best_epoch") is not None
             and result.get("valid_count") == len(saved_records)
             and isinstance(history, list) and len(history) > 0
             and _load_json(run_dir / "history.json") == history,
             "Result, history and prediction artifacts must be present for the selected run")

    # Recalculate selected-chapter native outputs and locate the exact first minimum.
    best_index = min(range(len(history)), key=lambda index: float(history[index]["valid"]["MAE"]))
    selected_epoch = int(history[best_index]["epoch"])
    mae_values = [float(row["valid"]["MAE"]) for row in history]
    stale = 0
    best_mae = math.inf
    for value in mae_values:
        if value < best_mae:
            best_mae, stale = value, 0
        else:
            stale += 1
    final_epoch = int(history[-1]["epoch"])
    budget = identity.get("budget", {})
    patience = int(budget.get("early_stopping_patience", -1))
    expected_complete = (final_epoch >= int(budget.get("max_epochs", 0)) or stale >= patience)
    best_first_ok = (selected_epoch == int(result.get("best_epoch", -1))
                     and selected_epoch == best_index + 1
                     and abs(best_mae - float(result.get("valid_metrics", {}).get("MAE", math.inf))) <= 1e-7
                     and int(result.get("max_epochs", -1)) == budget.get("max_epochs")
                     and int(result.get("patience", -1)) == patience
                     and int(result.get("trainable_parameters", -1)) == PARAMETER_COUNT
                     and result.get("complete") is expected_complete)
    rec.must(f"{label}: first-minimum MAE selection and early-stop accounting", best_first_ok,
             f"Selected epoch {selected_epoch}; history has {len(history)} epochs, final stale count {stale}")
    valid_ids_order_ok = all(int(row.get("epoch", -1)) == index + 1 for index, row in enumerate(history))
    rec.must(f"{label}: contiguous epoch history", valid_ids_order_ok,
             "History epoch numbers must be the full sequential training trace")
    scheduler_trace = _replay_scheduler(history, float(config["learning_rate"]))
    expected_lrs = scheduler_trace["learning_rates"]
    lrs = [float(row["learning_rate"]) for row in history]
    scheduler_history_ok = len(expected_lrs) == len(lrs) and all(_close(a, b, 1e-12)
                                                                 for a, b in zip(expected_lrs, lrs))
    rec.must(f"{label}: scheduler learning-rate trace", scheduler_history_ok,
             "Independent ReduceLROnPlateau replay must match each recorded epoch learning rate")

    import torch
    best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
    last_checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    best_state = best_checkpoint.get("model", {})
    last_state = last_checkpoint.get("model", {})
    best_numpy = _state_numpy(best_state)
    state_valid = (_expected_model_shapes(best_numpy)
                   and _model_parameter_count(best_state) == PARAMETER_COUNT
                   and _model_parameter_count(last_state) == PARAMETER_COUNT
                   and all(np.isfinite(value).all() for value in best_numpy.values()))
    rec.must(f"{label}: selected model shape and 494101 parameters", state_valid,
             "Saved selected model must match the frozen 1952-to-234 head and rank-32 projections")
    cp_identity_ok = (best_checkpoint.get("identity") == identity
                      and last_checkpoint.get("identity") == identity
                      and int(best_checkpoint.get("epoch", -1)) == selected_epoch
                      and int(last_checkpoint.get("epoch", -1)) == final_epoch
                      and int(last_checkpoint.get("stale", -1)) == stale
                      and _close(last_checkpoint.get("best"), best_mae, 1e-7)
                      and last_checkpoint.get("history") == history
                      and best_checkpoint.get("parameter_count") == PARAMETER_COUNT
                      and best_checkpoint.get("metrics", {}).get("MAE") == result.get("valid_metrics", {}).get("MAE"))
    rec.must(f"{label}: best/last checkpoint identity and selected epoch", cp_identity_ok,
             "Both checkpoints must bind the run identity and the selected/last epoch")
    required_resume = {"optimizer", "scheduler", "torch_rng", "numpy_rng", "python_rng", "loader_rng", "cuda_rng"}
    resume_keys_ok = required_resume <= set(last_checkpoint)
    optimizer = last_checkpoint.get("optimizer", {})
    optimizer_state = optimizer.get("state", {}) if isinstance(optimizer, dict) else {}
    optimizer_groups = optimizer.get("param_groups", []) if isinstance(optimizer, dict) else []
    optimizer_ok = (resume_keys_ok and len(optimizer_state) == 6 and len(optimizer_groups) == 1
                    and _close(optimizer_groups[0].get("weight_decay"), 1e-4, 1e-12)
                    and all({"step", "exp_avg", "exp_avg_sq"} <= set(value) for value in optimizer_state.values()))
    scheduler = last_checkpoint.get("scheduler", {})
    scheduler_ok = (scheduler.get("mode") == "min" and scheduler.get("patience") == 3
                    and _close(scheduler.get("factor"), .5, 1e-12)
                    and _close(scheduler.get("best"), scheduler_trace["best"], 1e-12)
                    and scheduler.get("num_bad_epochs") == scheduler_trace["num_bad_epochs"]
                    and scheduler.get("last_epoch") == scheduler_trace["last_epoch"]
                    and _close(optimizer_groups[0].get("lr"), lrs[-1], 1e-12)
                    if optimizer_groups else False)
    expected_optimizer_steps = final_epoch * math.ceil(EXPECTED_COUNTS["train"] / int(config["batch_size"]))
    optimizer_steps_ok = all(_close(value.get("step").item() if hasattr(value.get("step"), "item") else value.get("step"),
                                    expected_optimizer_steps, 1e-9)
                             for value in optimizer_state.values()) if optimizer_state else False
    optimizer_ok &= optimizer_steps_ok
    rng_ok = False
    if resume_keys_ok:
        rng_ok = (torch.is_tensor(last_checkpoint["torch_rng"]) and last_checkpoint["torch_rng"].dtype == torch.uint8
                  and torch.is_tensor(last_checkpoint["loader_rng"]) and last_checkpoint["loader_rng"].dtype == torch.uint8
                  and isinstance(last_checkpoint["numpy_rng"], tuple) and len(last_checkpoint["numpy_rng"]) == 5
                  and isinstance(last_checkpoint["python_rng"], tuple)
                  and isinstance(last_checkpoint["cuda_rng"], list) and len(last_checkpoint["cuda_rng"]) > 0
                  and all(torch.is_tensor(value) and value.dtype == torch.uint8 for value in last_checkpoint["cuda_rng"]))
    rec.must(f"{label}: last optimizer, scheduler and all RNG states", optimizer_ok and scheduler_ok and rng_ok,
             "Last checkpoint must preserve AdamW moments, plateau scheduler state, CPU/NumPy/Python/loader/CUDA RNG")
    rec.must(f"{label}: registered training budget", identity.get("budget") == {
        "batch_size": 32, "learning_rate": .0001, "weight_decay": .0001, "max_epochs": 80,
        "early_stopping_patience": 12,
        "scheduler": {"name": "ReduceLROnPlateau", "metric": "valid native MAE", "patience": 3, "factor": .5},
        "gradient_clip_norm": 1.0}, "Identity training budget must match the pre-registered contract")

    independent_metrics = native_metrics(record_predictions, targets)
    metric_source = result.get("valid_metrics", {})
    metrics_ok = all(_close(independent_metrics[key], metric_source.get(key), 1e-7)
                     for key in ("MAE", "Corr"))
    # The producer field uses its author name Mult_acc_2; verify the equivalent strict-positive form too.
    metrics_ok &= _close(independent_metrics["Acc2_positive_strict"], metric_source.get("Mult_acc_2"), 1e-7)
    rec.must(f"{label}: independent clipped native MAE/Corr/Acc2", metrics_ok,
             "NumPy recomputation from saved VALID outputs must match clipped author metrics")
    rec.must(f"{label}: selected checkpoint native MAE is first minimum",
             _close(independent_metrics["MAE"], history[best_index]["valid"]["MAE"], 1e-7),
             "Saved selected predictions must reproduce the first minimum validation native MAE")

    # NumPy and independently functional Torch CPU replay from the selected checkpoint.
    numpy_output = numpy_forward(features, best_numpy, case)
    torch_output = torch_cpu_forward(features, best_state, case)
    np_diff = float(np.max(np.abs(numpy_output.astype(np.float64) - record_predictions.astype(np.float64))))
    torch_diff = float(np.max(np.abs(torch_output.astype(np.float64) - record_predictions.astype(np.float64))))
    replay_ok = np.isfinite(numpy_output).all() and np.isfinite(torch_output).all() \
        and np_diff <= REPLAY_TOLERANCE and torch_diff <= REPLAY_TOLERANCE
    rec.must(f"{label}: independent NumPy and Torch CPU replay", replay_ok,
             f"maximum |NumPy-saved|={np_diff:.3g}; maximum |TorchCPU-saved|={torch_diff:.3g}; tolerance={REPLAY_TOLERANCE}")
    per_fit = {
        "case": case, "seed": seed, "complete": bool(result.get("complete")),
        "selected_epoch": selected_epoch, "epochs_run": final_epoch,
        "early_stopped": bool(stale >= patience), "loop_seconds": float(result.get("loop_seconds", 0)),
        "trainable_parameters": int(result.get("trainable_parameters", -1)),
        "parameter_count_from_checkpoint": _model_parameter_count(best_state),
        "native_metrics": independent_metrics,
        "numpy_max_abs_error_vs_saved": np_diff, "torch_cpu_max_abs_error_vs_saved": torch_diff,
        "artifact_sha256": result_file_hashes,
        "verified_feature_hash_count": feature_hash_count,
        "verified_implementation_hash_count": implementation_count,
        "hash_failures": feature_hash_failures + implementation_failures,
        "predictions": record_predictions,
        "targets": targets.copy(), "ids": list(ids), "identity": identity,
        "history": history, "best_state": best_state,
    }
    return per_fit


def _audit_reference(repo_root: Path, *, name: str, config_relative: str, run_root: str,
                     model_name: str, expected_parameters: int, seed: int,
                     author_csv_sha: str, ids: list[str], targets: np.ndarray,
                     digests: DigestCache, rec: Recorder) -> dict[str, Any]:
    config_path = repo_root / config_relative
    label = f"{name}/seed_{seed}"
    config = _load_json(config_path)
    config_sha = digests.digest(config_path)
    run_dir = repo_root / run_root / model_name / f"seed_{seed}"
    identity_path, result_path = run_dir / "identity.json", run_dir / "result.json"
    history_path, predictions_path = run_dir / "history.json", run_dir / "valid_predictions.json"
    best_path = run_dir / "best.pt"
    for path in (identity_path, result_path, history_path, predictions_path, best_path):
        if not path.is_file():
            rec.must(f"{label}: reference artifact", False, f"Missing {path.relative_to(repo_root)}")
            raise ValueError(f"Missing reference artifact {path}")
    identity = _load_json(identity_path)
    result = _load_json(result_path)
    history = _load_json(history_path)
    records = _load_json(predictions_path)
    identity_ok = (identity.get("config") == config and identity.get("seed") == seed
                   and identity.get("smoke_limit") is None
                   and identity.get("author_csv_sha256") == author_csv_sha
                   and config.get("run_root") == run_root and config.get("model") == model_name
                   and config.get("dataset") == "simsv2" and config.get("seeds") == list(SEEDS)
                   and identity.get("device") == "cuda")
    rec.must(f"{label}: actual registered reference config path", identity_ok,
             f"Identity must match {config_relative} and the completed standardized run directory")
    implementation_count, implementation_failures = _verify_identity_files(identity, repo_root, digests, rec, label)
    feature_count, feature_failures = 0, []
    reference_cache = (repo_root / config["cache_dir"]).resolve()
    for split, manifest in identity.get("features", {}).items():
        count, failures = _verify_hash_map(manifest, reference_cache, f"{label}: reference {split} cache hashes",
                                           digests, rec=rec, repo_root=repo_root)
        feature_count += count
        feature_failures.extend(failures)
    prediction = _check_prediction_records(records, ids=ids, target=targets,
                                             label=f"{label}: reference VALID IDs and targets", rec=rec)
    prediction_hash_ok = (result.get("predictions_sha256") == digests.digest(predictions_path)
                          and result.get("checkpoint_sha256") == digests.digest(best_path))
    rec.must(f"{label}: result checkpoint/prediction hashes", prediction_hash_ok,
             "Reference result must bind its selected checkpoint and VALID predictions")
    best_index = min(range(len(history)), key=lambda i: float(history[i]["valid"]["MAE"]))
    selected_first = (int(history[best_index]["epoch"]) == int(result.get("best_epoch", -1))
                      and int(history[best_index]["epoch"]) == best_index + 1
                      and int(result.get("trainable_parameters", -1)) == expected_parameters
                      and result.get("complete") is True and result.get("smoke_only") is False
                      and result.get("test_evaluated") is False)
    rec.must(f"{label}: reference first-minimum selection and parameter budget", selected_first,
             f"First validation MAE minimum from {len(history)} recorded epochs must match the result")
    metrics = native_metrics(prediction, targets)
    reference_metrics = result.get("valid_metrics", {})
    metrics_ok = all(_close(metrics[key], reference_metrics.get(key), 1e-7) for key in ("MAE", "Corr"))
    metrics_ok &= _close(metrics["Acc2_positive_strict"], reference_metrics.get("Mult_acc_2"), 1e-7)
    rec.must(f"{label}: independent native metrics", metrics_ok,
             "Recomputed clipped MAE, Corr and positive-only Acc2 must match the saved reference result")
    rec.must(f"{label}: full reference counts and no TEST", result.get("train_count") == 2722
             and result.get("valid_count") == 647 and result.get("test_evaluated") is False,
             "Reference run must cover full author TRAIN/VALID and no TEST")
    return {"model": name, "seed": seed, "config_path": config_relative,
            "config_sha256": config_sha, "result_path": str(result_path.relative_to(repo_root)),
            "identity_sha256": digests.digest(identity_path),
            "checkpoint_sha256": digests.digest(best_path),
            "predictions_sha256": digests.digest(predictions_path),
            "selected_epoch": int(result["best_epoch"]), "epochs_run": len(history),
            "loop_seconds": float(result.get("loop_seconds", 0)),
            "trainable_parameters": int(result.get("trainable_parameters", -1)),
            "native_metrics": metrics, "predictions": prediction,
            "verified_feature_hash_count": feature_count,
            "verified_implementation_hash_count": implementation_count,
            "hash_failures": feature_failures + implementation_failures}


def _source_draws(ids: list[str], *, seed: int = BOOTSTRAP_SEED,
                  replicates: int = BOOTSTRAP_REPLICATES) -> tuple[list[str], np.ndarray]:
    sources = sorted({sample_id.split("/", 1)[0] for sample_id in ids})
    index = {source: i for i, source in enumerate(sources)}
    source_index = np.asarray([index[sample_id.split("/", 1)[0]] for sample_id in ids], dtype=np.int32)
    rng = np.random.default_rng(seed)
    counts = np.zeros((replicates, len(sources)), dtype=np.uint16)
    for draw in range(replicates):
        counts[draw] = np.bincount(rng.integers(0, len(sources), size=len(sources)),
                                   minlength=len(sources)).astype(np.uint16)
    return sources, counts


def paired_source_delta(candidate: np.ndarray, reference: np.ndarray, target: np.ndarray,
                        ids: list[str], source_names: list[str], draw_counts: np.ndarray) -> dict[str, Any]:
    candidate = np.clip(np.asarray(candidate, dtype=np.float32), -1, 1)
    reference = np.clip(np.asarray(reference, dtype=np.float32), -1, 1)
    target = np.clip(np.asarray(target, dtype=np.float32), -1, 1)
    if candidate.shape != reference.shape or candidate.shape != target.shape or len(ids) != len(target):
        raise ValueError("Paired source bootstrap requires aligned predictions, targets and IDs")
    source_lookup = {source: index for index, source in enumerate(source_names)}
    row_source = np.asarray([source_lookup[sample_id.split("/", 1)[0]] for sample_id in ids], dtype=np.int32)
    errors = np.abs(candidate - target) - np.abs(reference - target)
    sums = np.bincount(row_source, weights=errors.astype(np.float64), minlength=len(source_names))
    counts = np.bincount(row_source, minlength=len(source_names)).astype(np.float64)
    draw_denominator = draw_counts @ counts
    delta = (draw_counts @ sums) / draw_denominator
    lower, upper = np.quantile(delta, [.025, .975])
    return {"candidate_minus_reference_MAE": float(np.mean(np.abs(candidate - target))
            - np.mean(np.abs(reference - target))),
            "paired_source_bootstrap_95_ci": [float(lower), float(upper)],
            "replicates": int(draw_counts.shape[0]), "source_count": len(source_names),
            "sample_count": int(len(ids))}


def _summary_by_seed(records: dict[str, dict[int, dict[str, Any]]]) -> dict[str, Any]:
    return {model: {str(seed): records[model][seed]["native_metrics"] for seed in SEEDS}
            for model in records}


def _compare_models(candidate: dict[int, dict[str, Any]], references: dict[str, dict[int, dict[str, Any]]],
                    *, ids: list[str], targets: np.ndarray, source_names: list[str],
                    draw_counts: np.ndarray) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for reference_name, reference in references.items():
        entries: dict[str, Any] = {}
        for seed in SEEDS:
            entries[str(seed)] = paired_source_delta(candidate[seed]["predictions"],
                                                       reference[seed]["predictions"], targets, ids,
                                                       source_names, draw_counts)
        pooled_ids = ids * len(SEEDS)
        pooled_target = np.concatenate([targets for _ in SEEDS])
        candidate_pooled = np.concatenate([candidate[seed]["predictions"] for seed in SEEDS])
        reference_pooled = np.concatenate([reference[seed]["predictions"] for seed in SEEDS])
        entries["pooled"] = paired_source_delta(candidate_pooled, reference_pooled, pooled_target,
                                                  pooled_ids, source_names, draw_counts)
        entries["all_seed_directions_favor_candidate"] = all(
            entries[str(seed)]["candidate_minus_reference_MAE"] < 0 for seed in SEEDS)
        entries["pooled_CI_upper_below_zero"] = entries["pooled"]["paired_source_bootstrap_95_ci"][1] < 0
        comparisons[reference_name] = entries
    return comparisons


def _make_audit_inputs(repo_root: Path, config: dict[str, Any], author: dict[str, dict[str, float]],
                       digests: DigestCache, rec: Recorder) -> tuple[dict[str, np.ndarray], list[str], np.ndarray, dict[str, Any]]:
    csv_path = (repo_root / config["author_csv"]).resolve()
    cache_dir = (repo_root / config["cache_dir"]).resolve()
    normalization_dir = (repo_root / config["normalization_cache_dir"]).resolve()
    cache_metadata_path = cache_dir / "metadata.json"
    normalization_metadata_path = normalization_dir / "metadata.json"
    stats_path = normalization_dir / "normalization.json"
    csv_hash = digests.digest(csv_path)
    rec.must("pinned official author CSV SHA256", csv_hash == EXPECTED_CSV_SHA256,
             f"actual={csv_hash}; expected={EXPECTED_CSV_SHA256}")
    rec.must("complete TRAIN/VALID author ID counts", {k: len(v) for k, v in author.items()} == EXPECTED_COUNTS,
             f"found { {k: len(v) for k, v in author.items()} }")
    config_ok = (config.get("dataset") == "simsv2" and config.get("seeds") == list(SEEDS)
                 and config.get("device") == "cuda" and config.get("batch_size") == 32
                 and config.get("learning_rate") == 1e-4 and config.get("weight_decay") == 1e-4
                 and config.get("max_epochs") == 80 and config.get("patience") == 12
                 and config.get("rank") == 32 and config.get("hidden") == 234
                 and config.get("dropout") == .1 and config.get("author_csv") == "data/author/simsv2/meta.csv"
                 and config.get("cache_dir") == "data/feature_cache/simsv2_native_timed_grid_v1"
                 and config.get("normalization_cache_dir") == "data/feature_cache/simsv2_native_pooled_standardized_v1"
                 and config.get("run_root") == "runs/native_timed_moments_v1")
    rec.must("registered study config contract", config_ok,
             "Config must retain the registered dataset, cache, seeds, model shape and training budget")

    cache_meta = _load_json(cache_metadata_path)
    norm_meta = _load_json(normalization_metadata_path)
    stats = _load_json(stats_path)
    timed_meta_hash = digests.digest(cache_metadata_path)
    rec.must("timed-grid cache metadata and split scope", cache_meta.get("format") == "native-timed-grid-v1"
             and cache_meta.get("dataset") == "simsv2" and cache_meta.get("test_exported") is False
             and cache_meta.get("counts") == EXPECTED_COUNTS,
             "Timed cache must be the pinned complete TRAIN/VALID grid only")
    rec.must("pooled normalizer provenance and TRAIN fit", norm_meta.get("author_csv_sha256") == csv_hash
             and norm_meta.get("counts") == EXPECTED_COUNTS
             and norm_meta.get("normalization", {}).get("fit_split") == "train"
             and norm_meta.get("normalization", {}).get("fit_count") == EXPECTED_COUNTS["train"]
             and norm_meta.get("test_exported") is False,
             "Pooled coordinates must be fitted on all 2,722 official TRAIN clips")
    timed_hashes, timed_failures = _verify_hash_map(cache_meta.get("artifact_sha256"), cache_dir,
                                                    "all timed-grid artifact hashes", digests,
                                                    rec=rec, repo_root=repo_root)
    normalizer_hashes, normalizer_failures = _verify_hash_map(norm_meta.get("artifact_sha256"), normalization_dir,
                                                              "all pooled-normalizer artifact hashes", digests,
                                                              rec=rec, repo_root=repo_root)
    stats_expected = norm_meta.get("artifact_sha256", {}).get("normalization.json")
    stats_actual = digests.digest(stats_path)
    rec.must("pooled normalizer JSON SHA256", stats_expected == stats_actual,
             f"actual={stats_actual}; declared={stats_expected}")

    timed_source_root = Path(cache_meta.get("source_cache", ""))
    norm_source_root = Path(norm_meta.get("source_cache", ""))
    rec.must("normalization and timed-grid common source cache", timed_source_root.resolve() == norm_source_root.resolve(),
             f"timed={timed_source_root}; pooled={norm_source_root}")
    source_hash_count = 0
    source_hash_failures: list[str] = []
    for split, mapping in cache_meta.get("source_feature_sha256", {}).items():
        preferred = timed_source_root / "splits" / split
        count, failures = _verify_hash_map(mapping, timed_source_root,
                                           f"timed-grid {split} source artifact hashes", digests,
                                           rec=rec, preferred=preferred, repo_root=repo_root)
        source_hash_count += count
        source_hash_failures.extend(failures)
    for split, mapping in norm_meta.get("source_feature_sha256", {}).items():
        preferred = norm_source_root / "splits" / split
        count, failures = _verify_hash_map(mapping, norm_source_root,
                                           f"pooled normalizer {split} source artifact hashes", digests,
                                           rec=rec, preferred=preferred, repo_root=repo_root)
        source_hash_count += count
        source_hash_failures.extend(failures)
    source_metadata_path = timed_source_root / "metadata.json"
    source_meta = _load_json(source_metadata_path)
    raw_meta_hash_manifest = source_meta.get("artifact_sha256", source_meta.get("files"))
    raw_meta_hash_count, raw_meta_hash_failures = _verify_hash_map(
        raw_meta_hash_manifest, timed_source_root, "raw source cache artifact hashes", digests,
        rec=rec, repo_root=repo_root)
    source_feature_count = source_hash_count + raw_meta_hash_count

    # The registered producer/source hashes also bind to the actual source files.
    code_binding_failures = []
    for relpath, expected in (("scripts/prepare_native_timed_grid.py", cache_meta.get("preparation_sha256")),
                              ("jevaffect/native_training.py", cache_meta.get("native_reader_sha256")),
                              ("scripts/prepare_native_pooled_standardization.py", norm_meta.get("preparation_sha256")),
                              ("jevaffect/anchor_calibration.py", norm_meta.get("normalization_implementation_sha256"))):
        path = repo_root / relpath
        actual = digests.digest(path) if path.is_file() else None
        if actual != expected:
            code_binding_failures.append(f"{relpath}: source metadata binding differs")
    rec.must("timed-grid and pooled-normalizer producer/reader hashes", not code_binding_failures,
             "; ".join(code_binding_failures) if code_binding_failures else "Producer, reader and normalization implementation hashes match current files")

    split_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_ids: dict[str, list[str]] = {}
    split_targets: dict[str, np.ndarray] = {}
    required_files = {"ids.json", "raw_labels.json", "text.npy", "audio_mean.npy", "vision_mean.npy",
                      "audio_grid.npy", "vision_grid.npy", "pair_valid.npy", "presence.npy", "grid_times.npy"}
    expected_artifacts = set(cache_meta.get("artifact_sha256", {}))
    files_present = True
    for split in ("train", "valid"):
        directory_rel = cache_meta.get("split_paths", {}).get(split, split)
        directory = cache_dir / directory_rel
        ids = _load_json(directory / "ids.json")
        raw_labels = np.asarray(_load_json(directory / "raw_labels.json"), dtype=np.float32)
        arrays = {name: np.load(directory / f"{name}.npy", allow_pickle=False)
                  for name in ("text", "audio_mean", "vision_mean", "audio_grid", "vision_grid",
                               "pair_valid", "presence", "grid_times")}
        split_ids[split] = ids
        split_targets[split] = raw_labels
        split_arrays[split] = arrays
        count = EXPECTED_COUNTS[split]
        expected_shapes = {"text": (count, 768), "audio_mean": (count, 768), "vision_mean": (count, 384),
                           "audio_grid": (count, 4, 768), "vision_grid": (count, 4, 384),
                           "pair_valid": (count,), "presence": (count, 3), "grid_times": (count, 4)}
        shapes_ok = len(ids) == count and len(set(ids)) == count and raw_labels.shape == (count,)
        shapes_ok &= all(arrays[name].shape == shape for name, shape in expected_shapes.items())
        shapes_ok &= all(arrays[name].dtype.kind == "f" and np.isfinite(arrays[name]).all()
                         for name in ("text", "audio_mean", "vision_mean", "audio_grid", "vision_grid"))
        shapes_ok &= arrays["pair_valid"].dtype.kind == "b" and arrays["presence"].dtype.kind == "b"
        shapes_ok &= arrays["grid_times"].dtype == np.dtype("float64")
        ids_ok = set(ids) == set(author[split]) and np.allclose(raw_labels,
                    np.asarray([author[split][sample_id] for sample_id in ids], dtype=np.float32), atol=1e-6, rtol=0)
        rec.must(f"timed-grid {split}: shapes, types, IDs and author targets", shapes_ok and ids_ok,
                 f"expected {count} official rows with T/A/V dimensions {EXPECTED_DIMS}")
        pair_valid = arrays["pair_valid"].astype(bool)
        presence = arrays["presence"].astype(bool)
        pairing_ok = not np.any(pair_valid & ~(presence[:, 1] & presence[:, 2]))
        pairing_ok &= int(pair_valid.sum()) == int(cache_meta.get("coverage", {}).get(split, {}).get("valid_pairs", -1))
        pairing_ok &= np.all(np.diff(arrays["grid_times"][pair_valid], axis=1) > 0)
        rec.must(f"timed-grid {split}: pair eligibility and increasing query times", pairing_ok,
                 f"valid pairs={int(pair_valid.sum())}; invalid rows retain global fallback")
        directory_files = {path.name for path in directory.iterdir() if path.is_file()}
        files_present &= required_files <= directory_files
        if split == "valid":
            rec.must("timed-grid VALID has no TEST inputs", not (cache_dir / "test").exists()
                     and not (cache_dir / "splits" / "test").exists(),
                     "The audit consumes only the stored official TRAIN and VALID splits")
    rec.must("timed-grid split artifact completeness", files_present,
             "Each split must contain IDs, labels, globals, four-row grids, masks and times")

    train_ids = split_ids["train"]
    train_arrays = split_arrays["train"]
    valid_ids = split_ids["valid"]
    valid_arrays = split_arrays["valid"]
    # Require normalization cache IDs/targets and arrays to mirror the raw global inputs.
    normalization_consistency = True
    normalized_global_arrays: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "valid"):
        directory = normalization_dir / split
        norm_ids = _load_json(directory / "ids.json")
        norm_targets = np.asarray(_load_json(directory / "raw_labels.json"), dtype=np.float32)
        normalization_consistency &= norm_ids == split_ids[split]
        normalization_consistency &= np.array_equal(norm_targets, split_targets[split])
        norm_features = {}
        presence = split_arrays[split]["presence"].astype(bool)
        for modality, raw_name, mask_index, dimension in (
            ("text", "text", 0, 768), ("audio", "audio_mean", 1, 768), ("vision", "vision_mean", 2, 384)):
            stored = np.load(directory / f"{modality}.npy", allow_pickle=False)
            raw_value = split_arrays[split][raw_name].astype(np.float32)
            standard = ((raw_value - np.asarray(stats[modality]["mean"], np.float32))
                        / np.asarray(stats[modality]["scale"], np.float32)).astype(np.float32)
            standard[~presence[:, mask_index]] = 0
            normalization_consistency &= stored.shape == (len(raw_value), 1, dimension)
            normalization_consistency &= np.allclose(stored[:, 0], standard, atol=1e-7, rtol=0)
            norm_features[raw_name] = standard
        normalized_global_arrays[split] = norm_features
    rec.must("pooled standardized cache matches independently transformed timed globals",
             normalization_consistency,
             "Pooled cache IDs/labels and all three transformed global modalities must align with the raw timed cache")

    train_normalized = _validate_stats_and_normalize(train_arrays, train_arrays["presence"].astype(bool),
                                                       train_arrays["pair_valid"].astype(bool), stats,
                                                       rec=rec, context="TRAIN", verify_fit=True)
    valid_normalized = _validate_stats_and_normalize(valid_arrays, valid_arrays["presence"].astype(bool),
                                                       valid_arrays["pair_valid"].astype(bool), stats,
                                                       rec=rec, context="VALID", verify_fit=False)
    # The independent TRAIN statistics are reused unchanged for VALID and all moment grids.
    inputs = {key: value for key, value in valid_normalized.items()}
    labels = np.asarray([author["valid"][sample_id] for sample_id in valid_ids], dtype=np.float32)
    rec.must("canonical no-overlap pair coverage", int(train_arrays["pair_valid"].sum()) == 2715
             and int(valid_arrays["pair_valid"].sum()) == 646
             and cache_meta.get("time_contract", {}).get("corpus_physical_synchrony") == "UNVERIFIED",
             "Use the frozen pair counts and retain the corpus physical-synchrony qualification")
    details = {
        "timed_cache_metadata_sha256": timed_meta_hash,
        "normalization_sha256": stats_actual,
        "normalization_metadata_sha256": digests.digest(normalization_metadata_path),
        "timed_artifact_hashes_verified": timed_hashes,
        "normalizer_artifact_hashes_verified": normalizer_hashes,
        "source_feature_hashes_verified": source_feature_count,
        "source_hash_failures": timed_failures + normalizer_failures + source_hash_failures + raw_meta_hash_failures,
        "raw_source_metadata_sha256": digests.digest(source_metadata_path),
        "raw_source_artifact_hashes_verified": raw_meta_hash_count,
        "normalization_train_count": len(train_ids),
        "valid_count": len(valid_ids),
        "train_pair_valid": int(train_arrays["pair_valid"].sum()),
        "valid_pair_valid": int(valid_arrays["pair_valid"].sum()),
        "timing_synchrony": cache_meta.get("time_contract", {}).get("corpus_physical_synchrony"),
        "source_cache": str(timed_source_root),
    }
    return inputs, valid_ids, labels, {"details": details, "cache_meta": cache_meta,
                                      "norm_meta": norm_meta, "stats": stats,
                                      "train_normalized": train_normalized,
                                      "valid_arrays": valid_arrays}


def audit(repo_root: Path, matrix_path: Path, config_path: Path, report_path: Path) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    matrix_path = Path(matrix_path)
    if not matrix_path.is_absolute():
        matrix_path = (repo_root / matrix_path).resolve()
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = (repo_root / report_path).resolve()
    rec = Recorder()
    digests = DigestCache()
    report: dict[str, Any] = {
        "audit": "independent-cpu-numpy-native-timed-moment-audit-v1",
        "scope": "Official CH-SIMS v2 TRAIN/VALID development evidence; no TEST targets, predictions, or scores",
        "matrix_path": str(matrix_path), "registered_config_path": str(config_path),
        "report_path": str(report_path), "complete": False, "promotion_gate_passed": False,
        "numpy_replay_tolerance": REPLAY_TOLERANCE,
        "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": BOOTSTRAP_SEED,
                      "unit": "whole VALID source video", "scope": "conditional selected-model development interval"},
        "checks": rec.checks, "findings": rec.findings,
    }
    try:
        config = _load_json(config_path)
        config_sha = digests.digest(config_path)
        plan_path = repo_root / "docs" / "NATIVE_TIMED_MOMENT_PLAN_V1.md"
        plan_sha = digests.digest(plan_path)
        training = repo_root / "jevaffect" / "timed_moment_training.py"
        training_source_hash = digests.digest(training)
        rec.must("registered plan and config hashes", config_sha == sha256_file(config_path)
                 and len(plan_sha) == 64,
                 "Config and frozen study plan must be readable and hash-bound")
        report["registered_hashes"] = {"config_sha256": config_sha, "plan_sha256": plan_sha,
                                       "training_implementation_sha256": training_source_hash}
        plan_text = plan_path.read_text(encoding="utf-8").lower()
        gate_statements_ok = ("5000 paired whole-source valid bootstrap" in " ".join(plan_text.split())
                              and "seed20261014" in "".join(plan_text.split())
                              and "candidate-minus-reference absolute error" in " ".join(plan_text.split())
                              and "common paired permutations" in " ".join(plan_text.split()))
        rec.must("registered comparison and intervention gates", gate_statements_ok,
                 "Frozen plan must state the source bootstrap, fixed seed and paired permutation intervention")
        author, author_order = _read_author_csv(repo_root / config["author_csv"])
        inputs, valid_ids, valid_targets, cache = _make_audit_inputs(repo_root, config, author, digests, rec)
        report["cache_provenance"] = cache["details"]
        report["canonical_valid_id_sha256"] = hashlib.sha256(
            json.dumps(valid_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
        report["author_train_id_count"] = len(author["train"])
        report["author_valid_id_count"] = len(author["valid"])

        matrix = _load_json(matrix_path)
        expected_matrix = {(case, seed) for case in CASES for seed in SEEDS}
        expected_list = [{"case": case, "seed": seed} for case in CASES for seed in SEEDS]
        actual_keys = {(str(row.get("case")), int(row.get("seed", -1))) for row in matrix.get("results", [])}
        matrix_ok = (matrix.get("dataset") == "simsv2" and matrix.get("matrix_complete") is True
                     and matrix.get("run_count") == 12 and matrix.get("complete_run_count") == 12
                     and matrix.get("smoke_only") is False and matrix.get("test_evaluated") is False
                     and all(row.get("complete") is True and row.get("smoke_only") is False
                             and row.get("test_evaluated") is False for row in matrix.get("results", []))
                     and matrix.get("expected_full_matrix") == expected_list
                     and actual_keys == expected_matrix and len(matrix.get("results", [])) == 12
                     and len(actual_keys) == len(matrix.get("results", [])))
        rec.must("12-fit formal matrix completion", matrix_ok,
                 f"matrix_complete={matrix.get('matrix_complete')}; unique fits={len(actual_keys)}; expected 12")
        report["formal_matrix"] = {key: matrix.get(key) for key in
                                   ("dataset", "run_count", "complete_run_count", "matrix_complete", "smoke_only", "test_evaluated")}

        initialization = _seeded_initialization_audit(repo_root, rec)
        report["initialization_policy"] = initialization
        fit_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        fit_rows = {(str(row["case"]), int(row["seed"])): row for row in matrix.get("results", [])}
        for case in CASES:
            for seed in SEEDS:
                row = fit_rows.get((case, seed))
                if row is None:
                    rec.must(f"{case}/seed_{seed}: formal result row", False, "Missing from matrix")
                    continue
                disk_result_path = repo_root / "runs" / "native_timed_moments_v1" / "formal" / case / f"seed_{seed}" / "result.json"
                disk_result = _load_json(disk_result_path) if disk_result_path.is_file() else None
                rec.must(f"{case}/seed_{seed}: matrix matches result artifact", disk_result == row,
                         "Formal matrix row must exactly match the case/seed result.json")
                fit_by_key[(case, seed)] = _audit_fit(repo_root, row, config, config_sha, author,
                                                       valid_ids, valid_targets, inputs, digests, rec)
        report["audited_fit_count"] = len(fit_by_key)
        report["per_fit"] = [{key: value for key, value in row.items()
                              if key not in ("predictions", "targets", "ids", "identity", "history", "best_state")}
                             for _, row in sorted(fit_by_key.items())]
        report["case_seed_metrics"] = {case: {str(seed): fit_by_key[(case, seed)]["native_metrics"]
                                              for seed in SEEDS if (case, seed) in fit_by_key}
                                       for case in CASES}

        if len(fit_by_key) != 12:
            raise ValueError("Not all 12 formal fits could be replayed")
        candidate = {seed: fit_by_key[("paired_cov", seed)] for seed in SEEDS}
        references: dict[str, dict[int, dict[str, Any]]] = {
            case: {seed: fit_by_key[(case, seed)] for seed in SEEDS}
            for case in ("static_product", "marginal_variance", "unpaired_cov")}
        baseline_rows: dict[str, dict[int, dict[str, Any]]] = {
            "conditioned_pooled_fusion": {}, "conditioned_sequence_almt": {}}
        baseline_specs = (
            ("conditioned_pooled_fusion", "configs/native_simsv2_standardized_fusion.json",
             "runs/native_simsv2_standardized_v1", "early_fusion", 492033),
            ("conditioned_sequence_almt", "configs/native_simsv2_sequence_standardized_almt.json",
             "runs/native_simsv2_sequence_standardized_v1", "almt", 3288577),
        )
        baseline_evidence: dict[str, Any] = {}
        csv_hash = digests.digest(repo_root / config["author_csv"])
        for name, baseline_config, run_root, model_name, param_count in baseline_specs:
            baseline_evidence[name] = {"config_path": baseline_config, "config_sha256": digests.digest(repo_root / baseline_config),
                                       "parameters": param_count, "seeds": {}}
            for seed in SEEDS:
                row = _audit_reference(repo_root, name=name, config_relative=baseline_config,
                                       run_root=run_root, model_name=model_name, expected_parameters=param_count,
                                       seed=seed, author_csv_sha=csv_hash, ids=valid_ids,
                                       targets=valid_targets, digests=digests, rec=rec)
                row["predictions"] = np.asarray(row["predictions"], dtype=np.float32)
                baseline_rows[name][seed] = row
                baseline_evidence[name]["seeds"][str(seed)] = {key: value for key, value in row.items()
                                                                 if key not in ("predictions",)}
            references[name] = baseline_rows[name]
        report["reference_baselines"] = baseline_evidence
        report["compute_budgets"] = {
            "timed_moment_models": {"trainable_parameters": PARAMETER_COUNT,
                                    "budget": {"batch_size": 32, "max_epochs": 80, "patience": 12,
                                               "learning_rate": 1e-4, "weight_decay": 1e-4},
                                    "loop_seconds_by_case_seed": {
                                        case: {str(seed): fit_by_key[(case, seed)]["loop_seconds"] for seed in SEEDS}
                                        for case in CASES}},
            "conditioned_pooled_fusion": {"trainable_parameters": 492033,
                                          "relative_parameter_delta_vs_timed": (492033 - PARAMETER_COUNT) / PARAMETER_COUNT,
                                          "loop_seconds_by_seed": {str(seed): baseline_rows["conditioned_pooled_fusion"][seed]["loop_seconds"] for seed in SEEDS}},
            "conditioned_sequence_almt": {"trainable_parameters": 3288577,
                                           "loop_seconds_by_seed": {str(seed): baseline_rows["conditioned_sequence_almt"][seed]["loop_seconds"] for seed in SEEDS}},
        }

        sources, draw_counts = _source_draws(valid_ids)
        report["source_cluster_bootstrap"] = {
            "source_count": len(sources), "source_names_sha256": _json_hash(sources),
            "source_names": sources, "draws_shared_across_comparisons_and_seeds": True,
            "candidate": "paired_cov", "delta_definition": "candidate native absolute error minus reference native absolute error",
            "comparisons": _compare_models(candidate, references, ids=valid_ids, targets=valid_targets,
                                             source_names=sources, draw_counts=draw_counts),
        }

        interventions: dict[str, Any] = {"common_permutation": {}, "vision_roll_1": {}}
        shift_predictions: dict[int, np.ndarray] = {}
        paired_predictions: dict[int, np.ndarray] = {}
        common_maxima: dict[str, float] = {}
        shift_metrics: dict[str, Any] = {}
        for seed in SEEDS:
            fit = candidate[seed]
            state = fit["best_state"]
            weights = _state_numpy(state)
            permuted = dict(inputs)
            permuted["audio_grid"] = np.roll(inputs["audio_grid"], 1, axis=1)
            permuted["vision_grid"] = np.roll(inputs["vision_grid"], 1, axis=1)
            common_output = numpy_forward(permuted, weights, "paired_cov")
            common_delta = float(np.max(np.abs(common_output - fit["predictions"].astype(np.float64))))
            common_maxima[str(seed)] = common_delta
            rec.must(f"paired_cov/seed_{seed}: common paired permutation invariant",
                     common_delta <= REPLAY_TOLERANCE,
                     f"max absolute output change={common_delta:.3g}; tolerance={REPLAY_TOLERANCE}")
            vision_shift = dict(inputs)
            vision_shift["vision_grid"] = np.roll(inputs["vision_grid"], 1, axis=1)
            shift_output = numpy_forward(vision_shift, weights, "paired_cov")
            shift_predictions[seed] = shift_output.astype(np.float32)
            paired_predictions[seed] = common_output.astype(np.float32)
            metrics = native_metrics(shift_output, valid_targets)
            baseline_mae = fit["native_metrics"]["MAE"]
            delta_mae = float(metrics["MAE"] - baseline_mae)
            seed_intervention = paired_source_delta(shift_output, fit["predictions"], valid_targets,
                                                     valid_ids, sources, draw_counts)
            shift_metrics[str(seed)] = {"baseline_native_metrics": fit["native_metrics"],
                                        "vision_roll_1_native_metrics": metrics,
                                        "native_MAE_increase": delta_mae,
                                        "paired_source_bootstrap_95_ci_for_MAE_increase": seed_intervention["paired_source_bootstrap_95_ci"],
                                        "source_count": len(sources)}
        interventions["common_permutation"] = {"operation": "cyclically roll A and V four-row grids together by one row; globals fixed",
                                                  "max_abs_output_change_by_seed": common_maxima,
                                                  "tolerance": REPLAY_TOLERANCE}
        shifted_rows = {}
        for seed in SEEDS:
            rows = []
            for index, sample_id in enumerate(valid_ids):
                rows.append({"id": sample_id, "source": sample_id.split("/", 1)[0],
                             "target": float(valid_targets[index]),
                             "paired_prediction": float(candidate[seed]["predictions"][index]),
                             "common_permuted_prediction": float(paired_predictions[seed][index]),
                             "vision_roll_1_prediction": float(shift_predictions[seed][index])})
            shifted_rows[str(seed)] = rows
        interventions["vision_roll_1"] = {"operation": "cyclically roll only the paired checkpoint's V grid by one row at inference; globals and A grid fixed",
                                           "metrics_and_source_intervals_by_seed": shift_metrics,
                                           "per_sample_outputs": shifted_rows}
        report["paired_cov_inference_interventions"] = interventions

        comparisons = report["source_cluster_bootstrap"]["comparisons"]
        permutation_gate = all(value <= REPLAY_TOLERANCE for value in common_maxima.values())
        gate_checks = []
        for reference_name, comparison in comparisons.items():
            seed_deltas = {str(seed): comparison[str(seed)]["candidate_minus_reference_MAE"]
                           for seed in SEEDS}
            pooled_ci = comparison["pooled"]["paired_source_bootstrap_95_ci"]
            comparison_pass = (comparison["all_seed_directions_favor_candidate"]
                               and comparison["pooled_CI_upper_below_zero"])
            gate_checks.append({
                "name": f"paired_cov vs {reference_name}: lower MAE in all seeds and pooled source-CI upper below zero",
                "passed": comparison_pass,
                "detail": {"candidate_minus_reference_MAE_by_seed": seed_deltas,
                           "pooled_three_seed_source_bootstrap_95_ci": pooled_ci,
                           "all_seed_directions_favor_candidate": comparison["all_seed_directions_favor_candidate"],
                           "pooled_CI_upper_below_zero": comparison["pooled_CI_upper_below_zero"]},
            })
        for seed in SEEDS:
            detail = shift_metrics[str(seed)]
            roll_pass = (detail["native_MAE_increase"] > 0
                         and detail["paired_source_bootstrap_95_ci_for_MAE_increase"][0] > 0)
            gate_checks.append({
                "name": f"paired_cov seed {seed}: inference V roll worsens native MAE with positive source-CI lower bound",
                "passed": roll_pass,
                "detail": {"native_MAE_increase": detail["native_MAE_increase"],
                           "paired_source_bootstrap_95_ci": detail["paired_source_bootstrap_95_ci_for_MAE_increase"]},
            })
        gate_checks.append({"name": "common paired permutation preserves checkpoint outputs", "passed": permutation_gate,
                            "detail": {"max_abs_output_change_by_seed": common_maxima,
                                       "tolerance": REPLAY_TOLERANCE}})
        gate = {"passed": all(item["passed"] for item in gate_checks), "checks": gate_checks,
                "interpretation": "Conditional VALID development gate; it does not establish originality, physical synchrony, or final-test performance."}
        report["promotion_gate"] = gate
        report["promotion_gate_passed"] = gate["passed"]
        report["recommendation"] = ("registered empirical gate passed; conduct separately preregistered source-held research only"
                                    if gate["passed"] else "retain the registered negative/conditional result and stop this covariance study")
        report["compute_budgets"]["paired_cov_checkpoints"] = {
            str(seed): {"selected_epoch": candidate[seed]["selected_epoch"],
                        "epochs_run": candidate[seed]["epochs_run"],
                        "loop_seconds": candidate[seed]["loop_seconds"]} for seed in SEEDS}
        report["audit_notes"] = [
            "All numeric forward replay and intervention outputs were computed on CPU; the independent model formula uses NumPy float64 and exact erf GELU.",
            "Stored A/V grid anchors preserve the registered decoder-time interpolation convention; corpus physical synchrony remains UNVERIFIED.",
            "Validation source intervals are conditional on already-selected models and do not adjust for repeated VALID use.",
            "The paired covariance operator is an established diagonal projected cross-moment; this audit makes no originality claim.",
        ]
        report["provenance_coverage"] = {
            "verified_file_hashes_total_cached": len(digests.values),
            "hash_manifest_failures": (cache["details"]["source_hash_failures"]
                                       + [item["issue"] for item in rec.findings if "hash" in item["check"].lower()]),
            "verified_all_candidate_source_inputs": not cache["details"]["source_hash_failures"],
            "formal_fits": len(fit_by_key),
            "audited_reference_runs": 6,
        }
        report["complete"] = not rec.findings and len(fit_by_key) == 12
    except (AssertionError, KeyError, OSError, ValueError, TypeError, IndexError,
            json.JSONDecodeError, RuntimeError) as exc:
        if not any(item.get("issue") == str(exc) for item in rec.findings):
            rec.findings.append({"check": "audit execution", "issue": str(exc)})
        report["error"] = str(exc)
    report["checks_passed"] = sum(bool(item["passed"]) for item in rec.checks)
    report["checks_failed"] = sum(not bool(item["passed"]) for item in rec.checks)
    report["findings"] = rec.findings
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--matrix", type=Path, default=Path("runs/native_timed_moments_v1/matrix.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/native_timed_moments_v1.json"))
    parser.add_argument("--report", type=Path, default=Path("docs/native_timed_moment_audit_20261014.json"))
    args = parser.parse_args(argv)
    result = audit(args.repo_root, args.matrix, args.config, args.report)
    print(json.dumps({key: result.get(key) for key in
                      ("complete", "audited_fit_count", "checks_passed", "checks_failed",
                       "promotion_gate_passed", "recommendation", "report_path", "error", "findings")},
                     indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
