"""CPU-only, target-free routing diagnostics for the three completed seed-13 arms.

VALID M is read only in this script and stays outside model.forward. The oracle
route is a privileged-label diagnostic, not a deployable predictor or a model
selection rule. No TRAIN channel labels or TEST records are loaded.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_sequence_gates import CacheSplit, independent_native_metrics, sha256_file
from jevaffect.sequence_gates import MODALITIES, SequenceGateRegressor

CASES = ("post_norm", "distilled_post_norm", "distilled_static")
SHIFTS = (1, 17, 113)
EPS = 1e-12


def read_valid_m(path: Path) -> tuple[list[str], np.ndarray]:
    """Parse only VALID native M and IDs; do not convert any other score field."""
    ids: list[str] = []
    targets: list[float] = []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "clip_id", "mode", "label"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("Author CSV does not contain the required VALID M fields")
        for row in reader:
            if row["mode"] == "valid":
                ids.append(row["video_id"] + "/" + row["clip_id"])
                targets.append(float(row["label"]))
    values = np.asarray(targets, np.float32)
    if len(ids) != 647 or len(set(ids)) != 647 or not np.isfinite(values).all() or np.any(np.abs(values) > 1):
        raise ValueError("Expected 647 unique finite native VALID M targets")
    return ids, values


def masked_softmax_oracle(scores: np.ndarray, target: np.ndarray, present: np.ndarray,
                          temperature: float = 0.3) -> np.ndarray:
    """Compute privileged R* outside the model, with zero mass on absent inputs."""
    scores = np.asarray(scores, np.float64)
    target = np.asarray(target, np.float64).reshape(-1)
    present = np.asarray(present, bool)
    if scores.shape != present.shape or scores.shape != (len(target), 3):
        raise ValueError("Oracle score, target and availability shapes do not align")
    logits = -float(temperature) * np.square(scores - target[:, None])
    route = np.zeros_like(logits, dtype=np.float64)
    for i in range(len(route)):
        keep = present[i]
        if not keep.any():
            continue
        z = logits[i, keep] - np.max(logits[i, keep])
        p = np.exp(z)
        route[i, keep] = p / p.sum()
    return route.astype(np.float32)


def manual_mean_kl(teacher: np.ndarray, policy: np.ndarray, present: np.ndarray) -> float:
    """Mean KL(teacher||policy) over observed rows, computed outside the model."""
    teacher = np.asarray(teacher, np.float64)
    policy = np.asarray(policy, np.float64)
    present = np.asarray(present, bool)
    if teacher.shape != policy.shape or teacher.shape != present.shape:
        raise ValueError("KL inputs must have the same [N,3] shape")
    row_losses = []
    for i in range(len(teacher)):
        keep = present[i]
        if not keep.any():
            continue
        q = teacher[i, keep]
        p = np.maximum(policy[i, keep], EPS)
        p /= p.sum()
        positive = q > 0
        row_losses.append(float(np.sum(q[positive] * (np.log(q[positive]) - np.log(p[positive])))))
    return float(np.mean(row_losses)) if row_losses else 0.0


def availability_groups(present: np.ndarray) -> tuple[dict[str, list[int]], dict[str, int]]:
    groups: dict[str, list[int]] = {}
    for i, row in enumerate(np.asarray(present, bool)):
        key = "".join("1" if value else "0" for value in row)
        groups.setdefault(key, []).append(i)
    return groups, {key: len(indices) for key, indices in groups.items()}


def cyclic_policy_shift(policy: np.ndarray, present: np.ndarray, offset: int) -> np.ndarray:
    """Move policy ratios across clips only within identical availability groups."""
    policy = np.asarray(policy, np.float32)
    present = np.asarray(present, bool)
    shifted = np.zeros_like(policy)
    groups, _ = availability_groups(present)
    for indices in groups.values():
        idx = np.asarray(indices, np.int64)
        source = np.roll(idx, int(offset) % max(len(idx), 1))
        shifted[idx] = policy[source]
    if np.any(shifted[~present] != 0):
        raise AssertionError("Cyclic policy move introduced mass on an absent modality")
    return shifted


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _round_list(values, digits: int = 8):
    return np.round(np.asarray(values, np.float64), digits).tolist()


def _run_probe(model, cache: CacheSplit, ratios: np.ndarray, batch_size: int = 16) -> np.ndarray:
    if model.training:
        raise ValueError("Diagnostic probes require eval mode")
    if ratios.shape != (len(cache.ids), 3):
        raise ValueError("Probe route must have one [3] row per VALID clip")
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(cache.ids), batch_size):
            end = min(start + batch_size, len(cache.ids))
            features = cache.batch(np.arange(start, end, dtype=np.int64))
            route = torch.from_numpy(np.asarray(ratios[start:end], np.float32))
            output.append(model.probe(features, route)["prediction"].detach().cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def _all_absent_smoke(model, cache: CacheSplit) -> dict:
    features = cache.batch(np.asarray([0], np.int64))
    for modality in MODALITIES:
        features[modality] = torch.zeros_like(features[modality])
        features["lengths"][modality] = torch.zeros_like(features["lengths"][modality])
    model.eval()
    with torch.inference_mode():
        output = model(features)
        probe = model.probe(features, torch.zeros((1, 3), dtype=output["prediction"].dtype))
    passed = (
        torch.isfinite(output["prediction"]).all().item()
        and torch.isfinite(output["policy_ratios"]).all().item()
        and torch.isfinite(probe["prediction"]).all().item()
        and torch.count_nonzero(output["ratios"]).item() == 0
        and torch.count_nonzero(output["policy_ratios"]).item() == 0
    )
    return {"passed": bool(passed), "policy_mass": output["policy_ratios"].cpu().tolist(),
            "routed_mass": output["ratios"].cpu().tolist(), "prediction_finite": bool(torch.isfinite(output["prediction"]).all())}


def _compare_metrics(prediction: np.ndarray, target: np.ndarray, deploy_mae: float) -> dict:
    metrics = independent_native_metrics(prediction, target)
    return {"metrics": metrics, "delta_MAE_vs_deploy": float(metrics["MAE"] - deploy_mae)}


def diagnose(root: Path, report_path: Path) -> dict:
    root = root.resolve()
    cfg_path = root / "configs/native_sequence_gates_v1.json"
    cfg = _load_json(cfg_path)
    data_root = (root / cfg["standardized_cache"]).resolve()
    cache = CacheSplit(data_root, "valid")
    author_path = (root / cfg["author_csv"]).resolve()
    valid_ids = cache.ids
    if len(valid_ids) != 647 or len(set(valid_ids)) != 647:
        raise ValueError("Expected 647 unique VALID feature IDs")
    if sha256_file(author_path) != cfg["author_csv_sha256"]:
        raise ValueError("Author CSV hash differs from the registered SIMSv2 source")

    metadata = _load_json(data_root / "metadata.json")
    dims = {m: int(metadata["dimensions"][m]) for m in MODALITIES}
    id_counts: dict[str, int] = {}
    report: dict = {
        "schema": "native_sequence_gate_teacher_mechanism_diagnostic_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "diagnostic_only", "root": str(root),
        "runtime": {
            "python_version": sys.version.split()[0], "python_executable": sys.executable,
            "numpy_version": np.__version__, "torch_version": str(torch.__version__),
            "torch_path": str(Path(torch.__file__).resolve()), "device": "cpu",
            "mha_fastpath_enabled": bool(torch.backends.mha.get_fastpath_enabled()),
            "gpu_context_created": False, "cuda_device_or_encoder_used": False,
        },
        "input_bindings": {"config_sha256": sha256_file(cfg_path), "author_csv_sha256": sha256_file(author_path),
                           "standardized_cache_metadata_sha256": sha256_file(data_root / "metadata.json"),
                           "valid_rows": len(valid_ids), "valid_unique_ids": len(set(valid_ids)),
                           "valid_M_parsed": False,
                           "valid_TAV_parsed": False, "test_scores_parsed": False, "test_feature_split_opened": False},
        "diagnostic_contract": {
            "deployment_forward_calls_are_feature_only": True,
            "VALID_M_passed_as_target_argument": False,
            "VALID_M_used_externally_to_construct_privileged_oracle_R": True,
            "oracle_R_passed_only_to_explicit_eval_probe": True,
            "oracle_route_is_privileged_label_diagnostic": True,
            "oracle_is_not_deployable_not_benchmark_not_selection": True,
            "no_forward_features_target_call": True,
            "model_weights_or_training_plan_changed": False,
            "same_647_VALID_clips_used_for_all_rows": True,
        },
        "cycle_offsets_within_availability_strata": list(SHIFTS),
        "models": {},
    }
    clip_report: dict[str, list] = {}

    for case in CASES:
        run_dir = root / cfg["run_root"] / "fusion" / "seed_13" / case
        identity_path = run_dir / "identity.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        result_path = run_dir / "result.json"
        predictions_path = run_dir / "valid_predictions.json"
        if not all(path.is_file() for path in (identity_path, checkpoint_path, result_path, predictions_path)):
            raise FileNotFoundError(f"Seed13 {case} is not a completed formal case")
        identity = _load_json(identity_path)
        result = _load_json(result_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not result.get("complete") or identity.get("smoke_only") or not checkpoint.get("complete"):
            raise ValueError(f"Seed13 {case} is incomplete or smoke-only")
        if checkpoint.get("case") != case or checkpoint.get("seed") != 13:
            raise ValueError("Checkpoint case/seed does not match the registered diagnostic target")

        model = SequenceGateRegressor(dims, case, **cfg["model"])
        model.load_state_dict(checkpoint["best_state"], strict=True)
        model.eval()

        # Pass one is strictly target-free: y_M is not read or supplied until
        # all model outputs and availability masks have been collected.
        pred_parts, score_parts, policy_parts, route_parts, present_parts = [], [], [], [], []
        with torch.inference_mode():
            for start in range(0, len(valid_ids), 16):
                end = min(start + 16, len(valid_ids))
                indices = np.arange(start, end, dtype=np.int64)
                features = cache.batch(indices)
                output = model(features)
                pred_parts.append(output["prediction"].cpu().numpy())
                score_parts.append(output["unimodal_scores"].cpu().numpy())
                policy_parts.append(output["policy_ratios"].cpu().numpy())
                route_parts.append(output["ratios"].cpu().numpy())
                present_parts.append(np.stack([cache.lengths[m][indices] > 0 for m in MODALITIES], axis=1))
        deployed_prediction = np.concatenate(pred_parts).astype(np.float32)
        unimodal_scores = np.concatenate(score_parts).astype(np.float32)
        policy = np.concatenate(policy_parts).astype(np.float32)
        deployed_ratios = np.concatenate(route_parts).astype(np.float32)
        present = np.concatenate(present_parts).astype(bool)
        groups, group_counts = availability_groups(present)
        if np.any(deployed_ratios[~present] != 0) or np.any(policy[~present] != 0):
            raise ValueError(f"{case} assigned nonzero ratio to a missing modality")

        # Load M only after the target-free predictions have been collected.
        # It is used below in NumPy and metric code, never in a model primitive.
        valid_label_ids, targets = read_valid_m(author_path)
        if valid_label_ids != valid_ids:
            raise ValueError("VALID feature IDs and VALID M IDs are not in the same order")
        report["input_bindings"]["valid_M_parsed"] = True
        oracle = masked_softmax_oracle(unimodal_scores, targets, present, temperature=0.3)
        uniform_third = present.astype(np.float32) / 3.0
        uniform_one = present.astype(np.float32)
        shifted_policy = {offset: cyclic_policy_shift(policy, present, offset) for offset in SHIFTS}

        baseline_rows = _load_json(predictions_path)
        baseline_by_id = {row["id"]: float(row["prediction"]) for row in baseline_rows}
        max_checkpoint_delta = max(abs(float(baseline_by_id[sid]) - float(deployed_prediction[i])) for i, sid in enumerate(valid_ids))
        if set(baseline_by_id) != set(valid_ids) or max_checkpoint_delta > 1e-5:
            raise ValueError(f"{case}: target-free deployment output does not reproduce selected predictions")

        scenarios = {
            "deploy": (deployed_prediction, deployed_ratios),
            "uniform_present_R_1_over_3": (_run_probe(model, cache, uniform_third), uniform_third),
            "uniform_present_R_1": (_run_probe(model, cache, uniform_one), uniform_one),
            "privileged_VALID_M_oracle_R": (_run_probe(model, cache, oracle), oracle),
        }
        for offset, shifted in shifted_policy.items():
            scenarios[f"policy_R_cycle_shift_{offset}"] = (_run_probe(model, cache, shifted), shifted)

        deploy_metrics = independent_native_metrics(deployed_prediction, targets)
        scenario_report = {}
        per_clip_rows = []
        for i, sid in enumerate(valid_ids):
            probes = {name: float(prediction[i]) for name, (prediction, _) in scenarios.items() if name != "deploy"}
            per_clip_rows.append({
                "id": sid,
                "target_M_external_privileged_diagnostic": float(targets[i]),
                "present_mask_TAV": [bool(x) for x in present[i]],
                "unimodal_scores_TAV": _round_list(unimodal_scores[i]),
                "deployed_route_R": _round_list(deployed_ratios[i]),
                "policy_R_target_free": _round_list(policy[i]),
                "oracle_R_privileged_VALID_M": _round_list(oracle[i]),
                "oracle_policy_KL_row": float(manual_mean_kl(oracle[i:i + 1], policy[i:i + 1], present[i:i + 1])),
                "scenario_predictions": {"deploy": float(deployed_prediction[i]), **probes},
                "cycle_policy_R": {str(offset): _round_list(shifted_policy[offset][i]) for offset in SHIFTS},
            })
        for name, (prediction, ratio) in scenarios.items():
            metric_record = _compare_metrics(prediction, targets, deploy_metrics["MAE"])
            route_mass = ratio.sum(axis=1)
            metric_record["route_summary"] = {
                "mean_R_TAV": _round_list(ratio.mean(axis=0)),
                "mean_total_R": float(route_mass.mean()),
                "absent_ratio_max": float(np.max(np.abs(ratio[~present]))) if np.any(~present) else 0.0,
            }
            scenario_report[name] = metric_record

        oracle_kl = manual_mean_kl(oracle, policy, present)
        entropy = -np.sum(np.where(oracle > 0, oracle * np.log(np.maximum(oracle, EPS)), 0.0), axis=1)
        result_report = {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "identity_sha256": identity.get("identity_sha256"),
            "best_epoch": int(checkpoint["best_epoch"]),
            "registered_policy_status": identity.get("policy_head", {}).get("status"),
            "case_route_at_deployment": "learned policy" if case == "distilled_post_norm" else "R=1 per present modality",
            "policy_route_interpretation": {
                "post_norm": "dormant, untrained policy head; policy-cycle shifts are a random-head negative control",
                "distilled_static": "teacher-trained auxiliary policy; registered inference does not execute it",
                "distilled_post_norm": "teacher-trained policy used by the registered target-free inference path",
            }[case],
            "prediction_replay_max_abs_delta_vs_saved": float(max_checkpoint_delta),
            "valid_clip_count": len(valid_ids), "availability_strata": group_counts,
            "deployment_metrics": deploy_metrics,
            "scenarios": scenario_report,
            "privileged_oracle_summary": {
                "formula": "softmax(-0.3 * (unimodal_score - VALID_native_M)^2) over present modalities",
                "computed_outside_model_forward": True,
                "mean_KL_oracle_to_target_free_policy": oracle_kl,
                "mean_oracle_entropy_nats": float(entropy[present.any(1)].mean()) if present.any(1).any() else 0.0,
                "oracle_MAE_delta_vs_deploy": float(scenario_report["privileged_VALID_M_oracle_R"]["delta_MAE_vs_deploy"]),
                "interpretation": "VALID labels grant privileged routing information; an oracle change is explanatory only and is not a performance claim or selectable checkpoint.",
            },
            "policy_shift_offsets_preserve_availability": list(SHIFTS),
            "all_absent_safety": _all_absent_smoke(model, cache),
            "per_clip_rows": len(per_clip_rows),
        }
        report["models"][case] = result_report
        clip_report[case] = per_clip_rows
        id_counts[case] = len(per_clip_rows)

    report["per_clip_counts"] = id_counts
    report["per_clip_diagnostics"] = clip_report
    report["interpretation_guardrail"] = (
        "Oracle routing uses VALID native M after target-free scores are produced and is a privileged-label diagnostic only. "
        "Do not describe it as deployable, benchmark performance, evidence for model selection, or a trained performance improvement. "
        "If the oracle route also fails to improve a frozen fuser, that supports a mismatch between the score-error teacher and the fuser's actual path on this checkpoint; it does not prove every architecture is ineffective."
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temp_report.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temp_report.replace(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report", type=Path, default=ROOT / "docs/native_sequence_gate_teacher_diagnostic_20261014.json")
    args = parser.parse_args(argv)
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.mha.set_fastpath_enabled(False)
    report_path = args.report if args.report.is_absolute() else args.root / args.report
    report = diagnose(args.root, report_path)
    print(json.dumps({"status": report["status"], "models": list(report["models"]),
                      "per_clip_counts": report["per_clip_counts"], "report": str(report_path.resolve())}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
