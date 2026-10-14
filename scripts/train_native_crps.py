"""Fit the registered native scalar-score CRPS model matrix."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevaffect.native_crps_training import CASES, FOLDS, SEEDS, train
from jevaffect.native_training import digest, save_json
from jevaffect.nested_shape_training import load_stage


def _key(row):
    return (row["case"], row["seed"], row["fold"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--fold", type=int, choices=FOLDS)
    parser.add_argument("--allow-smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("cases") != list(CASES) or config.get("seeds") != list(SEEDS):
        raise ValueError("Registered native CRPS cases and seeds required")
    config["config_sha256"] = digest(args.config)
    plan_path = Path(config["plan"])
    if not plan_path.is_absolute():
        plan_path = Path(__file__).resolve().parents[1] / plan_path
    config["plan_sha256"] = digest(plan_path)
    suite_path = Path(config["point_suite"])
    if not suite_path.is_absolute():
        suite_path = Path(__file__).resolve().parents[1] / suite_path
    suite_document = json.loads(suite_path.read_text(encoding="utf-8"))
    suite_identity = {
        "config_sha256": config["config_sha256"],
        "plan_sha256": config["plan_sha256"],
        "point_suite_sha256": digest(suite_path),
        "smoke_only": bool(suite_document["smoke_only"]),
        "official_VALID_used": False,
        "official_TEST_used": False,
    }

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if output.exists():
        if not args.resume:
            raise ValueError("CRPS suite output exists; pass --resume to continue it")
        previous = json.loads(output.read_text(encoding="utf-8"))
        if previous.get("suite_identity") != suite_identity:
            raise ValueError("CRPS suite output identity changed")
        for row in previous.get("results", []):
            key = _key(row)
            if key in prior:
                raise ValueError("Duplicate model key in prior CRPS suite output")
            prior[key] = row

    selected_seeds = [args.seed] if args.seed is not None else list(SEEDS)
    selected_folds = [args.fold] if args.fold is not None else list(FOLDS)
    selected_cases = [args.case] if args.case is not None else list(CASES)
    for seed in selected_seeds:
        for fold in selected_folds:
            tables, binding = load_stage(suite_path, seed, fold, args.allow_smoke)
            for case in selected_cases:
                result = train(config, tables, binding, case=case, seed=seed, fold=fold,
                               resume=args.resume, stop_after=args.stop_after)
                prior[_key(result)] = result
                rows = list(prior.values())
                expected = {(case_name, current_seed, current_fold)
                            for case_name in CASES for current_seed in SEEDS for current_fold in FOLDS}
                present = {_key(row) for row in rows}
                suite_complete = (
                    present == expected
                    and all(row["complete"] and not row["smoke_only"] and row["quadrature_gate_pass"] for row in rows)
                )
                save_json(output, {
                    "suite_identity": suite_identity,
                    "results": sorted(rows, key=lambda row: (row["seed"], row["fold"], CASES.index(row["case"]))),
                    "suite_complete": suite_complete,
                    "formal_fit_count": sum(bool(row["complete"] and not row["smoke_only"]) for row in rows),
                    "expected_formal_fit_count": len(expected),
                    "quadrature_gate_pass": all(row["quadrature_gate_pass"] for row in rows if row["complete"]),
                    "official_VALID_used": False,
                    "official_TEST_used": False,
                })


if __name__ == "__main__":
    main()
