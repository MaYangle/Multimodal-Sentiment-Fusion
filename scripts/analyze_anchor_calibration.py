"""TRAIN-OOF/official-VALID diagnostics for the registered AnchorCDF experiment.

Diagnostics explain development results; none identifies a causal source of
error or supplies an independent held-source pipeline evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.anchor_calibration import bin_targets, disagreement
from jevaffect.anchor_calibration_training import load_table
from jevaffect.native_training import digest, save_json


def summarize(data):
    errors = np.abs(data["anchor"] - data["target"])
    spread = np.ptp(data["views"], axis=1)
    return {
        "count": len(errors), "point_MAE": float(errors.mean()),
        "hit_rate": float(np.mean(errors <= np.float32(.1))),
        "anchor_absolute_mean": float(np.abs(data["anchor"]).mean()),
        "view_MAE": np.mean(np.abs(data["views"] - data["target"][:, None]), axis=0).tolist(),
        "spread_quantiles": np.quantile(spread, [0, .25, .5, .75, 1]).tolist(),
        "spread_error_Spearman": float(spearmanr(spread, errors).statistic),
        "differences_mean": disagreement(data["views"]).mean(0).tolist(),
    }


def row_losses(rows):
    m = np.array([r["prediction"] for r in rows], np.float32)
    y = np.array([r["target"] for r in rows], np.float32)
    p = np.array([r["p_hit"] for r in rows], np.float64)
    hit = (np.abs(m-y) <= np.float32(.1)).astype(np.float64)
    losses = {"hit_Brier": (p-hit)**2}
    if "bin_probabilities" in rows[0]:
        bins = np.array([r["bin_probabilities"] for r in rows], np.float64)
        losses["bin_log_score"] = -np.log(np.maximum(bins[np.arange(len(y)), bin_targets(y)], 1e-12))
    return losses


def source_point_comparison(tables, replicates=2000, bootstrap_seed=20261013):
    """Paired whole-source resampling, retaining clip-weighted MAE."""
    ids = tables[0]["ids"]
    if any(t["ids"] != ids for t in tables):
        raise ValueError("Cross-seed TRAIN OOF IDs differ")
    sources = np.array([sid.split("/")[0] for sid in ids])
    groups = sorted(set(sources))
    indices = [np.flatnonzero(sources == group) for group in groups]
    count = np.array([len(i) for i in indices])
    # Negative is a text improvement over fusion on the same held-source rows.
    delta = np.stack([np.abs(t["views"][:,0]-t["target"])-np.abs(t["anchor"]-t["target"]) for t in tables])
    sums = np.stack([delta[:, i].sum(1) for i in indices], axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    samples = rng.integers(0,len(groups),(replicates,len(groups)))
    boot = sums[:,samples].sum(2)/count[samples].sum(1)[None]
    return {"contrast":"text-minus-fusion native MAE; negative favors text", "sources":len(groups),
        "replicates":replicates,"seed":bootstrap_seed,
        "per_seed_delta":delta.mean(1).tolist(),"per_seed_percentile95":np.quantile(boot,[.025,.975],axis=1).T.tolist(),
        "three_seed_mean_delta":float(delta.mean()),"three_seed_mean_percentile95":np.quantile(boot.mean(0),[.025,.975]).tolist(),
        "scope":"Conditional on three selected point-teacher pipelines; model capacity/input dimensions differ, and this is TRAIN OOF development evidence"}


def analyze(suite_path, table_dir):
    suite = json.loads(Path(suite_path).read_text())
    expected = {(c, s) for c in ("global", "scale", "mixture", "views", "disagreement", "direct_hit", "mdn") for s in (13, 42, 2026)}
    results = suite["results"]
    if not suite["suite_complete"] or suite["smoke_only"] or len(results) != 21 or {(r["case"], r["seed"]) for r in results} != expected:
        raise ValueError("Complete registered formal suite required")
    records = {}
    for result in results:
        path = Path(result["root"]) / "valid_predictions.json"
        if digest(path) != result["predictions_sha256"]:
            raise ValueError("Prediction artifact changed")
        records[(result["case"], result["seed"])] = json.loads(path.read_text())
    diagnostics = {}; train_tables = []
    for seed in (13, 42, 2026):
        train, metadata = load_table(table_dir, seed, "train")
        train_tables.append(train)
        valid, _ = load_table(table_dir, seed, "valid")
        fit = np.array(metadata["calibration_split"]["fit_indices"])
        spread = np.ptp(valid["views"], axis=1)
        cuts = np.quantile(np.ptp(train["views"][fit], axis=1), [.25, .5, .75])
        strata = {
            "neutral": valid["target"] == 0,
            "weak": (np.abs(valid["target"]) > 0) & (np.abs(valid["target"]) <= np.float32(.4)),
            "strong": np.abs(valid["target"]) > np.float32(.4),
            **{f"spread_quartile_{j}": np.searchsorted(cuts, spread, side="left") == j for j in range(4)},
        }
        per_case = {}
        for case in ("global", "scale", "mixture", "views", "disagreement", "direct_hit", "mdn"):
            rows = records[(case, seed)]
            if [r["id"] for r in rows] != valid["ids"]:
                raise ValueError("Prediction IDs differ from canonical calibration table")
            losses = row_losses(rows)
            result = next(r for r in results if (r["case"], r["seed"]) == (case, seed))
            history_path = Path(result["root"]) / "history.json"
            item = {"strata": {name: {"count": int(mask.sum()), **{metric: float(values[mask].mean()) for metric, values in losses.items()}} for name, mask in strata.items() if mask.any()}}
            if history_path.exists():
                history = json.loads(history_path.read_text())
                item["learning"] = {"best_epoch": result["best_epoch"], "first_fit_objective": history[0]["fit_objective"],
                    "last_fit_objective": history[-1]["fit_objective"], "first_selection_objective": history[0]["selection_objective"],
                    "last_selection_objective": history[-1]["selection_objective"]}
            per_case[case] = item
        select_sources = set(metadata["calibration_split"]["select_sources"])
        teacher_fit = np.array([sid.split("/")[0] not in select_sources for sid in valid["ids"]])
        diagnostics[str(seed)] = {"train_OOF": summarize(train), "official_VALID": summarize(valid),
            "fit_OOF_spread_quartile_edges": cuts.tolist(), "case_diagnostics": per_case,
            "valid_teacher_source_membership": {role: {"count": int(mask.sum()),
                "point_MAE": float(np.abs(valid["anchor"][mask]-valid["target"][mask]).mean()),
                "spread_mean": float(spread[mask].mean())} for role, mask in (("fit_source", teacher_fit), ("selection_source", ~teacher_fit)) if mask.any()}}
    return {"suite_sha256": digest(suite_path), "table_metadata_sha256": digest(Path(table_dir)/"metadata.json"),
        "seeds": diagnostics, "source_OOF_point_comparison":source_point_comparison(train_tables), "test_evaluated": False,
        "scope": "Official VALID development diagnostics; no causal attribution, no nested outer pipeline or final-test claim"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--table-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    report = analyze(a.suite, a.table_dir)
    save_json(a.output, report)
    print(json.dumps({s: {k: r[k] for k in ("train_OOF", "official_VALID")} for s, r in report["seeds"].items()}, indent=2))
