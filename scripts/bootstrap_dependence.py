"""Paired source-video cluster bootstrap, conditional on three frozen runs."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t
from jevaffect.native_training import digest, save_json
from jevaffect.student_pool import components, pool_quantile

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); a = p.parse_args()
    report = json.loads(a.results.read_text())
    if not report["complete"] or report["test_evaluated"]:
        raise ValueError("Frozen validation study required")
    root = Path(report["config"]["run_root"]) / "readouts"
    by_seed = {}; ids = None; hashes = {}
    for row in report["results"]:
        data = {}
        for name, known in row["methods"].items():
            path = root / f"seed_{row['seed']}" / (name + ".json"); records = json.loads(path.read_text())
            current_ids = [r["id"] for r in records]
            if ids is None:
                ids = current_ids
            if current_ids != ids:
                raise ValueError("Unpaired method/seed IDs")
            hashes[str(path)] = digest(path); y = np.array([r["target"] for r in records])
            if "nig" in records[0]:
                params = np.array([r["nig"] for r in records]); mu, v, alpha, beta = params.T
                df = 2 * alpha; scale = np.sqrt(beta * (1 + v) / (alpha * v))
                nll = -t.logpdf(y, df=df, loc=mu, scale=scale)
                quantile = t.ppf(.95, df=df); low, high = mu - scale * quantile, mu + scale * quantile
            else:
                params = np.array([r["expert_nig"] for r in records]); available = np.array([r["availability"] for r in records])
                mu, df, scale, w, _ = components(params, available)
                logw = np.full_like(w, -np.inf); np.log(w, out=logw, where=w > 0)
                nll = -logsumexp(logw + t.logpdf(y[:, None], df=df, loc=mu, scale=scale), axis=1)
                low, high = pool_quantile(params, available, .05), pool_quantile(params, available, .95)
            data[name] = {"NLL": nll, "coverage90": ((y >= low) & (y <= high)).astype(float), "width90": high - low}
            expected = known["uncertainty"]
            for key, value in (("NLL", expected["density_NLL"]), ("coverage90", expected["intervals"]["0.9"]["coverage"]), ("width90", expected["intervals"]["0.9"]["mean_width"])):
                if abs(data[name][key].mean() - value) > 1e-8:
                    raise ValueError("Readout metrics differ from completed report")
        by_seed[row["seed"]] = data
    sources = np.array([sid.split("/")[0] for sid in ids]); groups = sorted(set(sources))
    index = [np.flatnonzero(sources == g) for g in groups]; counts = np.array([len(i) for i in index])
    rng = np.random.default_rng(105); draws = rng.integers(len(groups), size=(2000, len(groups)))
    denominator = counts[draws].sum(1)
    output = {"scope": "source-video paired bootstrap on official valid, conditional on frozen selected models",
              "test_evaluated": False, "results_sha256": digest(a.results), "readout_sha256": hashes,
              "seed": 105, "replicates": 2000, "sources": len(groups), "clips": len(ids),
              "estimator": "clip-weighted mean after source resampling; percentile 95% interval",
              "limitation": "Does not include training-population, OOF fit, or validation-selection uncertainty; no multiplicity correction.", "contrasts": []}
    for baseline in ("global_NIG", "independent", "scaled_proper_pool"):
        for metric in ("NLL", "coverage90", "width90"):
            per_seed = {}
            diffs = [data["correlated"][metric] - data[baseline][metric] for data in by_seed.values()]
            for label, difference in list(zip(map(str, by_seed), diffs)) + [("mean_over_three_frozen_seeds", np.mean(diffs, axis=0))]:
                totals = np.array([difference[i].sum() for i in index])
                sampled = totals[draws].sum(1) / denominator
                per_seed[label] = {"difference": float(difference.mean()), "percentile95": np.quantile(sampled, [.025, .975]).tolist()}
            output["contrasts"].append({"candidate": "correlated", "baseline": baseline, "metric": metric, "results": per_seed})
    save_json(a.output, output); print(json.dumps(output["contrasts"], indent=2))
