"""Compare against revision-pinned, publicly published author metric code."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import requests
import torch
from jevaffect.native_metrics import native_metrics


def main():
    records = []
    root = Path("work/native_protocol_reference")
    root.mkdir(parents=True, exist_ok=True)
    specs = [("simsv2", "thuiar/ch-sims-v2", "main", "utils/metricsTop.py", "SIMS2"),
             ("mosei", "thuiar/MMSA", "master", "src/MMSA/utils/metricsTop.py", "MOSEI")]
    rng = np.random.default_rng(105)
    for dataset, repo, branch, file, key in specs:
        response = requests.get(f"https://api.github.com/repos/{repo}/commits/{branch}", timeout=30)
        response.raise_for_status()
        revision = response.json()["sha"]
        url = f"https://raw.githubusercontent.com/{repo}/{revision}/{file}"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        local = root / (dataset + "_metrics.py")
        local.write_bytes(response.content)
        spec = importlib.util.spec_from_file_location(dataset + "_author_metrics", local)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference = module.MetricsTop("regression").getMetics(key)
        cases = []
        for dtype in (np.float32, np.float64):
            bound = 1 if dataset == "simsv2" else 3
            y = rng.uniform(-bound, bound, 200).astype(dtype)
            p = rng.uniform(-4, 4, 200).astype(dtype)
            y[:12] = np.asarray([-1, -.7, -.4, -.1, 0, .1, .4, .7, 1, .5, -.5, 0], dtype=dtype)
            p[:12] = y[:12]
            # Boundary perturbations expose open/closed intervals and rounding.
            p[12:24] = np.nextafter(y[:12], np.asarray(np.inf, dtype=dtype))
            expected = reference(torch.from_numpy(p), torch.from_numpy(y))
            actual = native_metrics(p, y, dataset)
            for name, value in expected.items():
                # MOSEI author code rounds to four decimals; SIMSv2 doesn't.
                tolerance = 5.1e-5 if dataset == "mosei" else 1e-6
                if actual[name] is None or not np.isclose(actual[name], value, rtol=0, atol=tolerance):
                    raise AssertionError(f"{dataset} {dtype} {name}: ours={actual[name]}, author={value}")
            cases.append({"dtype": str(dtype), "count": len(y), "metrics_checked": list(expected), "passed": True})
        records.append({"dataset": dataset, "repo": repo, "revision": revision, "source_url": url,
                        "source_sha256": hashlib.sha256(response.content).hexdigest(), "cases": cases})
    out = Path("docs/native_metric_conformance.json")
    out.write_text(json.dumps({"references": records, "passed": True, "test_dataset_scored": False}, indent=2), encoding="utf-8")
    print(out.read_text())


if __name__ == "__main__":
    main()
