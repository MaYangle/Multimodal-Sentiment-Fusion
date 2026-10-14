"""Copy a source-only checkout and bind every copied file by SHA256."""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args(); source = Path(__file__).resolve().parents[1]
    output = a.output_dir.resolve()
    if output.exists():
        raise ValueError("Use a new empty destination; source snapshots cannot be overwritten")
    output.mkdir(parents=True)
    for name in ("README.md", "README.zh-CN.md", "pyproject.toml", "LICENSE", "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", ".gitignore"):
        shutil.copy2(source / name, output / name)
    for name in ("jevaffect", "vendor", "scripts", "tests", "configs", "docs"):
        shutil.copytree(source / name, output / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {}
    for path in sorted(output.rglob("*")):
        if path.is_file():
            hashes[path.relative_to(output).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": str(source),
              "output": str(output), "scope": "code/docs/configs/tests only; no data, weights, environments or Git metadata",
              "files": hashes}
    (output / "source_snapshot.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files": len(hashes), "output": str(output)}))
