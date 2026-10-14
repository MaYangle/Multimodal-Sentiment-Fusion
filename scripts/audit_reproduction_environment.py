"""Verify isolated dependency imports and an unchanged source snapshot."""
import argparse
import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--expected-source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); source = a.expected_source.resolve(); prefix = Path(sys.prefix).resolve()
    cfg = (prefix / "pyvenv.cfg").read_text()
    if sys.prefix == sys.base_prefix or "include-system-site-packages = false" not in cfg:
        raise ValueError("A new venv without system packages is required")
    modules = {}
    for name in ("numpy", "torch", "einops", "scipy", "transformers", "imageio_ffmpeg", "PIL"):
        module = importlib.import_module(name); path = Path(module.__file__).resolve()
        if not path.is_relative_to(prefix):
            raise ValueError("Dependency imported outside isolated environment: " + name)
        modules[name] = {"version": module.__version__, "path": str(path)}
    project = importlib.import_module("jevaffect")
    if not Path(project.__file__).resolve().is_relative_to(source):
        raise ValueError("Project imported from a different checkout")
    manifest_path = source / "source_snapshot.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != expected:
            raise ValueError("Snapshot source changed: " + name)
    torch = importlib.import_module("torch")
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "executable": sys.executable, "prefix": str(prefix), "base_prefix": sys.base_prefix,
              "platform": platform.platform(), "include_system_site_packages": False,
              "project_import": project.__file__, "dependencies": modules,
              "source_snapshot_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "source_files_verified": len(manifest["files"]), "cuda_available": torch.cuda.is_available(),
              "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "audit_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "passed": True}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
