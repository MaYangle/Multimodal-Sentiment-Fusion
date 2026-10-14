"""Inventory public author-linked Drive folders without downloading payloads."""
import argparse
import dataclasses
import json
from pathlib import Path
import gdown

FOLDERS = {
    "simsv2": ("1wFvGS0ebKRvT3q6Xolot-sDtCNfz7HRA", "https://github.com/thuiar/ch-sims-v2"),
    "multibench_mosei": ("1A_hTmifi824gypelGobgl2M-5Rw9VWHv", "https://github.com/pliang279/MultiBench"),
    "legacy_checkpoints": ("1deCsD3TXacpuov78v7PldhXL5zFSjNjL", "https://github.com/MaYangle/Multimodal-Sentiment-Analysis"),
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=FOLDERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ident, publisher = FOLDERS[args.source]
    report = {"source": args.source, "publisher": publisher, "folder_id": ident, "payload_downloaded": False}
    try:
        files = gdown.download_folder(id=ident, skip_download=True, quiet=True, use_cookies=False, timeout=30)
        if files is None:
            raise RuntimeError("Folder inventory returned no result")
        report["files"] = [dataclasses.asdict(x) if dataclasses.is_dataclass(x) else x._asdict() for x in files]
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
