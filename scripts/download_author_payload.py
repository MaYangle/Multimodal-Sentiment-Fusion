"""Resume a payload selected from a recorded public author-folder inventory."""
from __future__ import annotations
import argparse
import hashlib
import inspect
import json
from urllib.parse import urlencode
from datetime import datetime, timezone
from pathlib import Path
import gdown


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--direct", action="store_true", help="Use the public usercontent endpoint, retaining resumable partials")
    p.add_argument("--read-timeout", type=int, default=40)
    p.add_argument("--retries", type=int, default=1)
    args = p.parse_args()
    if Path(args.name).name != args.name or args.name in (".", ".."):
        raise ValueError("Choose a safe file basename")
    inventory = json.loads(args.inventory.read_text())
    selected = [r for r in inventory["files"] if r["path"] == args.name]
    if len(selected) != 1:
        raise ValueError("Exactly one recorded author file must match")
    file = selected[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / args.name
    receipt = target.with_suffix(target.suffix + ".provenance.json")
    if target.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        with target.open("rb") as f:
            current = hashlib.file_digest(f, "sha256").hexdigest()
        if saved["file_id"] != file["id"] or saved["sha256"] != current:
            raise ValueError("Existing payload identity/hash changed")
        print(json.dumps({"cached": True, "bytes": target.stat().st_size, "sha256": current}), flush=True)
        return
    if target.exists():
        raise ValueError("Complete-looking payload without receipt; inspect rather than overwrite")
    state = {"last_bucket": -1, "observed_total_bytes": None}

    def progress(current, total):
        state["observed_total_bytes"] = total
        bucket = current // (64 * 1024 * 1024)
        if bucket != state["last_bucket"]:
            state["last_bucket"] = bucket
            print(json.dumps({"received_bytes": current, "total_bytes": total}), flush=True)

    if args.read_timeout < 1 or args.retries < 0:
        raise ValueError("Invalid transport retry/timeout")
    options = dict(id=file["id"], output=str(target), use_cookies=False,
                   resume=True, timeout=(30, args.read_timeout), retries=args.retries, quiet=True)
    transport = {"mode": "gdown publisher file ID"}
    if args.direct:
        import requests
        url = "https://drive.usercontent.google.com/download?" + urlencode({"id": file["id"], "export": "download", "confirm": "t"})
        partials = list(args.output_dir.glob(args.name + "*.part"))
        if len(partials) > 1:
            raise ValueError("Ambiguous partial files; inspect before resuming")
        existing = partials[0] if partials else None
        check_bytes = min(existing.stat().st_size, 65536) if existing else 1
        with requests.get(url, headers={"Range": f"bytes=0-{max(check_bytes, 1)-1}"}, stream=True, timeout=(30, args.read_timeout)) as response:
            expected_prefix = f"bytes 0-{max(check_bytes, 1)-1}/"
            content_range = response.headers.get("Content-Range", "")
            if response.status_code != 206 or not content_range.startswith(expected_prefix):
                problem = {"file_id": file["id"], "stage": "direct prefix preflight", "status": response.status_code,
                           "Content-Range": content_range, "Content-Type": response.headers.get("Content-Type"),
                           "Content-Length": response.headers.get("Content-Length"), "transfer_complete": False}
                if "text/html" in (response.headers.get("Content-Type") or ""):
                    preview = next(response.iter_content(4096), b"").lower()
                    problem["quota_text_detected"] = b"quota" in preview or b"too many users" in preview
                receipt.with_suffix(receipt.suffix + ".preflight_failure.json").write_text(json.dumps(problem, indent=2), encoding="utf-8")
                raise ValueError("Public endpoint did not return exact requested prefix")
            prefix_bytes = bytearray()
            for chunk in response.iter_content(max(check_bytes, 1)):
                prefix_bytes.extend(chunk)
                if len(prefix_bytes) > max(check_bytes, 1):
                    raise ValueError("Public response exceeded the requested prefix")
            if len(prefix_bytes) != max(check_bytes, 1):
                raise ValueError("Short public prefix response")
            if existing and check_bytes:
                with existing.open("rb") as handle:
                    if handle.read(check_bytes) != prefix_bytes:
                        raise ValueError("Preserved partial does not match this publisher file")
            transport = {"mode": "public usercontent direct Range", "url": url,
                         "remote_bytes": int(content_range.split("/")[-1]), "Last-Modified": response.headers.get("Last-Modified"),
                         "ETag": response.headers.get("ETag"), "preserved_prefix_bytes": existing.stat().st_size if existing else 0,
                         "prefix_compared_bytes": check_bytes if existing else 0}
        options.pop("id")
        options["url"] = url
        print(json.dumps({"transport_preflight": transport}), flush=True)
    if "progress" in inspect.signature(gdown.download).parameters:
        options["progress"] = progress
    try:
        result = gdown.download(**options)
        if result is None or not target.exists():
            raise RuntimeError("Payload transfer did not complete")
    except Exception as exc:
        error = {"file_id": file["id"], "name": args.name, "error_type": type(exc).__name__,
                 "observed_total_bytes": state["observed_total_bytes"],
                 "partial_files": [{"name": f.name, "bytes": f.stat().st_size} for f in args.output_dir.glob(args.name + "*.part")],
                 "complete": False, "time_utc": datetime.now(timezone.utc).isoformat()}
        receipt.with_suffix(receipt.suffix + ".failure.json").write_text(json.dumps(error, indent=2), encoding="utf-8")
        raise
    with target.open("rb") as f:
        actual = hashlib.file_digest(f, "sha256").hexdigest()
    report = {"file_id": file["id"], "name": args.name, "publisher": inventory["publisher"],
              "folder_id": inventory["folder_id"], "inventory_sha256": hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
              "bytes": target.stat().st_size, "sha256": actual, "transfer_complete": True,
              "publisher_checksum_available": False, "content_schema_validated": False,
              "input_form": inventory.get("input_form", "author-linked processed features, not raw video"), "time_utc": datetime.now(timezone.utc).isoformat()}
    report["transport"] = transport
    if args.direct and target.stat().st_size != transport["remote_bytes"]:
        raise ValueError("Completed target size differs from public range preflight")
    receipt.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
