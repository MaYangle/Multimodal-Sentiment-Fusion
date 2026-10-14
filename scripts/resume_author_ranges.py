"""Bounded transport retries that reuse verified ranges and stop on no progress."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--size", type=int, required=True)
    p.add_argument("--prefix", type=Path, required=True)
    p.add_argument("--attempts", type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.attempts <= 20:
        raise ValueError("A bounded retry count between1 and20 is required")
    parts = a.output_dir / (a.name + ".range_parts")
    parts.mkdir(parents=True, exist_ok=True)
    state_path = parts / "supervisor.json"

    def received():
        # Complete blocks are checksum-validated by the child on every resume;
        # partial bytes only measure actual transfer progress, not completeness.
        return a.prefix.stat().st_size + sum(f.stat().st_size for pattern in ("*.block", "*.partial") for f in parts.glob(pattern))

    command = [sys.executable, str(Path(__file__).with_name("download_author_ranges.py")), "--inventory", str(a.inventory),
               "--name", a.name, "--size", str(a.size), "--prefix", str(a.prefix), "--output-dir", str(a.output_dir),
               "--workers", "2", "--chunk-mib", "8"]
    state = {"attempts": [], "complete": False, "running": True, "time_utc": datetime.now(timezone.utc).isoformat(),
             "note": "Received-byte counters include incomplete blocks; no schema/completeness claim."}
    state_path.write_text(json.dumps(state, indent=2))
    stagnant = 0
    for attempt in range(a.attempts):
        before = received()
        stderr = parts / f"resume_attempt_{attempt+1}_{int(time.time())}.stderr.log"
        with stderr.open("w", encoding="utf-8") as errors:
            result = subprocess.run(command, stderr=errors)
        after = received()
        row = {"attempt": attempt + 1, "exit_code": result.returncode, "bytes_before": before,
               "bytes_after": after, "bytes_added": after - before, "stderr": str(stderr)}
        state["attempts"].append(row)
        print(json.dumps(row), flush=True)
        if result.returncode == 0:
            state.update(complete=True, running=False)
            state_path.write_text(json.dumps(state, indent=2))
            return
        text = stderr.read_text(encoding="utf-8")
        # Integrity/version/wrong-range failures need diagnosis, not retries.
        retryable = any(name in text for name in ("ConnectionError", "ReadTimeout", "ConnectTimeout", "HTTPError"))
        if not retryable or "ValueError" in text:
            state.update(running=False, stopped="non-transport failure; inspect publisher response or cache")
            state_path.write_text(json.dumps(state, indent=2))
            return
        stagnant = stagnant + 1 if after <= before else 0
        if stagnant >= 3:
            state.update(running=False, stopped="three attempts without added bytes")
            state_path.write_text(json.dumps(state, indent=2))
            return
        state_path.write_text(json.dumps(state, indent=2))
        if attempt + 1 < a.attempts:
            time.sleep(min(15 * (attempt + 1), 60))
    state.update(running=False, stopped="bounded retry budget exhausted; cached bytes retained")
    state_path.write_text(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
