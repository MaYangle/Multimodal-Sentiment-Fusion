"""Verified parallel HTTP-range downloader for an explicitly pinned public file."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--size', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', type=Path)
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    parts = args.output.parent / (args.output.name + '.parts')
    parts.mkdir(exist_ok=True)
    prefix_size = args.prefix.stat().st_size if args.prefix else 0
    if not 0 <= prefix_size <= args.size:
        raise ValueError('Invalid preserved prefix length')
    meta = {'url': args.url, 'sha256': args.sha256, 'size': args.size, 'prefix_size': prefix_size}
    meta_path = parts / 'source.json'
    if meta_path.exists() and json.loads(meta_path.read_text()) != meta:
        raise ValueError('Resume source differs')
    meta_path.write_text(json.dumps(meta, indent=2))
    if args.output.exists():
        with args.output.open('rb') as f:
            valid = args.output.stat().st_size == args.size and hashlib.file_digest(f, 'sha256').hexdigest() == args.sha256
        if valid:
            print('Already verified', args.output, flush=True)
            return
        raise ValueError('Existing final output fails integrity; refusing overwrite')
    ranges = [(i, min(i + 16 * 1024 * 1024, args.size) - 1) for i in range(prefix_size, args.size, 16 * 1024 * 1024)]

    def fetch(interval):
        start, end = interval
        target = parts / f'{start}-{end}.block'
        if target.exists() and target.stat().st_size == end - start + 1:
            return target.stat().st_size
        partial = target.with_suffix('.partial')
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > end - start + 1:
                raise ValueError('Oversized partial range')
            if offset == end - start + 1:
                partial.replace(target)
                return offset
            requested = start + offset
            try:
                with requests.get(args.url, headers={'Range': f'bytes={requested}-{end}', 'Accept-Encoding': 'identity'}, stream=True, timeout=(30, 60)) as r:
                    expected = f'bytes {requested}-{end}/{args.size}'
                    if r.status_code != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError(f'Wrong range response: {r.status_code} {r.headers.get("Content-Range")} expected {expected}')
                    written = offset
                    with partial.open('ab') as f:
                        for block in r.iter_content(1024 * 1024):
                            if written + len(block) > end - start + 1:
                                raise ValueError('Response exceeds requested range')
                            f.write(block)
                            written += len(block)
                if partial.stat().st_size != end - start + 1:
                    raise IOError('Short HTTP range')
                partial.replace(target)
                return written
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError('Range retries exhausted')

    received = prefix_size
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for size in pool.map(fetch, ranges):
            received += size
            print(json.dumps({'completed_range_bytes': received, 'total_bytes': args.size, 'seconds': round(time.monotonic() - started, 1)}), flush=True)
    combined = args.output.with_suffix(args.output.suffix + '.assembling')
    digest = hashlib.sha256()
    sources = ([args.prefix] if args.prefix else []) + [parts / f'{start}-{end}.block' for start, end in ranges]
    with combined.open('wb') as dst:
        for source in sources:
            with source.open('rb') as src:
                while block := src.read(8 * 1024 * 1024):
                    dst.write(block)
                    digest.update(block)
    if combined.stat().st_size != args.size or digest.hexdigest() != args.sha256:
        raise RuntimeError('Full-file SHA256 mismatch; preserved pieces retained for investigation')
    combined.replace(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(args.sha256 + '  ' + args.output.name + '\n')
    print(json.dumps({'verified': True, 'output': str(args.output), 'seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    main()
