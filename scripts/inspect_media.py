"""Decode real media and persist content hashes, timestamps and stream status."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.media import decode_video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; select a new inspection directory.')
    decoded = decode_video(args.input)
    args.output.mkdir(parents=True)
    for key in ('audio', 'frames', 'frame_times'):
        np.save(args.output / (key + '.npy'), decoded[key], allow_pickle=False)
    (args.output / 'metadata.json').write_text(json.dumps(decoded['metadata'], indent=2), encoding='utf-8')
    print(json.dumps(decoded['metadata']), flush=True)


if __name__ == '__main__':
    main()
