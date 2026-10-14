"""Fit registered native probability shapes on source-nested frozen tables."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.nested_shape_training import load_stage, train
from jevaffect.native_training import save_json, digest
from jevaffect.shape_controls import CASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--seed', type=int, choices=(13,42,2026))
    parser.add_argument('--fold', type=int, choices=(0,1,2))
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after <= 0:
        parser.error('--stop-after must be positive')
    config = json.loads(args.config.read_text())
    if tuple(config['cases']) != CASES or config['seeds'] != [13,42,2026]:
        raise ValueError('Registered case/seed matrix changed')
    config['config_sha256'] = digest(args.config)
    config['plan_sha256'] = digest(config['plan'])
    results = []
    for seed in [args.seed] if args.seed is not None else config['seeds']:
        for fold in [args.fold] if args.fold is not None else range(3):
            tables, binding = load_stage(config['point_suite'], seed, fold, args.allow_smoke)
            for case in [args.case] if args.case else config['cases']:
                results.append(train(config, tables, binding, case=case, seed=seed, fold=fold,
                                     resume=args.resume, stop_after=args.stop_after))
                complete = len(results) == 90 and all(r['complete'] and not r['smoke_only'] for r in results)
                save_json(args.output, dict(results=results, suite_complete=complete,
                                            official_VALID_used=False, official_TEST_used=False))


if __name__ == '__main__':
    main()
