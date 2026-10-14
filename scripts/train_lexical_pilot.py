"""Real-data CPU sanity baseline: train-only hashed character TF-IDF + softmax.

This supplemental lexical model is not the planned pretrained text baseline.
It never evaluates the test split. Use official CSVs and explicitly labeled pilot limits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.protocol import CLASSES, map_simsv2_label


def encode(texts, dimension):
    result = np.zeros((len(texts), dimension), dtype=np.float32)
    for row, text in enumerate(texts):
        text = ''.join(text.lower().split())
        for width in (1, 2, 3):
            for i in range(max(0, len(text) - width + 1)):
                token = text[i:i + width].encode('utf-8')
                index = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), 'little') % dimension
                result[row, index] += 1
    return result


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def scores(y, p):
    pred = p.argmax(axis=1)
    cm = np.zeros((5, 5), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * cm.diagonal(), denom, out=np.zeros(5), where=denom != 0)
    return {'accuracy': float(np.mean(pred == y)), 'macro_f1': float(f1.mean()),
            'nll': float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            'confusion_matrix': cm.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--dimension', type=int, default=4096)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--valid-limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != 'resume'}
    config['csv_sha256'] = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    config['model'] = 'supplemental_char_tfidf_softmax_pilot'
    cfg_path = args.output / 'config.json'
    if cfg_path.exists():
        if not args.resume:
            raise SystemExit('Run exists. Use a new output directory or --resume.')
        if json.loads(cfg_path.read_text()) != config:
            raise SystemExit('Resume configuration/data differs from the recorded run.')
    cfg_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    with args.csv.open(encoding='utf-8-sig', newline='') as f:
        # Test rows are never encoded, scored or used for fitting.
        rows = [r for r in csv.DictReader(f) if r['mode'] in ('train', 'valid')]
    train = [r for r in rows if r['mode'] == 'train']
    valid = [r for r in rows if r['mode'] == 'valid']
    rng = np.random.default_rng(args.seed)
    if args.train_limit:
        train = [train[i] for i in rng.permutation(len(train))[:args.train_limit]]
    if args.valid_limit:
        valid = [valid[i] for i in rng.permutation(len(valid))[:args.valid_limit]]
    if not train or not valid:
        raise ValueError('Official train and valid rows required')
    labels = lambda subset: np.array([CLASSES.index(map_simsv2_label(float(r['label']))) for r in subset])
    yt, yv = labels(train), labels(valid)
    x = encode([r['text'] for r in train], args.dimension)
    v = encode([r['text'] for r in valid], args.dimension)
    idf = np.log((1 + len(x)) / (1 + (x > 0).sum(0))).astype(np.float32) + 1
    for features in (x, v):
        np.log1p(features, out=features)
        features *= idf
        features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    x = np.column_stack((x, np.ones(len(x), dtype=np.float32)))
    v = np.column_stack((v, np.ones(len(v), dtype=np.float32)))
    w = np.zeros((args.dimension + 1, 5), dtype=np.float32)
    first = np.zeros_like(w)
    second = np.zeros_like(w)
    start, step, best, stale = 1, 0, -1.0, 0
    last = args.output / 'last.npz'
    if args.resume:
        state = np.load(last, allow_pickle=False)
        w, first, second = state['w'], state['first'], state['second']
        start, step, best, stale = int(state['epoch']) + 1, int(state['step']), float(state['best']), int(state['stale'])
        rng.bit_generator.state = json.loads(str(state['rng']))
    started = time.perf_counter()
    for epoch in range(start, args.epochs + 1):
        order = rng.permutation(len(x))
        for offset in range(0, len(x), 128):
            indices = order[offset:offset + 128]
            batch, target = x[indices], yt[indices]
            delta = softmax(batch @ w)
            delta[np.arange(len(target)), target] -= 1
            grad = batch.T @ delta / len(target)
            grad[:-1] += 1e-4 * w[:-1]
            step += 1
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad * grad
            w -= 0.03 * (first / (1 - 0.9 ** step)) / (np.sqrt(second / (1 - 0.999 ** step)) + 1e-8)
        metrics = scores(yv, softmax(v @ w))
        improved = metrics['accuracy'] > best
        if improved:
            best, stale = metrics['accuracy'], 0
            np.savez_compressed(args.output / 'best.npz', w=w, idf=idf, epoch=epoch)
        else:
            stale += 1
        entry = {'epoch': epoch, 'validation': metrics, 'elapsed_seconds': time.perf_counter() - started}
        with (args.output / 'history.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(entry) + '\n')
        np.savez_compressed(last, w=w, first=first, second=second, epoch=epoch, step=step,
                            best=best, stale=stale, rng=json.dumps(rng.bit_generator.state))
        print(json.dumps(entry), flush=True)
        if stale >= 8:
            break
    state = np.load(args.output / 'best.npz', allow_pickle=False)
    p = softmax(v @ state['w'])
    majority = int(np.bincount(yt, minlength=5).argmax())
    summary = {'model': config['model'], 'seed': args.seed, 'train_samples': len(train),
               'validation_samples': len(valid), 'best_epoch': int(state['epoch']), 'validation': scores(yv, p),
               'train_majority_class': CLASSES[majority], 'majority_validation_accuracy': float(np.mean(yv == majority)),
               'seconds_this_invocation': time.perf_counter() - started,
               'test_evaluated': False, 'pretrained_encoder': False,
               'note': 'Real labels and transcripts; supplemental lexical sanity run, not final multimodal evidence.'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with (args.output / 'validation_predictions.jsonl').open('w', encoding='utf-8') as f:
        for row, label, probabilities in zip(valid, yv, p):
            f.write(json.dumps({'id': row['video_id'] + '/' + row['clip_id'], 'target': int(label),
                                'probabilities': probabilities.tolist()}, ensure_ascii=False) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
