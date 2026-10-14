"""Cache verified multilingual BERT token features from real SIMSv2 transcripts.

Only official train/valid rows are encoded. This is the provided-transcript
condition; it is not an ASR or multimodal experiment. Batch-boundary resume
never marks partially written features ready for a training run.
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jevaffect.protocol import CLASSES, map_simsv2_label


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--meta-csv', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version
    if not 4 <= args.max_length <= 512 or args.batch_size < 1:
        raise ValueError('Invalid encoding length/batch size')
    provenance = json.loads((args.model_dir / 'provenance.json').read_text(encoding='utf-8'))
    if not provenance.get('verified') or provenance.get('repo') != 'google-bert/bert-base-multilingual-cased':
        raise ValueError('A verified pinned mBERT download is required')
    actual_weight_hash = sha256(args.model_dir / 'model.safetensors')
    if actual_weight_hash != provenance['sha256']:
        raise ValueError('Encoder weight hash mismatch')
    identity = {'source_csv_sha256': sha256(args.meta_csv), 'encoder_sha256': actual_weight_hash,
                'encoder_revision': provenance['revision'], 'max_length': args.max_length,
                'batch_size': args.batch_size, 'device': args.device, 'torch': torch.__version__,
                'dtype': 'float32', 'condition': 'provided_transcripts', 'transformers_version': transformers_version, 'numpy_version': np.__version__,
                'implementation_sha256': sha256(__file__)}
    progress_path = args.output / 'progress.json'
    if args.output.exists():
        if not args.resume or not progress_path.is_file():
            raise ValueError('Output exists; resume requires its matching progress.json')
        progress = json.loads(progress_path.read_text(encoding='utf-8'))
        if progress['identity'] != identity:
            raise ValueError('Resume encoder/data/config identity mismatch')
        if progress.get('complete'):
            print('Cache already complete and identity matches', flush=True)
            return
    else:
        args.output.mkdir(parents=True)
        progress = {'identity': identity, 'completed_rows': {}, 'truncated_rows': {}, 'complete': False}
        write_json(progress_path, progress)
    with args.meta_csv.open(encoding='utf-8-sig', newline='') as f:
        records = [r for r in csv.DictReader(f) if r['mode'] in ('train', 'valid')]
    all_ids = [r['video_id'] + '/' + r['clip_id'] for r in records]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('Duplicate train/valid IDs')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(args.model_dir, local_files_only=True).to(args.device).eval()
    hidden = int(model.config.hidden_size)
    source_hash = {'csv': identity['source_csv_sha256'], 'encoder': actual_weight_hash}
    meta = {'dataset': 'CH-SIMS-v2.0', 'condition': 'provided_transcripts',
            'modalities': ['text'], 'text_features_ready': False, 'complete': False,
            'feature_dims': {'text': hidden}, 'sequence_lengths': {'text': args.max_length},
            'source_hashes': source_hash, 'source_sha256': hashlib.sha256(json.dumps(source_hash,sort_keys=True).encode()).hexdigest(), 'encoder': provenance,
            'class_order': list(CLASSES), 'test_encoded': False}
    write_json(args.output / 'meta.json', meta)
    started = time.perf_counter()
    for split in ('train', 'valid'):
        rows = [r for r in records if r['mode'] == split]
        if not rows:
            raise ValueError('Missing official ' + split + ' split')
        ids = [r['video_id'] + '/' + r['clip_id'] for r in rows]
        targets = np.array([CLASSES.index(map_simsv2_label(float(r['label']))) for r in rows], dtype=np.int64)
        directory = args.output / split
        directory.mkdir(exist_ok=True)
        write_json(directory / 'ids.json', ids)
        np.save(directory / 'labels.npy', targets, allow_pickle=False)
        offset = int(progress['completed_rows'].get(split, 0))
        mode = 'r+' if offset else 'w+'
        features = np.lib.format.open_memmap(directory / 'text.npy', mode=mode, dtype=np.float32,
                                             shape=(len(rows), args.max_length, hidden))
        lengths = np.lib.format.open_memmap(directory / 'text_lengths.npy', mode=mode, dtype=np.int64, shape=(len(rows),))
        for begin in range(offset, len(rows), args.batch_size):
            texts = [r['text'] for r in rows[begin:begin + args.batch_size]]
            full_lengths = [len(x) for x in tokenizer(texts, add_special_tokens=True, truncation=False)['input_ids']]
            batch = tokenizer(texts, padding='max_length', truncation=True,
                              max_length=args.max_length, return_tensors='pt')
            length = batch['attention_mask'].sum(1).numpy()
            with torch.inference_mode():
                values = model(**{k: v.to(args.device) for k, v in batch.items()}).last_hidden_state
                values = values * batch['attention_mask'].to(args.device).unsqueeze(-1)
                values = values.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite encoder features')
            end = begin + len(texts)
            features[begin:end] = values
            lengths[begin:end] = length
            features.flush(); lengths.flush()
            progress['completed_rows'][split] = end
            progress['truncated_rows'][split] = progress['truncated_rows'].get(split, 0) + sum(n > args.max_length for n in full_lengths)
            write_json(progress_path, progress)
            print(json.dumps({'split': split, 'encoded': end, 'total': len(rows)}), flush=True)
        del features, lengths
    if args.device == 'cuda':
        torch.cuda.synchronize()
    meta.update(text_features_ready=True, complete=True,
                encode_seconds_this_invocation=time.perf_counter() - started,
                truncated_rows=progress['truncated_rows'],
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None)
    meta['artifact_sha256'] = {str(p.relative_to(args.output)): sha256(p)
                              for split in ('train', 'valid') for p in (args.output / split).iterdir() if p.is_file()}
    write_json(args.output / 'meta.json', meta)
    progress['complete'] = True
    write_json(progress_path, progress)
    print(json.dumps({'complete': True, 'counts': progress['completed_rows'], 'test_encoded': False}), flush=True)


if __name__ == '__main__':
    main()
