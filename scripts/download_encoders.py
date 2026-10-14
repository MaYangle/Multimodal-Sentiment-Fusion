"""Download revision-pinned public encoders, verify weights, preserve provenance.

This does not authorize redistribution of third-party artifacts. Model cards
and source-specific license decisions must accompany any later release.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone


MODELS = {
    'dinov2-small': {'repo': 'facebook/dinov2-small',
 'revision': 'ed25f3a31f01632728cabb09d1542f84ab7b0056',
 'files': ['README.md',
           'config.json',
           'preprocessor_config.json',
           'model.safetensors'],
 'weight': 'model.safetensors',
 'sha256': 'ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1',
 'model_card_license': 'apache-2.0'},
    'whisper-base': {'repo': 'openai/whisper-base',
 'revision': 'e37978b90ca9030d5170a5c07aadb050351a65bb',
 'files': ['README.md',
           'added_tokens.json',
           'config.json',
           'generation_config.json',
           'merges.txt',
           'normalizer.json',
           'preprocessor_config.json',
           'special_tokens_map.json',
           'tokenizer.json',
           'tokenizer_config.json',
           'vocab.json',
           'model.safetensors'],
 'weight': 'model.safetensors',
 'sha256': '07cadb9f25677c8d50df603e66a98fbd842cce45047139baeb16e6219a1e807b',
 'model_card_license': 'apache-2.0'},
    'gpt2-chinese-cluecorpussmall': {
        'repo': 'uer/gpt2-chinese-cluecorpussmall',
        'revision': 'c2c0249d8a2731f269414cc3b22dff021f8e07a3',
        'files': ['README.md', 'config.json', 'tokenizer_config.json', 'vocab.txt', 'pytorch_model.bin'],
        'weight': 'pytorch_model.bin',
        'sha256': '5478b950f53650c82f7f7e91010787c870813a76adeabe23c8571b01af1379b4',
        'model_card_license': None,
    },
    'mbert': {
        'repo': 'google-bert/bert-base-multilingual-cased',
        'revision': '3f076fdb1ab68d5b2880cb87a0886f315b8146f8',
        'files': ['README.md', 'config.json', 'tokenizer_config.json', 'tokenizer.json', 'vocab.txt', 'model.safetensors'],
        'weight': 'model.safetensors',
        'sha256': '876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc',
        'model_card_license': 'apache-2.0',
    },
    'wavlm': {
        'repo': 'microsoft/wavlm-base-plus',
        'revision': '4c66d4806a428f2e922ccfa1a962776e232d487b',
        'files': ['README.md', 'config.json', 'preprocessor_config.json', 'pytorch_model.bin'],
        'weight': 'pytorch_model.bin',
        'sha256': '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51',
        'model_card_license': None,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODELS, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault('HF_HOME', str(args.output_root / '.hf-cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    from huggingface_hub import hf_hub_download
    definition = MODELS[args.model]
    destination = args.output_root / args.model
    destination.mkdir(parents=True, exist_ok=True)
    record = dict(definition, started_utc=datetime.now(timezone.utc).isoformat(), verified=False)
    (destination / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    for filename in definition['files']:
        if filename == definition['weight'] and (destination / filename).is_file():
            with (destination / filename).open('rb') as existing:
                if hashlib.file_digest(existing, 'sha256').hexdigest() == definition['sha256']:
                    print(f'Already verified {args.model}/{filename}', flush=True)
                    continue
        print(f'Downloading {args.model}/{filename}', flush=True)
        hf_hub_download(repo_id=definition['repo'], filename=filename,
                        revision=definition['revision'], local_dir=destination, token=False)
    with (destination / definition['weight']).open('rb') as handle:
        actual = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual != definition['sha256']:
        raise RuntimeError(f'{args.model} weight SHA256 mismatch: {actual}')
    record['files_sha256'] = {}
    for name in definition['files']:
        with (destination / name).open('rb') as handle:
            record['files_sha256'][name] = hashlib.file_digest(handle, 'sha256').hexdigest()
    record.update(verified=True, completed_utc=datetime.now(timezone.utc).isoformat())
    (destination / 'provenance.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps({'model': args.model, 'verified': True, 'path': str(destination)}), flush=True)


if __name__ == '__main__':
    main()
