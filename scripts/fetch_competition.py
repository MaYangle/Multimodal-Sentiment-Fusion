"""Snapshot official Kaggle metadata using the installed skill, without logging secrets."""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skill-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--competition', default='msa-challenge-at-the-4th-pazhou-ai-competition')
    args = parser.parse_args()
    if not os.environ.get('KAGGLE_API_TOKEN'):
        token_file = Path.home() / '.kaggle' / 'access_token'
        if token_file.is_file():
            os.environ['KAGGLE_API_TOKEN'] = token_file.read_text().strip()
    if not os.environ.get('KAGGLE_API_TOKEN'):
        raise SystemExit('KAGGLE_API_TOKEN or ~/.kaggle/access_token required; no secret was printed.')
    sys.path.insert(0, str(args.skill_dir / 'scripts'))
    from runtime import competition_pages, kaggle_api
    args.output.mkdir(parents=True, exist_ok=True)
    pages = competition_pages(args.competition)
    (args.output / 'pages.json').write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding='utf-8')
    for name, content in pages.items():
        safe_name = ''.join(c for c in name if c.isalnum() or c in '-_')
        (args.output / (safe_name + '.md')).write_text(content, encoding='utf-8')
    api = kaggle_api()
    files = {'files': [], 'complete': False}
    token = None
    seen_tokens = set()
    while True:
        response = api.competition_list_files(args.competition, page_token=token, page_size=1000)
        page = response.to_dict()
        files['files'].extend(page.get('files', []))
        token = page.get('nextPageToken')
        print('Listed', len(files['files']), 'files', flush=True)
        if not token:
            files['complete'] = True
            break
        if token in seen_tokens:
            raise RuntimeError('Repeated file-list page token; refusing an incomplete manifest')
        seen_tokens.add(token)
    (args.output / 'files.json').write_text(json.dumps(files, indent=2, default=str), encoding='utf-8')
    manifest = {'retrieved_at': datetime.now(timezone.utc).isoformat(), 'competition': args.competition,
                'source': 'https://www.kaggle.com/competitions/' + args.competition, 'files': {}}
    for p in args.output.iterdir():
        if p.is_file() and p.name != 'snapshot.json':
            manifest['files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    (args.output / 'snapshot.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'pages': list(pages), 'output': str(args.output),
                      'file_count': len(files['files']), 'complete': files['complete'],
                      'total_bytes': sum(f.get('totalBytes', 0) for f in files['files'])}, default=str))


if __name__ == '__main__':
    main()
