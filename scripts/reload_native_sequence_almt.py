"""CPU replay of all three selected normalized-sequence ALMT checkpoints."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.native_training import digest,NativeSplit,author_rows,evaluate,save_json
from jevaffect.native_predict import load_model
from scripts.prepare_native_sequence_standardization import _close_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();suite=json.loads(args.suite.read_text())
    if not suite['suite_complete'] or len(suite['results'])!=3 or {r['seed'] for r in suite['results']}!={13,42,2026}:
        raise ValueError('Three completed native ALMT seeds required')
    torch.set_num_threads(4);evidence=[]
    for row in suite['results']:
        if row['model']!='almt' or row['smoke_only'] or row['test_evaluated']:raise ValueError('Formal ALMT only')
        root=Path('runs/native_simsv2_sequence_standardized_v1/almt')/f"seed_{row['seed']}"
        identity=json.loads((root/'identity.json').read_text());cfg=identity['config']
        if Path(cfg['run_root']).resolve()/'almt'/f"seed_{row['seed']}" != root.resolve():raise ValueError('Root identity mismatch')
        if digest(root/'best.pt')!=row['checkpoint_sha256'] or digest(root/'valid_predictions.json')!=row['predictions_sha256']:
            raise ValueError('Selected artifacts changed')
        metadata=json.loads((Path(cfg['cache_dir'])/'metadata.json').read_text())
        for rel,expected in metadata['artifact_sha256'].items():
            if digest(Path(cfg['cache_dir'])/rel)!=expected:raise ValueError('Cache artifact changed: '+rel)
        spec=load_model(root/'best.pt',cfg['cache_dir'],'cpu')
        if spec['shapes']!={m:tuple(s) for m,s in metadata['sequence_shapes'].items()}:raise ValueError('Sequence descriptor collapsed')
        parsed=author_rows(cfg['author_csv'],'simsv2')
        data=NativeSplit(cfg['cache_dir'],'valid',parsed,('text','audio','vision'),pooled=False)
        if data.hashes!=identity['features']['valid']:raise ValueError('Trained VALID arrays changed')
        metrics,actual=evaluate(spec['model'],torch.utils.data.DataLoader(data,batch_size=cfg['batch_size'],shuffle=False),'cpu','simsv2')
        stored=json.loads((root/'valid_predictions.json').read_text())
        if [(r['id'],r['target']) for r in actual]!=[(r['id'],r['target']) for r in stored]:raise ValueError('Prediction IDs/targets differ')
        error=max(abs(a['prediction']-b['prediction']) for a,b in zip(actual,stored))
        if error>1e-5:raise ValueError('CPU prediction replay exceeded1e-5')
        last=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        history=json.loads((root/'history.json').read_text())
        if last['identity']!=identity or last['history']!=history or not(last['stale']>=cfg['patience'] or len(history)>=cfg['max_epochs']):
            raise ValueError('Last checkpoint/history/budget mismatch')
        if min(history,key=lambda h:h['valid']['MAE'])['epoch']!=row['best_epoch']:raise ValueError('First-min selection mismatch')
        evidence.append(dict(seed=row['seed'],rows=len(actual),epochs=len(history),best_epoch=row['best_epoch'],
                             cpu_gpu_max_prediction_delta=error,CPU_metrics=metrics,cache_artifacts=len(metadata['artifact_sha256']),
                             sequence_shapes=spec['shapes'],checkpoint_sha256=digest(root/'best.pt')))
        _close_dataset(data)
    report=dict(complete=True,evidence=evidence,suite_sha256=digest(args.suite),code_sha256=digest(__file__),
                inference_code_sha256=digest('jevaffect/native_predict.py'),official_TEST_used=False,
                scope='selected native ALMT CPU model replay and input/budget checks; official VALID is selection set')
    save_json(args.output,report);print(json.dumps({'complete':True,'models':len(evidence),'max_delta':max(r['cpu_gpu_max_prediction_delta'] for r in evidence)}))


if __name__=='__main__':main()
