"""Exact head-study reproduction comparison; point tables are reused."""
import argparse
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from jevaffect.native_training import digest,save_json

CASES=('scale','mixture','median_mdn','bern_d6')


def identity_without_output_location(identity):
    result=copy.deepcopy(identity)
    result.pop('config_sha256',None)
    result['config'].pop('config_sha256',None)
    result['config'].pop('run_root',None)
    return result


def exact_state(a,b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(exact_state(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)) or isinstance(b,(tuple,list)):
        return type(a)==type(b) and len(a)==len(b) and all(exact_state(x,y) for x,y in zip(a,b))
    return a==b


def read(path):return json.loads(Path(path).read_text())


def compare(original,reproduction):
    expected={(c,s,f) for c in CASES for s in (13,42,2026) for f in range(3)}
    suites=[read(p) for p in (original,reproduction)]
    maps=[]
    for suite in suites:
        if (not suite['suite_complete'] or len(suite['results'])!=36
                or suite['suite_identity']['smoke_only'] or suite['official_VALID_used'] or suite['official_TEST_used']):
            raise ValueError('Two full formal TRAIN-only 36-fit studies required')
        mapped={(r['case'],r['seed'],r['fold']):r for r in suite['results']}
        if set(mapped)!=expected:raise ValueError('Reproduction model matrix differs')
        maps.append(mapped)
    evidence=[]
    for key in sorted(expected):
        rows=[m[key] for m in maps];roots=[Path(r['root']) for r in rows]
        identities=[read(p/'identity.json') for p in roots]
        if identity_without_output_location(identities[0])!=identity_without_output_location(identities[1]):
            raise ValueError(f'Identity differs beyond run_root/config file hash: {key}')
        fields=('complete','smoke_only','best_epoch','best_selection_CRPS_48','epochs','updates','parameters',
                'initial_encoder_sha256','outer','quadrature_gate_pass','implementation_sha256','plan_sha256','point_binding')
        if any(rows[0][k]!=rows[1][k] for k in fields):
            raise ValueError(f'Readout/selection/budget changed: {key}')
        for r,p in zip(rows,roots):
            for name,hash_key in [('best.pt','best_sha256'),('last.pt','last_sha256'),('history.json','history_sha256'),('outer_predictions.json','predictions_sha256')]:
                if digest(p/name)!=r[hash_key]:raise ValueError('Changed reproduction artifact: '+str(p/name))
        forecasts=[read(p/'outer_predictions.json') for p in roots]
        if forecasts[0]!=forecasts[1]:raise ValueError('Per-row CDF/event/quantile/score forecasts differ')
        histories=[[{k:v for k,v in row.items() if k!='seconds'} for row in read(p/'history.json')] for p in roots]
        if histories[0]!=histories[1]:raise ValueError('Full epoch objectives/budget differ')
        for name in ('best.pt','last.pt'):
            states=[torch.load(p/name,map_location='cpu',weights_only=False) for p in roots]
            if not exact_state(states[0]['model'],states[1]['model']):raise ValueError('Model tensors differ')
            if name=='last.pt' and not exact_state(states[0]['optimizer'],states[1]['optimizer']):raise ValueError('Optimizer tensors/state differ')
        evidence.append(dict(case=key[0],seed=key[1],fold=key[2],prediction_rows=len(forecasts[0]),
                             tensors_optimizer_predictions_metrics_history_selection_budget_exact=True))
    return dict(passed=True,verified_fits=len(evidence),verified_prediction_rows=sum(r['prediction_rows'] for r in evidence),
                original_sha256=digest(original),reproduction_sha256=digest(reproduction),evidence=evidence,
                scope='independent source-copy full probability-head retraining; audited point tables reused, no point/encoder/media re-extraction or final-method claim',
                official_VALID_used=False,official_TEST_used=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True);p.add_argument('--reproduction',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();report=compare(args.original,args.reproduction);save_json(args.output,report)
    print(json.dumps({k:report[k] for k in ('passed','verified_fits','verified_prediction_rows','scope')}))
