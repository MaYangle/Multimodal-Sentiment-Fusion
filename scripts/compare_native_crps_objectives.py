"""Matched objective-change diagnostic; never changes the registered gate."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from jevaffect.native_training import digest,save_json
from jevaffect.nested_shape_training import load_stage,context
from jevaffect.native_crps_training import predict,CASES
from jevaffect.shape_controls import ShapeHead


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--nll-suite',type=Path,default=Path('runs/native_nested_shapes_v1/suite.json'))
    p.add_argument('--crps-suite',type=Path,default=Path('runs/native_cdf_crps_v1/suite.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();suites=[json.loads(f.read_text()) for f in (args.nll_suite,args.crps_suite)]
    if not all(s['suite_complete'] for s in suites):raise ValueError('Two complete studies required')
    torch.set_num_threads(4)
    old={(r['case'],r['seed'],r['fold']):r for r in suites[0]['results'] if r['case'] in CASES}
    new={(r['case'],r['seed'],r['fold']):r for r in suites[1]['results']}
    expected={(c,s,f) for c in CASES for s in (13,42,2026) for f in range(3)}
    if set(old)!=expected or set(new)!=expected:raise ValueError('Registered comparable36-case matrix required')
    rows=[]
    for key in sorted(expected):
        first,second=old[key],new[key];path=Path(first['root'])/'best.pt'
        if digest(path)!=first['best_sha256']:raise ValueError('NLL checkpoint changed')
        cp=torch.load(path,map_location='cpu',weights_only=False);identity=cp['identity'];cfg=identity['config']
        tables,binding=load_stage(cfg['point_suite'],key[1],key[2])
        second_identity=json.loads((Path(second['root'])/'identity.json').read_text())
        if binding!=second_identity['binding'] or identity['normalization']!=second_identity['normalization']:
            raise ValueError('Input tables or optimization-only normalizers differ between objectives')
        model=ShapeHead(key[0],context(tables['shape_fit']).shape[1],cfg['hidden'],cfg['dropout']).eval()
        model.load_state_dict(cp['model'])
        metrics,records=predict(model,tables['outer'],identity['normalization'],cfg['batch_size'])
        newrecords=json.loads((Path(second['root'])/'outer_predictions.json').read_text())
        if [(r['id'],r['target'],r['native_point'],r['anchor']) for r in records]!=[(r['id'],r['target'],r['native_point'],r['anchor']) for r in newrecords]:
            raise ValueError('Original point/target/canonical rows changed between objectives')
        selected=('CRPS_48','native_hit_Brier','native_hit_log_score','raw_hit_Brier','bin_log_score')
        rows.append(dict(case=key[0],seed=key[1],fold=key[2],count=len(records),
                         NLL_selected_epoch=first['best_epoch'],CRPS_selected_epoch=second['best_epoch'],
                         NLL_fit_readout={k:metrics[k] for k in selected},CRPS_fit_readout={k:second['outer'][k] for k in selected}))
    merged=[]
    for case in CASES:
        for seed in (13,42,2026):
            selected_rows=[r for r in rows if (r['case'],r['seed'])==(case,seed)]
            count=sum(r['count'] for r in selected_rows)
            if count!=2722:raise ValueError('Incomplete per-seed outer coverage')
            metrics=selected_rows[0]['NLL_fit_readout']
            a={k:sum(r['count']*r['NLL_fit_readout'][k] for r in selected_rows)/count for k in metrics}
            b={k:sum(r['count']*r['CRPS_fit_readout'][k] for r in selected_rows)/count for k in metrics}
            merged.append(dict(case=case,seed=seed,count=count,NLL_fit=a,CRPS_fit=b,
                               delta_CRPS_fit_minus_NLL_fit={k:b[k]-a[k] for k in metrics}))
    report=dict(scope='post-study matched objective diagnostic; same point/context/stats, no refit or gate change',
                NLL_suite_sha256=digest(args.nll_suite),CRPS_suite_sha256=digest(args.crps_suite),
                diagnostic_code_sha256=digest(__file__),fits_read=36,rows=rows,merged=merged,
                official_VALID_used=False,official_TEST_used=False)
    save_json(args.output,report)
    print(json.dumps({'fits':len(rows),'merges':len(merged),'scope':report['scope']}))


if __name__=='__main__':main()
