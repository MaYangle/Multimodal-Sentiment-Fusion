"""Pre-fit numerical assay on completed NLL-control checkpoints; no new fit."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.cdf_crps import per_sample
from jevaffect.shape_controls import ShapeHead
from jevaffect.nested_shape_training import load_stage,context,loader
from jevaffect.native_training import digest,save_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,default=Path('runs/native_nested_shapes_v1/suite.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    suite=json.loads(args.suite.read_text())
    if not suite['suite_complete']:raise ValueError('Completed NLL study required')
    torch.set_num_threads(4)
    rows=[]
    for row in suite['results']:
        if row['case'] not in ('scale','mixture','median_mdn','bern_d6'):continue
        root=Path(row['root']);cp_path=root/'best.pt'
        if digest(cp_path)!=row['best_sha256']:raise ValueError('NLL checkpoint changed')
        cp=torch.load(cp_path,map_location='cpu',weights_only=False)
        identity=cp['identity'];cfg=identity['config']
        tables,_=load_stage(cfg['point_suite'],row['seed'],row['fold'])
        data=tables['outer'];stats=identity['normalization']
        model=ShapeHead(row['case'],context(data).shape[1],cfg['hidden'],cfg['dropout']).eval()
        model.load_state_dict(cp['model'])
        values={n:[] for n in (48,96,192)};scale_min=float('inf')
        with torch.no_grad():
            for x,m,_,y in loader(data,stats,cfg['batch_size']):
                out=model(x,m)
                if 'scale' in out['parameters']:scale_min=min(scale_min,float(out['parameters']['scale'].min()))
                for n in values:values[n].append(per_sample(out,y,n).numpy())
        values={n:np.concatenate(v) for n,v in values.items()}
        rows.append(dict(case=row['case'],seed=row['seed'],fold=row['fold'],count=len(data['ids']),
                         checkpoint_sha256=digest(cp_path),mean={str(n):float(v.mean()) for n,v in values.items()},
                         mean_absolute_difference_48_192=float(np.abs(values[48]-values[192]).mean()),
                         maximum_absolute_difference_48_192=float(np.abs(values[48]-values[192]).max()),
                         mean_absolute_difference_96_192=float(np.abs(values[96]-values[192]).mean()),
                         minimum_logistic_scale=None if not np.isfinite(scale_min) else scale_min))
    if len(rows)!=36:raise ValueError('Four cases xthree seeds xthree folds required')
    report=dict(scope='pre-CRPS-fit numerical assay of already-completed NLL controls; no refit/tuning',
                suite_sha256=digest(args.suite),quadrature_code_sha256=digest('jevaffect/cdf_crps.py'),
                rows=rows,complete=True,numerical_gate_passed=all(r['mean_absolute_difference_48_192']<=1e-4 for r in rows),
                max_mean_absolute_difference_48_192=max(r['mean_absolute_difference_48_192'] for r in rows),
                official_VALID_used=False,official_TEST_used=False)
    save_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))


if __name__=='__main__':main()
