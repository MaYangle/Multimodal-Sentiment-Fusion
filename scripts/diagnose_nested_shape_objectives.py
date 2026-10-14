"""Post-study TRAIN diagnostic; never changes fitted objectives or promotion."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from jevaffect.nested_shape_training import load_stage, context, loader, selection
from jevaffect.shape_controls import ShapeHead, hit_probability
from jevaffect.native_training import digest, save_json
from jevaffect.anchor_calibration import bin_targets, event_metrics, TOLERANCE


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    suite = json.loads(args.suite.read_text())
    if not suite['suite_complete'] or len(suite['results']) != 90:
        raise ValueError('Complete registered study required before diagnostic')
    torch.set_num_threads(4)
    rows, targets_by_id = [], {}
    for seed in (13,42,2026):
        for fold in range(3):
            tables, _ = load_stage(args.points, seed, fold)
            outer = tables['outer']
            point, anchor, target = (outer[k] for k in ('native_point','anchor','target'))
            if seed == 13:
                targets_by_id.update(zip(outer['ids'], target.tolist()))
            for row in [r for r in suite['results'] if r['seed']==seed and r['fold']==fold]:
                root = Path(row['root'])
                cp = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
                identity = cp['identity']; cfg = identity['config']; stats = identity['normalization']
                model = ShapeHead(row['case'], context(tables['shape_fit']).shape[1], cfg['hidden'], cfg['dropout'])
                model.load_state_dict(cp['model'])
                fit = selection(model, loader(tables['shape_fit'], stats, cfg['batch_size']))
                selected = selection(model, loader(tables['shape_select'], stats, cfg['batch_size']))
                pred_path = root/'outer_predictions.json'
                if digest(pred_path) != row['predictions_sha256']:
                    raise ValueError('Outer predictions changed')
                records = json.loads(pred_path.read_text())
                native_hit_probabilities = []
                model.eval()
                with torch.no_grad():
                    for x, m, _, _ in loader(outer, stats, cfg['batch_size']):
                        native_hit_probabilities.append(hit_probability(model(x, m), m, TOLERANCE).numpy())
                native_hit_probabilities = np.concatenate(native_hit_probabilities)
                native_errors = np.abs(anchor-target)
                native_hit = native_errors <= np.float32(.1)
                native_event = event_metrics(native_hit_probabilities, native_hit, native_errors)
                clipped = point != anchor
                strata = {}
                for name, mask in [('all', np.ones(len(point), bool)), ('anchor_clipped', clipped), ('anchor_in_range', ~clipped)]:
                    strata[name] = dict(count=int(mask.sum()),
                        bin_NLL=float(np.mean([r['bin_NLL'] for r, keep in zip(records, mask) if keep])) if mask.any() else None,
                        raw_hit_Brier=float(np.mean([r['hit_Brier'] for r, keep in zip(records, mask) if keep])) if mask.any() else None,
                        raw_point_MAE=float(np.abs(point[mask]-target[mask]).mean()) if mask.any() else None,
                        clipped_point_MAE=float(np.abs(anchor[mask]-target[mask]).mean()) if mask.any() else None)
                rows.append(dict(case=row['case'],seed=seed,fold=fold,selected_epoch=cp['epoch'],
                                 shape_fit_NLL=fit,shape_select_NLL=selected,outer_NLL=row['outer']['bin_log_score'],
                                 fit_to_outer_gap=row['outer']['bin_log_score']-fit,strata=strata,
                                 supplemental_author_clipped_point_hit_event=native_event))
    y = np.asarray(list(targets_by_id.values()), np.float32)
    bins = bin_targets(y)
    within = []
    for index in range(5):
        values = y[bins==index]
        within.append(dict(bin=index,count=len(values),unique_native_scores=len(np.unique(values)),
                           minimum=float(values.min()),maximum=float(values.max()),
                           score_population_SD=float(values.std())))
    report = dict(suite_sha256=digest(args.suite),point_suite_sha256=digest(args.points),
                  scope='post-study author TRAIN diagnostic; no refitting or promotion-rule changes',
                  native_target_count=len(y),unique_native_scores=len(np.unique(y)),within_author_bins=within,
                  rows=rows,official_VALID_used=False,official_TEST_used=False,
                  native_metric_note='CH author MAE/Corr clip predictions; frozen raw point is unchanged but native_raw_point producer key is an alias of clipped author-protocol metrics',
                  supplemental_event_note='additional clipped-point hit diagnostic matches CH author scoring; primary registered event stays centered on raw point and promotion rule is unchanged',
                  interpretation='bin scores coarsen multiple native targets; bin likelihood alone does not identify within-bin event probabilities')
    save_json(args.output, report)
    print(json.dumps({'count':len(rows),'unique_native_scores':report['unique_native_scores'],'within_author_bins':within}))


if __name__ == '__main__':
    main()
