"""Source-nested native shape fits; outer data never select a checkpoint."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from jevaffect import shape_controls as shapes
from jevaffect.anchor_calibration import BOUNDARIES, TOLERANCE, bin_targets, normalizer, normalize, diagnostics
from jevaffect.native_training import digest, save_json, save_checkpoint
from jevaffect.nested_shape_points import validate_nested_manifest
from jevaffect.native_metrics import native_metrics


def load_stage(suite_path, seed, fold, allow_smoke=False):
    suite_path = Path(suite_path)
    suite = json.loads(suite_path.read_text())
    if suite['smoke_only'] and not allow_smoke or not suite['smoke_only'] and not suite['suite_complete']:
        raise ValueError('Complete formal point suite or explicit smoke suite required')
    manifest_path = Path(suite['manifest_path'])
    if digest(manifest_path) != suite['manifest_sha256']:
        raise ValueError('Nested point manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['official_VALID_used'] or manifest['official_TEST_used']:
        raise ValueError('Point stage used official held data')
    validate_nested_manifest(manifest['canonical_author_train_ids'], manifest['nested_manifest'])
    rows = [r for r in suite['results'] if (r['seed'], r['fold']) == (seed, fold)]
    if len(rows) != 1 or not rows[0]['complete']:
        raise ValueError('Selected point stage incomplete or duplicated')
    stage = rows[0]
    for name, expected in stage['implementation_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Point-stage implementation changed: '+name)
    for basename, key in [('best.pt', 'point_best_checkpoint_sha256'),
                          ('last.pt', 'point_last_checkpoint_sha256'),
                          ('history.json', 'point_history_sha256'),
                          ('identity.json', 'point_identity_sha256')]:
        if digest(Path(stage['point_model_root'])/basename) != stage[key]:
            raise ValueError('Point-stage artifact changed: '+basename)
    tables = {}
    roles = manifest['nested_manifest']['folds'][fold]
    for role, entry in stage['tables'].items():
        if digest(entry['path']) != entry['sha256'] or digest(entry['ids_path']) != entry['ids_sha256']:
            raise ValueError('Nested probability table changed')
        with np.load(entry['path'], allow_pickle=False) as archive:
            data = {k: np.array(archive[k], copy=True) for k in ('context', 'anchor', 'native_point', 'target')}
            ids = archive['ids'].tolist()
        if ids != json.loads(Path(entry['ids_path']).read_text()) or ids != roles[role+'_ids']:
            raise ValueError('Nested probability IDs/order changed')
        n = len(ids)
        if (data['context'].shape != (n, manifest['config']['hidden']) or any(data[k].shape != (n,) for k in ('anchor', 'native_point', 'target'))
                or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in data.values())
                or np.any(np.abs(data['target']) > 1) or not np.array_equal(data['anchor'], np.clip(data['native_point'], -1, 1))):
            raise ValueError('Invalid frozen point tables')
        data['ids'] = ids
        tables[role] = data
    if set(tables) != {'shape_fit', 'shape_select', 'outer'}:
        raise ValueError('Exactly three source-held tables required')
    binding = dict(suite_sha256=digest(suite_path), manifest_sha256=suite['manifest_sha256'],
                   point_best_checkpoint_sha256=stage['point_best_checkpoint_sha256'],
                   tables={k: v['sha256'] for k, v in stage['tables'].items()},
                   ids={k: v['ids_sha256'] for k, v in stage['tables'].items()},
                   source_csv_sha256=manifest['author_csv_sha256'],
                   nested_manifest_sha256=manifest['nested_manifest_sha256'],
                   smoke_only=suite['smoke_only'])
    return tables, binding


def context(data):
    return np.concatenate((data['context'], data['anchor'][:, None]), -1)


def loader(data, stats, batch_size, *, generator=None):
    ds = TensorDataset(torch.from_numpy(normalize(context(data), stats)),
                       torch.from_numpy(data['anchor']), torch.from_numpy(data['native_point']),
                       torch.from_numpy(data['target']))
    return DataLoader(ds, batch_size=batch_size, shuffle=generator is not None,
                      generator=generator, num_workers=0)


def loss(output, target):
    p = shapes.probabilities(output, BOUNDARIES.astype(np.float64))
    labels = torch.from_numpy(bin_targets(target.detach().cpu().numpy())).to(target.device)
    return -p[torch.arange(len(target), device=target.device), labels].clamp_min(1e-12).log().mean()


@torch.no_grad()
def selection(model, batches):
    model.eval()
    total, count = 0., 0
    for x, m, _, y in batches:
        total += float(loss(model(x, m), y))*len(y)
        count += len(y)
    return total/count


@torch.no_grad()
def predict(model, data, stats, batch_size):
    model.eval()
    probabilities, hits, intervals, medians, crps = [], [], [], [], []
    levels = [.25, .75, .1, .9, .05, .95, .025, .975, .5]
    grid = torch.linspace(-1., 1., 1001, dtype=torch.float64)
    for x, m, point, y in loader(data, stats, batch_size):
        out = model(x, m)
        probabilities.append(shapes.probabilities(out, BOUNDARIES.astype(np.float64)).numpy())
        hits.append(shapes.hit_probability(out, point, TOLERANCE).numpy())
        q = shapes.quantiles(out, levels).numpy()
        intervals.append(q[:, :8]); medians.append(q[:, 8])
        c = shapes.cdf(out, grid[None].expand(len(y), -1))
        truth = (grid[None] >= y.double()[:, None]).double()
        crps.append(torch.trapezoid((c-truth).square(), grid, dim=1).numpy())
    p, h, q, median, cr = [np.concatenate(v) for v in (probabilities, hits, intervals, medians, crps)]
    point, target = data['native_point'], data['target']
    view = dict(bins=p, p_hit=h, intervals={level: (q[:, 2*j], q[:, 2*j+1]) for j, level in enumerate((.5, .8, .9, .95))})
    metrics = diagnostics(point, target, view)
    metrics['CRPS_trapezoid_1001'] = float(cr.mean())
    metrics['anchor_clipped_count'] = int(np.count_nonzero(point != data['anchor']))
    metrics['native_raw_point'] = native_metrics(point, target, 'simsv2')
    if model.case != 'mdn' and not np.array_equal(median, data['anchor'].astype(np.float64)):
        raise ValueError('Fixed probabilistic median moved')
    labels = bin_targets(target)
    hit_truth = (np.abs(point-target) <= np.float32(.1)).astype(np.float64)
    nll = -np.log(np.maximum(p[np.arange(len(p)), labels], 1e-12))
    hp = np.clip(h, 1e-12, 1-1e-12)
    records = []
    for i, sid in enumerate(data['ids']):
        interval_values, scores = {}, {}
        for j, nominal in enumerate((.5, .8, .9, .95)):
            lo, hi = q[i, 2*j:2*j+2]; alpha = 1-nominal
            interval_values[str(nominal)] = [float(lo), float(hi)]
            scores[str(nominal)] = float(hi-lo+2/alpha*(max(lo-float(target[i]), 0)+max(float(target[i])-hi, 0)))
        records.append(dict(id=sid, target=float(target[i]), native_point=float(point[i]), anchor=float(data['anchor'][i]),
                            median=float(median[i]), bin_probabilities=p[i].tolist(), p_hit=float(h[i]),
                            intervals=interval_values, interval_scores=scores, bin_NLL=float(nll[i]),
                            bin_Brier=float(np.sum((p[i]-np.eye(5)[labels[i]])**2)), hit_Brier=float((h[i]-hit_truth[i])**2),
                            hit_log_score=float(-hit_truth[i]*np.log(hp[i])-(1-hit_truth[i])*np.log1p(-hp[i])),
                            CRPS_trapezoid_1001=float(cr[i])))
    metrics['interval_scores'] = {str(level): float(np.mean([r['interval_scores'][str(level)] for r in records])) for level in (.5, .8, .9, .95)}
    return metrics, records


def train(config, tables, binding, *, case, seed, fold, resume=False, stop_after=None):
    if case not in shapes.CASES or seed not in (13, 42, 2026) or fold not in range(3) or config['device'] != 'cpu':
        raise ValueError('Registered case, seed, fold and CPU shape runtime required')
    torch.set_num_threads(config['torch_threads']); torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    fit, select = tables['shape_fit'], tables['shape_select']
    stats = normalizer(context(fit), np.arange(len(fit['ids'])))
    identity = dict(config=config, case=case, seed=seed, fold=fold, binding=binding,
                    normalization=stats, fit_ids=fit['ids'], select_ids=select['ids'],
                    implementation={n: digest(Path(__file__).with_name(n)) for n in (
                        'nested_shape_training.py', 'shape_controls.py', 'anchored_bernstein.py',
                        'anchor_cdf.py', 'anchor_calibration.py', 'native_training.py', 'native_metrics.py')},
                    runtime=dict(torch=torch.__version__, numpy=np.__version__),
                    selection='first minimum shape_select native-bin NLL', outer_used_for_selection=False)
    model = shapes.ShapeHead(case, context(fit).shape[1], config['hidden'], config['dropout'])
    opt = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    fit_loader = loader(fit, stats, config['batch_size'], generator=generator)
    select_loader = loader(select, stats, config['batch_size'])
    root = Path(config['run_root'])/('smoke' if binding['smoke_only'] else 'formal')/case/f'seed_{seed}'/f'fold_{fold}'
    root.mkdir(parents=True, exist_ok=True)
    last = root/'last.pt'
    epoch0, best, stale, history = 0, float('inf'), 0, []
    if last.exists():
        if not resume:
            raise ValueError('Shape checkpoint exists; pass --resume')
        state = torch.load(last, map_location='cpu', weights_only=False)
        if state['identity'] != identity:
            raise ValueError('Shape checkpoint identity changed')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer'])
        epoch0, best, stale, history = state['epoch'], state['best'], state['stale'], state['history']
        torch.set_rng_state(state['torch_rng']); np.random.set_state(state['numpy_rng'])
        random.setstate(state['python_rng']); generator.set_state(state['loader_rng'])
    save_json(root/'identity.json', identity)
    cap = min(config['max_epochs'], epoch0+stop_after) if stop_after else config['max_epochs']
    for epoch in range(epoch0, cap):
        if stale >= config['patience']:
            break
        started = time.perf_counter(); model.train(); total, count, updates = 0., 0, 0
        for x, m, _, y in fit_loader:
            opt.zero_grad(set_to_none=True)
            objective = loss(model(x, m), y)
            if not torch.isfinite(objective):
                raise ValueError('Nonfinite native-bin shape objective')
            objective.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError('Inactive or nonfinite shape gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_norm']); opt.step()
            total += float(objective.detach())*len(y); count += len(y); updates += 1
        score = selection(model, select_loader)
        history.append(dict(epoch=epoch+1, fit_NLL=total/count, selection_NLL=score, updates=updates,
                            seconds=time.perf_counter()-started))
        if score < best:
            best, stale = score, 0
            save_checkpoint(root/'best.pt', dict(identity=identity, model=model.state_dict(), epoch=epoch+1, selection_NLL=score))
        else:
            stale += 1
        save_checkpoint(last, dict(identity=identity, model=model.state_dict(), optimizer=opt.state_dict(),
                                    epoch=epoch+1, best=best, stale=stale, history=history,
                                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                                    python_rng=random.getstate(), loader_rng=generator.get_state()))
        save_json(root/'history.json', history)
    complete = stale >= config['patience'] or len(history) >= config['max_epochs']
    selected = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(selected['model'])
    if selection(model, select_loader) != selected['selection_NLL']:
        raise ValueError('Selected shape score is not reproducible')
    # Only completed fits may read the outer table for any scored readout.
    metrics, records = predict(model, tables['outer'], stats, config['batch_size']) if complete else (None, None)
    if complete:
        save_json(root/'outer_predictions.json', records)
    result = dict(case=case, seed=seed, fold=fold, root=str(root), complete=complete,
                  smoke_only=binding['smoke_only'], best_epoch=selected['epoch'], epochs=len(history),
                  updates=sum(r['updates'] for r in history), parameters=sum(p.numel() for p in model.parameters()),
                  loop_seconds=sum(r['seconds'] for r in history), outer=metrics,
                  best_sha256=digest(root/'best.pt'), last_sha256=digest(last), history_sha256=digest(root/'history.json'),
                  predictions_sha256=digest(root/'outer_predictions.json') if complete else None,
                  official_VALID_used=False, official_TEST_used=False)
    save_json(root/'result.json', result)
    print(json.dumps(result, allow_nan=False), flush=True)
    return result
