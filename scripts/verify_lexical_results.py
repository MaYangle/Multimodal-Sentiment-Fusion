import json, platform, sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'D:/MYL/JevAffect')
from jevaffect.metrics import classification_metrics, negative_log_likelihood, brier_score, top_label_ece_15
root=Path('D:/MYL/JevAffect')
results=[]
for seed in [13,42,2026]:
    run=root/'runs'/f'lexical-full-{seed}'
    rows=[json.loads(line) for line in (run/'validation_predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    y=np.array([x['target'] for x in rows]); p=np.array([x['probabilities'] for x in rows])
    metrics=classification_metrics(y,p.argmax(1))
    original=json.loads((run/'summary.json').read_text())
    assert abs(metrics['accuracy']-original['validation']['accuracy'])<1e-12
    assert abs(metrics['macro_f1']-original['validation']['macro_f1'])<1e-12
    metrics.update(nll=negative_log_likelihood(y,p),brier=brier_score(y,p),ece15=top_label_ece_15(y,p))
    results.append({'seed':seed,'metrics':metrics,'train_samples':2722,'validation_samples':647})
report={'scope':'CH-SIMS v2 official validation only; supplemental lexical baseline; no test scored',
        'implementation':'scripts/train_lexical_pilot.py','python':sys.version,'numpy':np.__version__,
        'runs':results,'accuracy_mean':float(np.mean([x['metrics']['accuracy'] for x in results])),
        'accuracy_sample_std':float(np.std([x['metrics']['accuracy'] for x in results],ddof=1)),
        'macro_f1_mean':float(np.mean([x['metrics']['macro_f1'] for x in results])),
        'macro_f1_sample_std':float(np.std([x['metrics']['macro_f1'] for x in results],ddof=1)),
        'timing_note':'Existing seconds_this_invocation fields cover optimization, validation, and checkpoint I/O only; exclude input encoding and process startup.'}
(root/'docs'/'lexical_validation_results.json').write_text(json.dumps(report,indent=2,default=lambda x:x.tolist()),encoding='utf-8')
print({k:v for k,v in report.items() if k!='runs'})
