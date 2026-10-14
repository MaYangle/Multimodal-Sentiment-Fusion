"""Train a small, real two-stage CMC pseudo pilot on the raw-video subset."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys,hashlib,json,time,random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from jevaffect.cmc_raw_cache import load_raw_feature_pilot_train_valid
from vendor.cmc.builders import build_pseudo_pretraining_bundle,build_fusion_model,build_fusion_optimizer
from vendor.cmc.models import SupConLoss
from vendor.cmc.training import PseudoLabelBank,train_pseudo_pretrain_epoch,evaluate_pseudo_pretrain_epoch,train_fusion_epoch,evaluate_fusion
from jevaffect.protocol import CLASSES
from jevaffect.metrics import classification_metrics,negative_log_likelihood,top_label_ece_15
from jevaffect.training import _atomic_torch_save,_atomic_json,_atomic_jsonl

SEED=13
torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);random.seed(SEED);np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
device=torch.device('cuda')
if not torch.cuda.is_available():raise RuntimeError('CMC real pilot requires actual CUDA GPU')
output=r/'runs/cmc_raw_pilot/seed_13'
if output.exists():raise RuntimeError('Pilot already exists; inspect rather than overwrite')
output.mkdir(parents=True)
train,valid,cache_id=load_raw_feature_pilot_train_valid(r/'data/feature_cache/simsv2_raw_prefix')
assert len(train)==339 and len(valid)==43
model={'embed_dim':32,'nhead':2,'transformer_layers':4,'out_dropout':0.3,'top_k':3,'temperature':0.07}
settings={'seed':SEED,'stage1_epochs':5,'stage2_epochs':15,'batch_size':16,'finetune_text':False,'model':model,'cache_identity':cache_id['source_sha256'],'text_mode':cache_id['text_input_mode'],'research_scope':'biased raw-prefix pilot; no formal benchmark','test_scored':False,'source_method':'CMC pseudo-label port of c695f719'}
_atomic_json(output/'config.json',settings)
_atomic_json(output/'environment.json',{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,'numpy':np.__version__})
gen=torch.Generator().manual_seed(SEED)
tr=DataLoader(train,batch_size=16,shuffle=True,generator=gen,num_workers=0,drop_last=False)
va=DataLoader(valid,batch_size=16,shuffle=False,num_workers=0,drop_last=False)
bundle=build_pseudo_pretraining_bundle(train,model_config=model,optimizer_config={'lr':1e-3,'lr_bert':5e-5,'weight_decay_bert':1e-3},bert_model_dir=None,finetune_text=False,device=device)
tb=PseudoLabelBank.from_multimodal_labels(train.ids,train.labels)
vb=PseudoLabelBank.from_multimodal_labels(valid.ids,valid.labels)
contrast=SupConLoss(0.07)
initial={'text':0.8,'audio':0.99,'vision':0.99}
gamma={'text':0.5,'audio':0.5,'vision':0.5}
ema_initial={'text':0.8,'audio':0.9,'vision':0.6}
ema_gamma={'text':2.5,'audio':5.0,'vision':2.0}
history=[];resources=[];start=time.perf_counter()
for epoch in range(1,6):
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();t=time.perf_counter()
    train_result=train_pseudo_pretrain_epoch(bundle['models'],bundle['ema_models'],bundle['optimizers'],tr,tb,epoch=epoch,device=device,contrastive_criterion=contrast,initial_pseudo_momenta=initial,pseudo_momentum_gamma=gamma,initial_ema_momenta=ema_initial,ema_momentum_gamma=ema_gamma)
    valid_result=evaluate_pseudo_pretrain_epoch(bundle['models'],bundle['ema_models'],va,vb,epoch=epoch,device=device,contrastive_criterion=contrast,initial_pseudo_momenta=initial,pseudo_momentum_gamma=gamma)
    torch.cuda.synchronize()
    row={'stage':'pseudo_pretrain','epoch':epoch,'train_total':train_result['total'],'valid_total':valid_result['total'],'valid_samples':valid_result['samples']}
    if not all(np.isfinite(v) for k,v in row.items() if k in ('train_total','valid_total')):raise ValueError('nonfinite pseudo objective')
    history.append(row);resources.append({'stage':'pseudo_pretrain','epoch':epoch,'train_valid_seconds':time.perf_counter()-t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
    _atomic_jsonl(output/'metrics.jsonl',history);_atomic_jsonl(output/'resources.jsonl',resources)
    print(json.dumps(row),flush=True)
fusion,transfer=build_fusion_model(train,bundle['ema_models'],model_config=model,bert_model_dir=None,finetune_text=False,device=device)
_atomic_json(output/'transfer.json',transfer)
optimizer=build_fusion_optimizer(fusion,lr=1e-3,lr_bert=5e-5,weight_decay_bert=1e-3)
best_acc=-1.;best_nll=float('inf');best_epoch=0;stale=0
for epoch in range(1,16):
    torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();t=time.perf_counter()
    train_result=train_fusion_epoch(fusion,tr,optimizer,tb,device=device,contrastive_criterion=contrast)
    valid_result=evaluate_fusion(fusion,va,vb,device=device,contrastive_criterion=contrast)
    torch.cuda.synchronize()
    labels=np.asarray(valid_result['labels'],dtype=np.int64)
    probs=np.asarray(valid_result['probabilities_uncalibrated'],dtype=np.float64)
    pred=probs.argmax(1)
    metrics=classification_metrics(labels,pred)
    if abs(metrics['accuracy']-valid_result['accuracy'])>1e-12:raise ValueError('metric mismatch')
    row={'stage':'fusion','epoch':epoch,'train_total':train_result['total'],'valid_accuracy':metrics['accuracy'],'valid_macro_f1':metrics['macro_f1'],'valid_nll':negative_log_likelihood(labels,probs),'valid_ece15':top_label_ece_15(labels,probs),'samples':metrics['n_samples']}
    history.append(row);resources.append({'stage':'fusion','epoch':epoch,'train_valid_seconds':time.perf_counter()-t,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
    improved=row['valid_accuracy']>best_acc or (row['valid_accuracy']==best_acc and row['valid_nll']<best_nll)
    if improved:
        best_acc=row['valid_accuracy'];best_nll=row['valid_nll'];best_epoch=epoch;stale=0
        state={'source_sha256':cache_id['source_sha256'],'seed':SEED,'epoch':epoch,'class_order':list(CLASSES),'model_config':model,'state_dict':fusion.state_dict()}
        _atomic_torch_save(output/'best.pt',state)
        records=[{'ID':sample_id,'split':'valid','label':int(label),'probabilities':p.tolist(),'argmax':CLASSES[int(np.argmax(p))],'class_order':list(CLASSES)} for sample_id,label,p in zip(valid_result['ids'],labels,probs)]
        _atomic_jsonl(output/'validation_predictions.jsonl',records)
    else:stale+=1
    _atomic_jsonl(output/'metrics.jsonl',history);_atomic_jsonl(output/'resources.jsonl',resources)
    print(json.dumps(row),flush=True)
    if stale>=5:break
summary={'scope':'339/43 biased raw-prefix pilot; NOT full-cohort benchmark','best_epoch':best_epoch,'best_accuracy':best_acc,'best_nll':best_nll,'seed':SEED,'stage1_epochs':5,'stage2_epochs':epoch,'train_valid_seconds':sum(x['train_valid_seconds'] for x in resources),'peak_cuda_allocated_bytes':max(x['peak_cuda_allocated_bytes'] for x in resources),'source_sha256':cache_id['source_sha256'],'test_scored':False}
_atomic_json(output/'summary.json',summary)
train.close();valid.close();print(json.dumps(summary),flush=True)
