"""Full CH-SIMS v2 five-class CMC pseudo reproduction on common raw features.

This is a protocol/backbone adaptation of CMC, with its actual two-stage
pseudo/EMA/SupCon logic. The text encoder is a frozen contextual mBERT cache,
not CMC's original trainable BERT. Only train/valid are ever loaded.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,math,platform,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from jevaffect.cmc_raw_cache import load_raw_feature_pilot_train_valid
from vendor.cmc.builders import build_pseudo_pretraining_bundle,build_fusion_model,build_fusion_optimizer
from vendor.cmc.models import SupConLoss
from vendor.cmc.training import PseudoLabelBank,train_pseudo_pretrain_epoch,evaluate_pseudo_pretrain_epoch,train_fusion_epoch,evaluate_fusion
from jevaffect.metrics import classification_metrics,negative_log_likelihood,top_label_ece_15
from jevaffect.protocol import CLASSES
from jevaffect.training import _atomic_json,_atomic_jsonl,_atomic_torch_save

SEEDS=(13,42,2026)
INITIAL={'text':0.8,'audio':0.99,'vision':0.99}
GAMMA={'text':0.5,'audio':0.5,'vision':0.5}
EMA_INITIAL={'text':0.8,'audio':0.9,'vision':0.6}
EMA_GAMMA={'text':2.5,'audio':5.0,'vision':2.0}
MODEL={'embed_dim':32,'nhead':2,'transformer_layers':4,'out_dropout':0.3,'top_k':3,'temperature':0.07}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def sha(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
def copy_model_state(model):return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
def rng(gen):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'loader':gen.get_state()}
def restore(state,gen):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda']);gen.set_state(state['loader'])
def bank_snapshot(bank):return bank.snapshot()
def bank_restore(bank,values):
    if set(values)!=set(bank.values):raise ValueError('Pseudo-bank ID set changed')
    bank.values=values
def stage1_state(bundle,train_bank,valid_bank,opt_state,gen,history,resource,identity,epoch,best_value,best_epoch,stale,complete):
    return {'identity_sha256':identity,'epoch':epoch,'models':{m:copy_model_state(x) for m,x in bundle['models'].items()},
            'ema_models':{m:copy_model_state(x) for m,x in bundle['ema_models'].items()},
            'optimizers':{m:x.state_dict() for m,x in bundle['optimizers'].items()},
            'train_bank':bank_snapshot(train_bank),'valid_bank':bank_snapshot(valid_bank),
            'rng_state':rng(gen),'history':history,'resources':resource,
            'best_value':best_value,'best_epoch':best_epoch,'stale':stale,'completed':complete}

def train_seed(seed:int,*,resume:bool=False,max_stage1_epochs_this_call:int|None=None,
               max_stage2_epochs_this_call:int|None=None):
    if seed not in SEEDS:raise ValueError('Seed not pre-registered')
    if not torch.cuda.is_available():raise RuntimeError('CMC full run requires local CUDA')
    device=torch.device('cuda');cache=r/'data/feature_cache/simsv2_full_raw_common'
    train,valid,source=load_raw_feature_pilot_train_valid(cache)
    if len(train)!=2722 or len(valid)!=647:raise ValueError('Not the full official author cohort')
    settings={'seed':seed,'cache_source_sha256':source['source_sha256'],
              'source_method':'CMC pseudo c695f719; five-class protocol, common frozen mBERT/WavLM/DINOv2 backbone',
              'stage1_max_epochs':100,'stage2_max_epochs':100,'patience':10,'batch_size':64,
              'model':MODEL,'source_pseudo_momentum':INITIAL,'source_pseudo_gamma':GAMMA,
              'source_ema_momentum':EMA_INITIAL,'source_ema_gamma':EMA_GAMMA,
              'learning_rate':1e-3,'contrast_temperature':0.07,
              'uses_ground_truth_unimodal_labels':False,'test_scored':False}
    implementation=hashlib.sha256(b''.join((r/name).read_bytes() for name in
      ('scripts/train_cmc_full_raw.py','jevaffect/cmc_raw_cache.py','vendor/cmc/models.py',
       'vendor/cmc/training.py','vendor/cmc/builders.py'))).hexdigest()
    identity=hashlib.sha256(canonical({'settings':settings,'implementation':implementation,
                                      'cache_meta_sha256':source['meta_sha256'],
                                      'torch':torch.__version__,'numpy':np.__version__})).hexdigest()
    out=r/f'runs/cmc_full_raw_common/seed_{seed}'
    last1=out/'stage1_last.pt';best1=out/'stage1_best.pt'
    last2=out/'stage2_last.pt';best2=out/'best.pt'
    if out.exists() and not resume:raise FileExistsError('Run exists; resume exact checkpoint')
    if resume and not (last1.is_file() or last2.is_file()):raise FileNotFoundError('No checkpoint to resume')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=64,shuffle=True,generator=gen,num_workers=0)
    val=DataLoader(valid,batch_size=64,shuffle=False,num_workers=0)
    bundle=build_pseudo_pretraining_bundle(train,model_config=MODEL,
       optimizer_config={'lr':1e-3,'lr_bert':5e-5,'weight_decay_bert':1e-3},
       bert_model_dir=None,finetune_text=False,device=device)
    bank=PseudoLabelBank.from_multimodal_labels(train.ids,train.labels)
    valid_bank=PseudoLabelBank.from_multimodal_labels(valid.ids,valid.labels)
    contrast=SupConLoss(0.07)
    out.mkdir(parents=True,exist_ok=True)
    if not resume:
        _atomic_json(out/'config.json',settings)
        _atomic_json(out/'run_manifest.json',{'identity_sha256':identity,'implementation_sha256':implementation,
                'cache_meta_sha256':source['meta_sha256'],'source_sha256':source['source_sha256'],
                'class_order':list(CLASSES),'train_count':2722,'valid_count':647,'test_scored':False})
        _atomic_json(out/'environment.json',{'python':sys.version.split()[0],'platform':platform.platform(),
                      'torch':torch.__version__,'numpy':np.__version__,
                      'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda})
    history1=[];resource1=[];stale1=0;best_value=float('inf');best_epoch1=0;first_epoch=1
    if resume:
        stage1=torch.load(last1,map_location='cpu',weights_only=True)
        if stage1['identity_sha256']!=identity:raise ValueError('Stage1 resume identity mismatch')
        for kind in ('models','ema_models'):
            for modality in bundle[kind]:bundle[kind][modality].load_state_dict(stage1[kind][modality])
        for modality in bundle['optimizers']:bundle['optimizers'][modality].load_state_dict(stage1['optimizers'][modality])
        bank_restore(bank,stage1['train_bank']);bank_restore(valid_bank,stage1['valid_bank'])
        restore(stage1['rng_state'],gen)
        history1=stage1['history'];resource1=stage1['resources'];stale1=stage1['stale']
        best_value=stage1['best_value'];best_epoch1=stage1['best_epoch'];first_epoch=stage1['epoch']+1
    stage1_this_call=0
    for epoch in range(first_epoch,101):
        if stale1>=10 or (max_stage1_epochs_this_call is not None and stage1_this_call>=max_stage1_epochs_this_call):break
        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();begin=time.perf_counter()
        trained=train_pseudo_pretrain_epoch(bundle['models'],bundle['ema_models'],bundle['optimizers'],
                  loader,bank,epoch=epoch,device=device,contrastive_criterion=contrast,
                  initial_pseudo_momenta=INITIAL,pseudo_momentum_gamma=GAMMA,
                  initial_ema_momenta=EMA_INITIAL,ema_momentum_gamma=EMA_GAMMA)
        checked=evaluate_pseudo_pretrain_epoch(bundle['models'],bundle['ema_models'],val,valid_bank,
                  epoch=epoch,device=device,contrastive_criterion=contrast,
                  initial_pseudo_momenta=INITIAL,pseudo_momentum_gamma=GAMMA)
        torch.cuda.synchronize()
        score=float(checked['total'])
        if not math.isfinite(score) or not math.isfinite(float(trained['total'])):raise ValueError('Nonfinite CMC pseudo objective')
        if score<best_value-1e-6:
            best_value=score;best_epoch1=epoch;stale1=0
        else:stale1+=1
        row={'stage':'pseudo_pretrain','epoch':epoch,'train_total':trained['total'],
             'valid_total':score,'best_valid_total':best_value,'best_epoch':best_epoch1}
        history1.append(row)
        resource1.append({'stage':'pseudo_pretrain','epoch':epoch,
                          'train_valid_seconds':time.perf_counter()-begin,
                          'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        complete=stale1>=10 or epoch>=100
        state=stage1_state(bundle,bank,valid_bank,bundle['optimizers'],gen,
                           history1,resource1,identity,epoch,best_value,best_epoch1,stale1,complete)
        _atomic_torch_save(last1,state)
        if best_epoch1==epoch:_atomic_torch_save(best1,state)
        _atomic_jsonl(out/'stage1_metrics.jsonl',history1)
        _atomic_jsonl(out/'stage1_resources.jsonl',resource1)
        print(json.dumps(row),flush=True)
        stage1_this_call+=1
        if complete:break
    if not best1.is_file() or not last1.is_file():raise ValueError('CMC pseudo phase has no selected weights')
    latest1=torch.load(last1,map_location='cpu',weights_only=True)
    if not latest1['completed']:
        return {'seed':seed,'stage':'pseudo_pretrain','completed':False,
                'last_epoch':latest1['epoch'],'best_valid_total':best_value}
    selected1=torch.load(best1,map_location='cpu',weights_only=True)
    if selected1['identity_sha256']!=identity:raise ValueError('Stage1 selected weights changed identity')
    for modality in bundle['ema_models']:
        bundle['ema_models'][modality].load_state_dict(selected1['ema_models'][modality])
    bank_restore(bank,selected1['train_bank']);bank_restore(valid_bank,selected1['valid_bank'])
    # Stage2 starts with the selected source-style EMA unimodal encoders.
    fusion,transfer=build_fusion_model(train,bundle['ema_models'],model_config=MODEL,
                     bert_model_dir=None,finetune_text=False,device=device)
    _atomic_json(out/'transfer.json',transfer)
    optimizer=build_fusion_optimizer(fusion,lr=1e-3,lr_bert=5e-5,weight_decay_bert=1e-3)
    history2=[];resource2=[];best_acc=-1.;best_nll=float('inf');best_epoch2=0;stale2=0;start2=1
    if resume and last2.is_file():
        stage2=torch.load(last2,map_location='cpu',weights_only=True)
        if stage2['identity_sha256']!=identity:raise ValueError('Stage2 resume identity mismatch')
        fusion.load_state_dict(stage2['model']);optimizer.load_state_dict(stage2['optimizer'])
        bank_restore(bank,stage2['train_bank']);bank_restore(valid_bank,stage2['valid_bank'])
        restore(stage2['rng_state'],gen)
        history2=stage2['history'];resource2=stage2['resources']
        best_acc=stage2['best_accuracy'];best_nll=stage2['best_nll']
        best_epoch2=stage2['best_epoch'];stale2=stage2['stale'];start2=stage2['epoch']+1
        if stage2['completed']:
            if not best2.is_file():raise ValueError('Completed fusion run lacks selected weights')
            return json.loads((out/'summary.json').read_text(encoding='utf-8'))
    stage2_this_call=0
    for epoch in range(start2,101):
        if stale2>=10 or (max_stage2_epochs_this_call is not None and stage2_this_call>=max_stage2_epochs_this_call):break
        torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();begin=time.perf_counter()
        trained=train_fusion_epoch(fusion,loader,optimizer,bank,device=device,contrastive_criterion=contrast)
        result=evaluate_fusion(fusion,val,valid_bank,device=device,contrastive_criterion=contrast)
        torch.cuda.synchronize()
        labels=np.asarray(result['labels'],dtype=np.int64)
        probs=np.asarray(result['probabilities_uncalibrated'],dtype=np.float64)
        metric=classification_metrics(labels,probs.argmax(-1))
        if len(result['ids'])!=647 or abs(metric['accuracy']-result['accuracy'])>1e-12:
            raise ValueError('CMC full cohort validation mismatch')
        nll=negative_log_likelihood(labels,probs)
        if not math.isfinite(nll) or not math.isfinite(float(trained['total'])):
            raise ValueError('Nonfinite CMC fusion loss')
        improved=metric['accuracy']>best_acc or (metric['accuracy']==best_acc and nll<best_nll)
        if improved:best_acc=metric['accuracy'];best_nll=nll;best_epoch2=epoch;stale2=0
        else:stale2+=1
        row={'stage':'fusion','epoch':epoch,'train_total':trained['total'],
             'valid_accuracy':metric['accuracy'],'valid_macro_f1':metric['macro_f1'],
             'valid_nll':nll,'valid_ece15':top_label_ece_15(labels,probs),
             'best_valid_accuracy':best_acc,'best_epoch':best_epoch2}
        history2.append(row)
        resource2.append({'stage':'fusion','epoch':epoch,'train_valid_seconds':time.perf_counter()-begin,
                          'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        complete=stale2>=10 or epoch>=100
        state={'identity_sha256':identity,'epoch':epoch,'model':copy_model_state(fusion),
               'optimizer':optimizer.state_dict(),'train_bank':bank_snapshot(bank),
               'valid_bank':bank_snapshot(valid_bank),'rng_state':rng(gen),
               'history':history2,'resources':resource2,'best_accuracy':best_acc,
               'best_nll':best_nll,'best_epoch':best_epoch2,'stale':stale2,'completed':complete}
        _atomic_torch_save(last2,state)
        if improved:
            _atomic_torch_save(best2,{'identity_sha256':identity,'epoch':epoch,
                                     'model':copy_model_state(fusion),'class_order':list(CLASSES),
                                     'cache_source_sha256':source['source_sha256']})
            rows=[{'ID':sid,'split':'valid','label':int(label),'probabilities':p.tolist(),
                   'argmax':CLASSES[int(np.argmax(p))],'class_order':list(CLASSES)}
                  for sid,label,p in zip(result['ids'],labels,probs)]
            _atomic_jsonl(out/'validation_predictions.jsonl',rows)
        _atomic_jsonl(out/'stage2_metrics.jsonl',history2)
        _atomic_jsonl(out/'stage2_resources.jsonl',resource2)
        print(json.dumps(row),flush=True)
        stage2_this_call+=1
        if complete:break
    summary={'seed':seed,'stage1_best_epoch':best_epoch1,'stage1_last_epoch':latest1['epoch'],
             'stage2_best_epoch':best_epoch2,'stage2_last_epoch':history2[-1]['epoch'] if history2 else 0,
             'best_accuracy':best_acc,'best_nll':best_nll,'completed':stale2>=10 or len(history2)>=100,
             'source_sha256':source['source_sha256'],
             'scope':'official CH-SIMS v2 full train/valid, common raw-video frozen encoders; no test scores'}
    _atomic_json(out/'summary.json',summary)
    train.close();valid.close();return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS,default=13)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-stage1-epochs-this-call',type=int)
    parser.add_argument('--max-stage2-epochs-this-call',type=int)
    args=parser.parse_args()
    print(json.dumps(train_seed(args.seed,resume=args.resume,
          max_stage1_epochs_this_call=args.max_stage1_epochs_this_call,
          max_stage2_epochs_this_call=args.max_stage2_epochs_this_call)),flush=True)
