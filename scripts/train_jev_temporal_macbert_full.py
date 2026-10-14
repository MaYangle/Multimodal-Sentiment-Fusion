"""Full-cohort MacBERT/WavLM/DINOv2 temporal/ordinal fusion control.

Runs on the same frozen MacBERT/WavLM/DINOv2 features as the matched ALMT
adaptation. Only the official train and validation cohorts are loaded.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,hashlib,json,random,sys,time,platform
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r))
from jevaffect.raw_temporal_data import load_temporal_train_valid
from jevaffect.jev_temporal import JevTemporalOrdinal
from jevaffect.metrics import classification_metrics,negative_log_likelihood,top_label_ece_15
from jevaffect.protocol import CLASSES
from jevaffect.training import _atomic_json,_atomic_jsonl,_atomic_torch_save

def shafile(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')
def rng_state(gen):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'loader':gen.get_state()}
def restore_rng(state,gen):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
    gen.set_state(state['loader'])
def move(batch,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
def objective(output,labels,batch):
    primary=F.cross_entropy(output['logits'],labels)
    branches=[]
    for name in ('audio','vision'):
        logits=output[name+'_window_logits']
        valid=output['valid_windows'][...,0 if name=='audio' else 1]
        loss=F.cross_entropy(logits.flatten(0,1),labels[:,None].expand_as(valid).reshape(-1),reduction='none')
        branches.append((loss*valid.reshape(-1)).sum()/valid.sum().clamp_min(1))
    return primary+0.15*sum(branches),primary.detach()
@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ids=[];labels=[];probs=[]
    for batch in loader:
        values=move(batch,device);out=model(values)
        ids.extend(batch['id']);labels.extend(values['label'].cpu().tolist())
        probs.extend(out['logits'].softmax(-1).cpu().tolist())
    y=np.asarray(labels,dtype=np.int64);p=np.asarray(probs,dtype=np.float64)
    m=classification_metrics(y,p.argmax(1))
    return {'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'nll':negative_log_likelihood(y,p),
            'ece15':top_label_ece_15(y,p),'ids':ids,'labels':labels,'probs':probs}
def train_seed(seed:int,resume:bool=False,max_epochs_this_call:int|None=None,utility_targets_path:str|None=None,utility_weight:float=0.2,matched_control:bool=False,gate_kind:str='learned',temporal:bool=True):
    if seed not in (13,42,2026):raise ValueError('Only preregistered seeds allowed')
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA is required')
    cache=r/'data/feature_cache/simsv2_full_macbert_raw_av'
    train,valid,source=load_temporal_train_valid(cache)
    if len(train)!=2722 or len(valid)!=647:raise ValueError('Not the official full train/validation cohort')
    if gate_kind not in ('learned','entropy','global_jsd','uniform'):raise ValueError('Invalid gate kind')
    if (utility_targets_path is not None or matched_control) and (gate_kind!='learned' or not temporal):
        raise ValueError('Utility comparison requires the learned temporal gate')
    targets_by_id=None;target_sha=None
    if utility_targets_path is not None:
        if matched_control:raise ValueError('Matched control cannot consume utility targets')
        target_path=Path(utility_targets_path).expanduser().resolve()
        oof_root=target_path.parent
        oof_summary=json.loads((oof_root/'summary.json').read_text(encoding='utf-8'))
        scale=json.loads((oof_root/'standardization.json').read_text(encoding='utf-8'))
        train_ids=json.loads((oof_root/'train_ids.json').read_text(encoding='utf-8'))
        if oof_summary['cache_source_sha256']!=source['source_sha256'] or train_ids!=list(train.ids):
            raise ValueError('Utility targets do not match the exact training split')
        if oof_summary['validation_or_test_label_used'] is not False or oof_summary['teacher_source_group_cross_fold_overlap']!=0:
            raise ValueError('Utility targets are not strictly training-only group-heldout targets')
        if hashlib.sha256((oof_root/'train_ids.json').read_bytes()).hexdigest()!=oof_summary['train_ids_sha256']:
            raise ValueError('OOF ID index SHA mismatch')
        target_sha=hashlib.sha256(target_path.read_bytes()).hexdigest()
        if target_sha!=scale['standardized_target_sha256'] or scale['source_target_sha256']!=oof_summary['target_sha256']:
            raise ValueError('Standardized OOF target SHA mismatch')
        target_array=np.load(target_path,allow_pickle=False)
        if target_array.shape!=(len(train),8,2) or int(np.isfinite(target_array).sum())!=oof_summary['target_valid_count']:
            raise ValueError('OOF target shape/mask mismatch')
        targets_by_id=dict(zip(train.ids,target_array))
    if not np.isfinite(utility_weight) or utility_weight<0:
        raise ValueError('utility_weight must be finite and nonnegative')
    settings={'seed':seed,'model':'JevTemporalOrdinal','hidden':96,'windows':8,
              'gate_kind':gate_kind,'temporal':temporal,'utility_supervision':targets_by_id is not None,
              'utility_weight':utility_weight if targets_by_id is not None else 0.0,
              'utility_targets_sha256':target_sha,'matched_control':matched_control,
              'branch_ce_weight':0.15,'max_epochs':80,'batch_size':32,'patience':12,
              'lr':3e-4,'weight_decay':1e-4,'selected_metric':'valid_accuracy',
              'cache_source_sha256':source['source_sha256'],'cohort':'2722/647 official CH-SIMS v2 train/valid; MacBERT text override'}
    codehash=hashlib.sha256(b''.join(Path(r/file).read_bytes() for file in
        ('jevaffect/jev_temporal.py','jevaffect/raw_temporal_data.py','jevaffect/cmc_raw_cache.py','scripts/train_jev_temporal_macbert_full.py'))).hexdigest()
    identity=hashlib.sha256(canonical({'settings':settings,'codehash':codehash,
                                       'meta_sha256':source['meta_sha256']})).hexdigest()
    run_family=('jev_temporal_macbert_full_utility' if targets_by_id is not None else
                'jev_temporal_macbert_full_utility_control' if matched_control else
                'jev_temporal_macbert_full_{}_{}'.format(gate_kind,'time' if temporal else 'no_time'))
    folder=r/f'runs/{run_family}/seed_{seed}'
    last=folder/'last.pt';best=folder/'best.pt'
    if folder.exists() and not resume:raise FileExistsError('Existing run; use --resume')
    if resume and not last.is_file():raise FileNotFoundError('No last.pt to resume')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    gen=torch.Generator().manual_seed(seed)
    loader=DataLoader(train,batch_size=32,shuffle=True,generator=gen,num_workers=0)
    validation=DataLoader(valid,batch_size=32,shuffle=False,num_workers=0)
    model=JevTemporalOrdinal(hidden=96,windows=8,gate_kind=gate_kind,temporal=temporal).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    folder.mkdir(parents=True,exist_ok=True)
    history=[];resources=[];start=1;stale=0;best_acc=-1;best_nll=float('inf');best_epoch=0
    if resume:
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state['identity_sha256']!=identity:raise ValueError('Resume config/code/data identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        start=state['epoch']+1;history=state['history'];resources=state['resources']
        stale=state['stale'];best_acc=state['best_accuracy'];best_nll=state['best_nll'];best_epoch=state['best_epoch']
        restore_rng(state['rng_state'],gen)
        if state['stale']>=12 or state['epoch']>=80:
            report=folder/'summary.json'
            if not best.is_file() or not report.is_file():
                raise ValueError('Completed checkpoint is missing its selected artifacts')
            result=json.loads(report.read_text(encoding='utf-8'))
            if result.get('completed') is not True:
                raise ValueError('Completed checkpoint and summary disagree')
            train.base.close();valid.base.close()
            return result
    else:
        _atomic_json(folder/'config.json',settings)
        _atomic_json(folder/'run_manifest.json',{'identity_sha256':identity,'code_sha256':codehash,
                                                 'cache_meta_sha256':source['meta_sha256'],
                                                 'class_order':list(CLASSES),'scope':'full author train/valid; no test scoring'})
        _atomic_json(folder/'environment.json',{'python':sys.version.split()[0],'torch':torch.__version__,
                                                 'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0),
                                                 'platform':platform.platform()})
    epochs_this_call=0
    for epoch in range(start,81):
        if max_epochs_this_call is not None and epochs_this_call>=max_epochs_this_call:break
        model.train();torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();started=time.perf_counter()
        total=0.;correct=0
        for batch in loader:
            values=move(batch,device);optimizer.zero_grad(set_to_none=True)
            if targets_by_id is not None:
                targets=torch.from_numpy(np.stack([targets_by_id[sid] for sid in batch['id']])).to(device)
                output=model(values,utility_targets=targets,utility_weight=utility_weight)
            else:
                output=model(values)
            loss,_=objective(output,values['label'].long(),values)
            if targets_by_id is not None:loss=loss+output['utility_loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite candidate training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            total+=float(loss.detach().item())*len(batch['id'])
            correct+=int((output['logits'].argmax(-1)==values['label']).sum())
        ev=evaluate(model,validation,device);torch.cuda.synchronize()
        improved=ev['accuracy']>best_acc or (ev['accuracy']==best_acc and ev['nll']<best_nll)
        if improved:
            best_acc=ev['accuracy'];best_nll=ev['nll'];best_epoch=epoch;stale=0
            _atomic_torch_save(best,{'identity_sha256':identity,'epoch':epoch,'model':model.state_dict(),
                                     'class_order':list(CLASSES)})
            _atomic_jsonl(folder/'validation_predictions.jsonl',[
                {'ID':sid,'label':int(y),'class_order':list(CLASSES),'probabilities':p,
                 'argmax':CLASSES[int(np.argmax(p))],'split':'valid'}
                for sid,y,p in zip(ev['ids'],ev['labels'],ev['probs'])])
        else:stale+=1
        row={'epoch':epoch,'train_loss':total/len(train),'train_accuracy':correct/len(train),
             'valid_accuracy':ev['accuracy'],'valid_macro_f1':ev['macro_f1'],'valid_nll':ev['nll'],
             'valid_ece15':ev['ece15'],'best_valid_accuracy':best_acc,'best_epoch':best_epoch}
        history.append(row)
        resources.append({'epoch':epoch,'train_valid_wall_seconds':time.perf_counter()-started,
                          'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        _atomic_jsonl(folder/'metrics.jsonl',history);_atomic_jsonl(folder/'resources.jsonl',resources)
        _atomic_torch_save(last,{'identity_sha256':identity,'epoch':epoch,'model':model.state_dict(),
                                 'optimizer':optimizer.state_dict(),'rng_state':rng_state(gen),
                                 'best_accuracy':best_acc,'best_nll':best_nll,'best_epoch':best_epoch,
                                 'stale':stale,'history':history,'resources':resources})
        print(json.dumps(row),flush=True);epochs_this_call+=1
        if stale>=12:break
    result={'seed':seed,'best_epoch':best_epoch,'best_validation_accuracy':best_acc,
            'last_epoch':history[-1]['epoch'] if history else 0,'completed':stale>=12 or len(history)>=80,
            'gate_kind':gate_kind,'temporal':temporal,
            'scope':'full official train/valid; OOF utility targets applied' if targets_by_id is not None else 'full official train/valid; utility supervision not applied'}
    _atomic_json(folder/'summary.json',result)
    train.base.close();valid.base.close();return result
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=(13,42,2026),default=13)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-epochs-this-call',type=int)
    parser.add_argument('--utility-targets')
    parser.add_argument('--utility-weight',type=float,default=0.2)
    parser.add_argument('--matched-control',action='store_true')
    parser.add_argument('--gate-kind',choices=('learned','entropy','global_jsd','uniform'),default='learned')
    parser.add_argument('--no-temporal',action='store_true')
    args=parser.parse_args()
    print(json.dumps(train_seed(args.seed,args.resume,args.max_epochs_this_call,args.utility_targets,args.utility_weight,args.matched_control,args.gate_kind,not args.no_temporal)),flush=True)
