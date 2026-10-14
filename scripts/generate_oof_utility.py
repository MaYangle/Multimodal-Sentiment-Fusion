"""Train-only source-group OOF modality-window loss-difference targets.

These targets support a Jev-inspired utility hypothesis. They are NOT evidence
of a novel improvement until controlled full-cohort model experiments pass.
"""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
repo=Path('D:/MYL/JevAffect');sys.path.insert(0,str(repo))
from jevaffect.jev_temporal import JevTemporalOrdinal
from jevaffect.raw_temporal_data import load_temporal_train_valid
from jevaffect.training import _atomic_json,_atomic_torch_save

def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def move(batch,device):return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
def objective(out,batch):
    y=batch['label'];main=F.cross_entropy(out['logits'],y)
    aux=[]
    for idx,mod in enumerate(('audio','vision')):
        valid=out['valid_windows'][...,idx]
        logits=out[mod+'_window_logits']
        ce=F.cross_entropy(logits.flatten(0,1),y[:,None].expand_as(valid).reshape(-1),reduction='none')
        aux.append((ce*valid.reshape(-1)).sum()/valid.sum().clamp_min(1))
    return main+0.15*sum(aux)
def folds_by_source(ids:list[str],seed:int=13,k:int=3)->list[list[int]]:
    groups={}
    for idx,sid in enumerate(ids):groups.setdefault(sid.split('/')[0],[]).append(idx)
    if len(groups)<k:raise ValueError('Not enough source groups for disjoint OOF folds')
    rng=random.Random(seed);names=list(groups);rng.shuffle(names)
    names.sort(key=lambda name:len(groups[name]),reverse=True)
    folds=[[] for _ in range(k)]
    for name in names:folds[min(range(k),key=lambda i:len(folds[i]))].extend(groups[name])
    if set().union(*(set(x) for x in folds))!=set(range(len(ids))):raise ValueError('Invalid OOF partition')
    return folds
def make_targets(batch,model,device:torch.device,window_count:int)->tuple[np.ndarray,np.ndarray]:
    x=move(batch,device);y=x['label'].long()
    with torch.no_grad():
        whole=model(x)
        duration=whole['time_windows_seconds'][:,-1,1]
        loss_all=F.cross_entropy(whole['logits'],y,reduction='none')
        target=torch.full((len(y),window_count,2),float('nan'),device=device)
        valid=whole['valid_windows']
        for channel,modality in enumerate(('audio','vision')):
            original_mask=x[modality+'_mask']
            times=x[modality+'_times']
            for window in range(window_count):
                lo=whole['time_windows_seconds'][:,window,0:1]
                hi=whole['time_windows_seconds'][:,window,1:2]
                in_window=(times>=lo)&((times<=hi) if window==window_count-1 else (times<hi))
                removed={name:value for name,value in x.items()}
                removed[modality+'_mask']=original_mask&~in_window
                alt=model(removed,duration_override=duration)
                loss_alt=F.cross_entropy(alt['logits'],y,reduction='none')
                benefit=(loss_alt-loss_all).clamp(-3,3)
                target[:,window,channel]=torch.where(valid[:,window,channel],benefit,
                                                      torch.full_like(benefit,float('nan')))
    return target.cpu().numpy(),valid.cpu().numpy()
def main():
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('OOF pilot requires real local CUDA')
    output=repo/'runs/jev_temporal_oof_pilot'
    if output.exists():raise FileExistsError('OOF output exists; inspect it before rerunning')
    output.mkdir(parents=True)
    train,valid,identity=load_temporal_train_valid(repo/'data/feature_cache/simsv2_raw_prefix')
    ids=list(train.ids);parts=folds_by_source(ids)
    labels=np.asarray(train.base.labels,dtype=np.int64)
    source_groups={sid:sid.split('/')[0] for sid in ids}
    all_targets=np.full((len(train),8,2),np.nan,dtype=np.float32)
    teacher_info=[]
    for fold,held in enumerate(parts):
        included=[idx for idx in range(len(train)) if idx not in set(held)]
        train_groups={source_groups[ids[idx]] for idx in included}
        held_groups={source_groups[ids[idx]] for idx in held}
        if train_groups & held_groups:raise AssertionError('OOF source group leakage')
        seed=13+fold
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        generator=torch.Generator().manual_seed(seed)
        dataloader=DataLoader(Subset(train,included),batch_size=16,shuffle=True,generator=generator,num_workers=0)
        teacher=JevTemporalOrdinal(hidden=96,windows=8,gate_kind='uniform',temporal=True).to(device)
        optimizer=torch.optim.AdamW(teacher.parameters(),lr=1e-3,weight_decay=1e-4)
        began=time.perf_counter();history=[]
        for epoch in range(1,9):
            teacher.train();total=0
            for batch in dataloader:
                x=move(batch,device)
                # Train teacher for actual modality removals; never use OOF labels
                # in optimizer steps, checkpoint selection, or augmentations.
                dropout=torch.rand((len(x['label']),2),generator=generator)<0.15
                for ch,modality in enumerate(('audio','vision')):
                    x[modality+'_mask']=x[modality+'_mask']&~dropout[:,ch].to(device).unsqueeze(-1)
                optimizer.zero_grad(set_to_none=True)
                out=teacher(x);loss=objective(out,x)
                if not torch.isfinite(loss):raise ValueError('Nonfinite teacher loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(teacher.parameters(),1.0);optimizer.step()
                total+=float(loss.detach())*len(batch['id'])
            history.append({'epoch':epoch,'train_loss':total/len(included)})
        teacher.eval();heldloader=DataLoader(Subset(train,held),batch_size=16,shuffle=False,num_workers=0)
        id_to_index={ids[idx]:idx for idx in held}
        for batch in heldloader:
            target,mask=make_targets(batch,teacher,device,8)
            for sid,row in zip(batch['id'],target):
                all_targets[id_to_index[sid]]=row
        path=output/f'teacher_fold_{fold}.pt'
        _atomic_torch_save(path,{'fold':fold,'seed':seed,'model':teacher.state_dict(),
                                 'cache_source_sha256':identity['source_sha256']})
        teacher_info.append({'fold':fold,'teacher_train':len(included),'heldout_target_rows':len(held),
                             'train_source_groups':sorted(train_groups),'heldout_source_groups':sorted(held_groups),
                             'train_history':history,'wall_seconds':time.perf_counter()-began,
                             'checkpoint_sha256':sha(path)})
        print(json.dumps({'fold':fold,'train':len(included),'oof':len(held),
                          'wall_seconds':teacher_info[-1]['wall_seconds']}),flush=True)
    if not np.any(np.isfinite(all_targets)):raise ValueError('No valid utility targets generated')
    np.save(output/'utility_targets.npy',all_targets,allow_pickle=False)
    _atomic_json(output/'train_ids.json',ids)
    finite=all_targets[np.isfinite(all_targets)]
    summary={'source':'verified real 339-train raw-video pilot only','dataset':'CH-SIMS-v2.0',
             'cache_source_sha256':identity['source_sha256'],'train_rows':len(train),
             'target_shape':list(all_targets.shape),'target_valid_count':int(len(finite)),
             'target_mean':float(np.mean(finite)),'target_std':float(np.std(finite)),
             'target_positive_fraction':float(np.mean(finite>0)),
             'target_sha256':sha(output/'utility_targets.npy'),
             'train_ids_sha256':sha(output/'train_ids.json'),
             'teacher_source_group_cross_fold_overlap':0,
             'teacher_training_labels_used':'training fold only',
             'target_annotations_used':'heldout fold labels, still official training split',
             'validation_or_test_label_used':False,
             'fixed_teacher_epochs':8,'fixed_windows':8,'teacher_modality_dropout':0.15,
             'teacher_folds':teacher_info,
             'research_limit':'small biased pilot; no originality or performance benefit claim'}
    _atomic_json(output/'summary.json',summary)
    train.base.close();valid.base.close()
    print(json.dumps({k:v for k,v in summary.items() if k!='teacher_folds'}),flush=True)
if __name__=='__main__':main()
