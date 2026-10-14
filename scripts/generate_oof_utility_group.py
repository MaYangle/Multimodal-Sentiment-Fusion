"""Group-disjoint train-only OOF window utility targets with fold recovery.

The group holdout's 540 validation labels and the official valid/test labels
are excluded from target generation, teacher training and fold selection.
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
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo))
from jevaffect.jev_temporal import JevTemporalOrdinal
from jevaffect.group_temporal_data import load_group_temporal_train_valid
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
    if not torch.cuda.is_available():raise RuntimeError('OOF teacher requires real local CUDA')
    output=repo/'runs/jev_temporal_oof_group'
    train,valid,identity=load_group_temporal_train_valid(repo/'data/feature_cache/simsv2_macbert_train_group_holdout_v1')
    if len(train)!=2182 or len(valid)!=540:raise ValueError('Group cohort count changed')
    ids=list(train.ids);parts=folds_by_source(ids)
    config={'source_sha256':identity['source_sha256'],
            'group_manifest_sha256':identity['group_manifest_sha256'],
            'train_ids_sha256':hashlib.sha256(json.dumps(ids,ensure_ascii=False).encode()).hexdigest(),
            'fold_seed':13,'fold_count':3,'teacher_epochs':8,
            'teacher_architecture':'JevTemporalOrdinal, hidden96, uniform gate, 8 physical-time windows',
            'teacher_lr':1e-3,'modality_dropout':0.15,
            'test_labels_used':False,'group_holdout_labels_used':False}
    output.mkdir(parents=True,exist_ok=True)
    config_path=output/'run_config.json'
    if config_path.is_file():
        if json.loads(config_path.read_text(encoding='utf-8'))!=config:
            raise ValueError('Existing OOF target run has changed data/config identity')
    else:_atomic_json(config_path,config)
    final_summary=output/'summary.json'
    if final_summary.is_file():
        completed=json.loads(final_summary.read_text(encoding='utf-8'))
        if (completed['cache_source_sha256']!=identity['source_sha256'] or
            sha(output/'utility_targets.npy')!=completed['target_sha256'] or
            not (output/'standardization.json').is_file()):
            raise ValueError('Completed OOF artifacts changed identity')
        print(json.dumps({'completed':True,'resumed_noop':True,
                          'train_rows':len(train),'target_valid_count':completed['target_valid_count']}),flush=True)
        train.base.close();valid.base.close();return
    source_groups={sid:sid.split('/')[0] for sid in ids}
    all_targets=np.full((len(train),8,2),np.nan,dtype=np.float32)
    teacher_info=[]
    for fold,held in enumerate(parts):
        held_set=set(held)
        included=[idx for idx in range(len(train)) if idx not in held_set]
        train_groups={source_groups[ids[idx]] for idx in included}
        held_groups={source_groups[ids[idx]] for idx in held}
        if train_groups & held_groups:raise AssertionError('OOF source group leakage')
        fold_meta_path=output/f'fold_{fold}_manifest.json'
        fold_target_path=output/f'fold_{fold}_targets.npy'
        path=output/f'teacher_fold_{fold}.pt'
        held_sha=hashlib.sha256(json.dumps(held,separators=(',',':')).encode()).hexdigest()
        if fold_meta_path.is_file():
            saved=json.loads(fold_meta_path.read_text(encoding='utf-8'))
            if (saved['held_indices_sha256']!=held_sha or saved['source_sha256']!=identity['source_sha256'] or
                sha(fold_target_path)!=saved['target_sha256'] or sha(path)!=saved['checkpoint_sha256']):
                raise ValueError('OOF fold recovery artifact changed identity')
            rows=np.load(fold_target_path,allow_pickle=False)
            if rows.shape!=(len(held),8,2):raise ValueError('Recovered OOF fold target shape changed')
            all_targets[np.asarray(held,dtype=np.int64)]=rows
            teacher_info.append(saved)
            print(json.dumps({'fold':fold,'recovered':True,'oof':len(held)}),flush=True)
            continue
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
        id_to_position={ids[index]:position for position,index in enumerate(held)}
        fold_targets=np.full((len(held),8,2),np.nan,dtype=np.float32)
        for batch in heldloader:
            target,mask=make_targets(batch,teacher,device,8)
            for sid,row in zip(batch['id'],target):
                fold_targets[id_to_position[sid]]=row
        if not np.isfinite(fold_targets).any():raise ValueError('Fold has no finite utility targets')
        all_targets[np.asarray(held,dtype=np.int64)]=fold_targets
        _atomic_torch_save(path,{'fold':fold,'seed':seed,'model':teacher.state_dict(),
                                 'cache_source_sha256':identity['source_sha256']})
        temporary=output/f'fold_{fold}_targets.tmp.npy'
        np.save(temporary,fold_targets,allow_pickle=False)
        temporary.replace(fold_target_path)
        fold_info={'fold':fold,'seed':seed,'teacher_train':len(included),
                   'heldout_target_rows':len(held),'held_indices_sha256':held_sha,
                   'source_sha256':identity['source_sha256'],
                   'train_source_groups':sorted(train_groups),'heldout_source_groups':sorted(held_groups),
                   'train_history':history,'wall_seconds':time.perf_counter()-began,
                   'checkpoint_sha256':sha(path),'target_sha256':sha(fold_target_path)}
        _atomic_json(fold_meta_path,fold_info)
        teacher_info.append(fold_info)
        print(json.dumps({'fold':fold,'train':len(included),'oof':len(held),
                          'wall_seconds':teacher_info[-1]['wall_seconds']}),flush=True)
    if not np.any(np.isfinite(all_targets)):raise ValueError('No valid utility targets generated')
    tmp_targets=output/'utility_targets.tmp.npy'
    np.save(tmp_targets,all_targets,allow_pickle=False)
    tmp_targets.replace(output/'utility_targets.npy')
    _atomic_json(output/'train_ids.json',ids)
    finite=all_targets[np.isfinite(all_targets)]
    summary={'source':'TRAIN-only source-video-disjoint CH-SIMS v2 group cohort','dataset':'CH-SIMS-v2.0',
             'cache_source_sha256':identity['source_sha256'],'train_rows':len(train),
             'group_manifest_sha256':identity['group_manifest_sha256'],
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
             'research_limit':'utility effect is from a teacher model; downstream benefit and faithfulness require independent tests'}
    scale=float(np.std(finite))
    if not np.isfinite(scale) or scale<1e-5:raise ValueError('OOF utility variation is too small')
    standardized=all_targets/scale
    standard_tmp=output/'utility_targets_standardized.tmp.npy'
    np.save(standard_tmp,standardized,allow_pickle=False)
    standard_tmp.replace(output/'utility_targets_standardized.npy')
    _atomic_json(output/'standardization.json',{
        'source_target_sha256':summary['target_sha256'],
        'source_scale_training_only_std':scale,
        'standardized_target_sha256':sha(output/'utility_targets_standardized.npy'),
        'finite_targets':int(len(finite)),
        'no_validation_or_test_labels':True,
        'purpose':'signed intervention effect rescaled for supervised gate loss'})
    _atomic_json(final_summary,summary)
    train.base.close();valid.base.close()
    print(json.dumps({k:v for k,v in summary.items() if k!='teacher_folds'}),flush=True)
if __name__=='__main__':main()
