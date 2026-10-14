"""Load the project's verified raw-video train/valid cache with its time masks."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from jevaffect.cmc_raw_cache import load_raw_feature_pilot_train_valid

class TemporalRawSplit(Dataset):
    def __init__(self,verified_split,cache_dir:Path,split:str):
        if split not in ('train','valid'):raise ValueError('Only train/valid may be loaded')
        self.base=verified_split;root=Path(cache_dir)/'splits'/split
        self.mask={m:np.load(root/(m+'_mask.npy'),mmap_mode='r',allow_pickle=False)
                   for m in ('text','audio','vision')}
        self.times={m:np.load(root/(m+'_times.npy'),mmap_mode='r',allow_pickle=False)
                    for m in ('audio','vision')}
        for m in self.mask:
            # Text is stored separately by CMC's contextual-text adapter.
            expected=self.base.text.shape[:2] if m=='text' else self.base.features[m].shape[:2]
            if self.mask[m].dtype!=np.bool_ or self.mask[m].shape!=expected:
                raise ValueError('Mask differs from verified feature shape: '+m)
        for m in self.times:
            if self.times[m].shape!=self.mask[m].shape or not np.isfinite(self.times[m]).all():
                raise ValueError('Timestamp shape/non-finiteness mismatch: '+m)
        self.ids=self.base.ids
    def __len__(self):return len(self.base)
    def __getitem__(self,index:int):
        row=self.base[index]
        for m in self.mask:
            row[m+'_mask']=torch.from_numpy(np.array(self.mask[m][index],dtype=np.bool_,copy=True))
        for m in self.times:
            row[m+'_times']=torch.from_numpy(np.array(self.times[m][index],dtype=np.float32,copy=True))
        return row

def load_temporal_train_valid(cache_dir:str|Path):
    root=Path(cache_dir)
    train,valid,identity=load_raw_feature_pilot_train_valid(root)
    return TemporalRawSplit(train,root,'train'),TemporalRawSplit(valid,root,'valid'),identity
