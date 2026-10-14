"""Full CH-SIMS v2 five-class text-encoder fine-tuning, train/valid only."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('HF_HUB_OFFLINE','1')
import argparse,csv,hashlib,json,math,platform,random,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
from transformers import AutoModel,AutoTokenizer,__version__ as transformers_version
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from jevaffect.protocol import CLASSES,map_simsv2_label
from jevaffect.metrics import classification_metrics,negative_log_likelihood,top_label_ece_15
from jevaffect.training import _atomic_json,_atomic_jsonl,_atomic_torch_save

SEEDS=(13,42,2026)
CSV_SHA='be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150'
MODEL_SHA='876f584f15ebf14887dec17539c114bb99a032e96b9e72507a51c41e205337fc'
REVISION='3f076fdb1ab68d5b2880cb87a0886f315b8146f8'

def sha(path:Path)->str:
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(obj)->bytes:return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')

class TextRows(Dataset):
    def __init__(self,records,tokens):
        self.records=records;self.tokens=tokens
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        row=self.records[index]
        return {'id':row['id'],'label':torch.tensor(row['label'],dtype=torch.long),
                'input_ids':self.tokens['input_ids'][index],
                'attention_mask':self.tokens['attention_mask'][index]}

class FiveClassBERT(nn.Module):
    def __init__(self,model_dir:Path):
        super().__init__()
        self.encoder=AutoModel.from_pretrained(model_dir,local_files_only=True)
        if self.encoder.config.hidden_size!=768:raise ValueError('Pinned mBERT hidden width changed')
        self.dropout=nn.Dropout(0.1)
        self.head=nn.Linear(768,5)
    def forward(self,input_ids,attention_mask):
        out=self.encoder(input_ids=input_ids,attention_mask=attention_mask)
        return self.head(self.dropout(out.last_hidden_state[:,0,:]))

def parse_data(csv_path:Path,model_dir:Path):
    if sha(csv_path)!=CSV_SHA:raise ValueError('Official source CSV SHA mismatch')
    provenance=json.loads((model_dir/'provenance.json').read_text(encoding='utf-8'))
    if (provenance.get('verified') is not True or provenance.get('sha256')!=MODEL_SHA
        or provenance.get('revision')!=REVISION or sha(model_dir/'model.safetensors')!=MODEL_SHA):
        raise ValueError('Pinned pretrained mBERT has not been fully verified')
    groups={'train':[],'valid':[]}
    with csv_path.open(encoding='utf-8-sig',newline='') as handle:
        for row in csv.DictReader(handle):
            split=row.get('mode')
            if split not in groups:continue
            sample_id=row['video_id']+'/'+row['clip_id']
            text=row['text']
            if not text.strip():raise ValueError('Missing official train/valid transcript')
            groups[split].append({'id':sample_id,'text':text,
                                  'label':CLASSES.index(map_simsv2_label(float(row['label'])))})
    if len(groups['train'])!=2722 or len(groups['valid'])!=647:
        raise ValueError('Unexpected official train/validation sizes')
    for split,rows in groups.items():
        if len({row['id'] for row in rows})!=len(rows):raise ValueError('Duplicate '+split+' IDs')
    if set(x['id'] for x in groups['train']) & set(x['id'] for x in groups['valid']):
        raise ValueError('Train/valid clip ID overlap')
    tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    tokenized={}
    for split,rows in groups.items():
        tokenized[split]=tokenizer([row['text'] for row in rows],padding='max_length',
                                   truncation=True,max_length=64,return_tensors='pt')
    return groups,tokenized,provenance

def rng_state(generator):
    n=np.random.get_state()
    return {'python':random.getstate(),'numpy':(n[0],n[1].tolist(),n[2],n[3],n[4]),
            'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all(),
            'loader':generator.get_state()}
def restore_rng(state,generator):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch_cpu']);torch.cuda.set_rng_state_all(state['torch_cuda'])
    generator.set_state(state['loader'])

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ids=[];labels=[];prob=[]
    for batch in loader:
        logits=model(batch['input_ids'].to(device),batch['attention_mask'].to(device))
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation logits')
        ids.extend(batch['id']);labels.extend(batch['label'].tolist())
        prob.extend(logits.float().softmax(-1).cpu().tolist())
    y=np.asarray(labels,dtype=np.int64);p=np.asarray(prob,dtype=np.float64)
    if not np.isfinite(p).all():raise ValueError('Invalid validation probabilities')
    m=classification_metrics(y,p.argmax(-1))
    return {'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],
            'nll':negative_log_likelihood(y,p),'ece15':top_label_ece_15(y,p),
            'ids':ids,'labels':labels,'probabilities':prob}

def train_seed(seed:int,*,resume:bool=False,max_epochs_this_call:int|None=None):
    if seed not in SEEDS:raise ValueError('Unregistered random seed')
    device=torch.device('cuda')
    if not torch.cuda.is_available():raise RuntimeError('Real CUDA GPU required')
    model_dir=ROOT/'data/models/mbert';source_csv=ROOT/'data/simsv2/meta.csv'
    rows,tokens,provenance=parse_data(source_csv,model_dir)
    source={'csv_sha256':CSV_SHA,'model_sha256':MODEL_SHA,'model_revision':REVISION,
            'tokenizer_files_sha256':{x:sha(model_dir/x) for x in ('tokenizer.json','vocab.txt','tokenizer_config.json')},
            'split_ids_sha256':{s:hashlib.sha256(canonical([r['id'] for r in rows[s]])).hexdigest() for s in ('train','valid')}}
    config={'seed':seed,'max_epochs':12,'batch_size':16,'early_stop':3,
            'encoder_lr':2e-5,'head_lr':1e-3,'weight_decay':0.01,
            'dropout':0.1,'max_tokens':64,'warmup_fraction':0.1,
            'selection':'valid_accuracy_then_nll','split':'official author train/valid',
            'source':source,'test_scored':False}
    impl=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    identity=hashlib.sha256(canonical({'config':config,'implementation_sha256':impl,
                                       'environment':{'python':sys.version.split()[0],
                                                      'torch':torch.__version__,'transformers':transformers_version}})).hexdigest()
    root=ROOT/f'runs/simsv2_mbert_finetune/seed_{seed}'
    last=root/'last.pt';best=root/'best.pt'
    if root.exists() and not resume:raise FileExistsError('Run exists; use --resume')
    if resume and not last.is_file():raise FileNotFoundError('No checkpoint to resume')
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    generator=torch.Generator().manual_seed(seed)
    trainloader=DataLoader(TextRows(rows['train'],tokens['train']),batch_size=16,shuffle=True,
                           generator=generator,num_workers=0,pin_memory=True)
    validloader=DataLoader(TextRows(rows['valid'],tokens['valid']),batch_size=32,shuffle=False,num_workers=0)
    model=FiveClassBERT(model_dir).to(device)
    optimizer=torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':2e-5},
                                  {'params':model.head.parameters(),'lr':1e-3}],weight_decay=0.01)
    updates=12*math.ceil(len(rows['train'])/16);warmup=round(updates*0.1)
    def factor(step):
        if step<warmup:return float(step+1)/max(1,warmup)
        return max(0.0,float(updates-step)/max(1,updates-warmup))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    root.mkdir(parents=True,exist_ok=True)
    history=[];resources=[];start=1;best_acc=-1.;best_nll=float('inf');best_epoch=0;stale=0
    if resume:
        state=torch.load(last,map_location='cpu',weights_only=True)
        if state.get('identity_sha256')!=identity:raise ValueError('Config/code/source/runtime resume identity mismatch')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng_state'],generator)
        start=state['epoch']+1;history=state['history'];resources=state['resources']
        best_acc=state['best_accuracy'];best_nll=state['best_nll'];best_epoch=state['best_epoch'];stale=state['stale']
        if state['completed']:
            if not best.is_file():raise ValueError('Completed run lacks selected weights')
            return json.loads((root/'summary.json').read_text(encoding='utf-8'))
    else:
        _atomic_json(root/'config.json',config)
        _atomic_json(root/'run_manifest.json',{'identity_sha256':identity,'implementation_sha256':impl,
                        'source':source,'class_order':list(CLASSES),'test_scored':False})
        _atomic_json(root/'environment.json',{'python':sys.version.split()[0],
                     'platform':platform.platform(),'torch':torch.__version__,'numpy':np.__version__,
                     'transformers':transformers_version,'gpu':torch.cuda.get_device_name(0),
                     'cuda':torch.version.cuda})
    this_call=0
    for epoch in range(start,13):
        if max_epochs_this_call is not None and this_call>=max_epochs_this_call:break
        model.train();torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize();begin=time.perf_counter()
        total=0;correct=0
        for batch in trainloader:
            ids=batch['input_ids'].to(device,non_blocking=True)
            mask=batch['attention_mask'].to(device,non_blocking=True)
            label=batch['label'].to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits=model(ids,mask)
            loss=F.cross_entropy(logits,label)
            if not torch.isfinite(loss):raise ValueError('Nonfinite training CE')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step();scheduler.step()
            total+=float(loss.detach())*len(label)
            correct+=int((logits.argmax(-1)==label).sum())
        valid=evaluate(model,validloader,device);torch.cuda.synchronize()
        improved=valid['accuracy']>best_acc or (valid['accuracy']==best_acc and valid['nll']<best_nll)
        if improved:
            best_acc=valid['accuracy'];best_nll=valid['nll'];best_epoch=epoch;stale=0
            _atomic_torch_save(best,{'identity_sha256':identity,'epoch':epoch,
                                     'encoder':model.encoder.state_dict(),'head':model.head.state_dict(),
                                     'class_order':list(CLASSES),'source':source})
            _atomic_jsonl(root/'validation_predictions.jsonl',[
                {'ID':sid,'label':int(label),'probabilities':probs,
                 'argmax':CLASSES[int(np.argmax(probs))],'class_order':list(CLASSES),'split':'valid'}
                for sid,label,probs in zip(valid['ids'],valid['labels'],valid['probabilities'])])
        else:stale+=1
        row={'epoch':epoch,'train_loss':total/len(rows['train']),'train_accuracy':correct/len(rows['train']),
             'valid_accuracy':valid['accuracy'],'valid_macro_f1':valid['macro_f1'],
             'valid_nll':valid['nll'],'valid_ece15':valid['ece15'],
             'best_valid_accuracy':best_acc,'best_epoch':best_epoch,
             'encoder_lr':optimizer.param_groups[0]['lr'],'head_lr':optimizer.param_groups[1]['lr']}
        history.append(row);resources.append({'epoch':epoch,'train_valid_seconds':time.perf_counter()-begin,
                                               'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device)})
        _atomic_jsonl(root/'metrics.jsonl',history);_atomic_jsonl(root/'resources.jsonl',resources)
        completed=stale>=3 or epoch==12
        _atomic_torch_save(last,{'identity_sha256':identity,'epoch':epoch,'model':model.state_dict(),
                                  'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                                  'rng_state':rng_state(generator),'history':history,'resources':resources,
                                  'best_accuracy':best_acc,'best_nll':best_nll,'best_epoch':best_epoch,
                                  'stale':stale,'completed':completed})
        print(json.dumps(row),flush=True);this_call+=1
        if completed:break
    result={'seed':seed,'best_epoch':best_epoch,'best_accuracy':best_acc,
            'best_nll':best_nll,'last_epoch':history[-1]['epoch'] if history else 0,
            'completed':stale>=3 or len(history)>=12,'scope':'official CH-SIMS v2 train/valid only'}
    _atomic_json(root/'summary.json',result)
    return result
def main(argv:list[str]|None=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-epochs-this-call',type=int)
    args=parser.parse_args(argv)
    if args.resume and args.seed is None:raise ValueError('Resume requires explicit seed')
    selected=(args.seed,) if args.seed else SEEDS
    results=[train_seed(seed,resume=args.resume,max_epochs_this_call=args.max_epochs_this_call) for seed in selected]
    print(json.dumps(results,indent=2))
if __name__=='__main__':main()
