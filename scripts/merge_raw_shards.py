"""Verify private Kaggle raw-feature shard archives and form a train/valid bundle.

The input must be the completed JevAffect notebook output. All record arrays
remain private data; this command never publishes or evaluates the test split.
"""
from __future__ import annotations
import argparse,csv,hashlib,io,json,re,zipfile
from pathlib import Path
from typing import Any

META_SHA='be7e482640e7ee24d2070033e1e2aef61ce183199286a8a18b0d6247556bf150'
CLASSES=('SNEG','WNEG','NEUT','WPOS','SPOS')
RECORD_PATH=re.compile(r'records/[0-9a-f]{64}\.npz\Z')

def sha(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def file_sha(path:Path)->str:
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def canonical_json(value:Any)->bytes:
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
def read_json(path:Path)->dict:
    out=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(out,dict):raise ValueError('JSON object required: '+str(path))
    return out
def records_jsonl(data:bytes)->list[dict]:
    return [json.loads(line) for line in data.decode('utf-8').splitlines() if line.strip()]
def write_json(path:Path,value:Any)->None:
    path.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')

def merge(source:Path,official_csv:Path,destination:Path,*,expected_count:int=3369)->dict:
    source=source.resolve();destination=destination.resolve()
    if destination.exists():raise FileExistsError('Destination already exists: '+str(destination))
    run=read_json(source/'run_manifest.json')
    if run.get('status')!='complete' or run.get('progress',{}).get('completed_records')!=expected_count:
        raise ValueError('Raw source is not a complete train/valid run')
    ident=run['identity']
    if ident.get('source_meta_csv_sha256')!=META_SHA or ident.get('encoded_splits')!=['train','valid']:
        raise ValueError('Unexpected official CSV or encoded split identity')
    if ident.get('split_counts')!={'train':2722,'valid':647,'test':1034} or ident.get('test_inputs_created')!=0:
        raise ValueError('Unexpected split counts or test inputs')
    if file_sha(official_csv)!=META_SHA:raise ValueError('Official CSV SHA mismatch')
    inputs_file=source/'input_records.jsonl'
    if file_sha(inputs_file)!=ident['input_manifest_sha256']:raise ValueError('Source input manifest SHA mismatch')
    inputs=records_jsonl(inputs_file.read_bytes())
    if len(inputs)!=expected_count or len({x['id'] for x in inputs})!=expected_count:
        raise ValueError('Source input records missing or duplicate')
    inputs_by_id={x['id']:x for x in inputs}
    for row in inputs:
        if row['split'] not in ('train','valid') or not isinstance(row.get('transcript'),str):
            raise ValueError('Source input contains invalid split/transcript')
    official={}
    with official_csv.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            if row['mode'] in ('train','valid'):
                sid=row['video_id']+'/'+row['clip_id']
                if sid in official:raise ValueError('Duplicate official train/valid ID')
                official[sid]=row
    if set(official)!={x['id'] for x in inputs}:raise ValueError('Full train/valid cohort ID mismatch')
    if {split:sum(x['split']==split for x in inputs) for split in ('train','valid')}!={'train':2722,'valid':647}:
        raise ValueError('Full input split counts mismatch')
    for row in inputs:
        ref=official[row['id']]
        if row['split']!=ref['mode'] or row['transcript']!=ref['text']:
            raise ValueError('Input transcript/split differs from official CSV: '+row['id'])
        if row['transcript_sha256']!=sha(ref['text'].encode('utf-8')):
            raise ValueError('Input transcript SHA mismatch: '+row['id'])
    archives=read_json(source/'shard_archives.json')
    listed=archives.get('archives')
    expected_shards=(expected_count+63)//64
    if archives.get('complete_archive_count')!=expected_shards or not isinstance(listed,list) or len(listed)!=expected_shards:
        raise ValueError('Expected all complete 64-record shards')
    provenance=run['encoder_provenance']
    if provenance.get('local_files_only') is not True or provenance.get('dimensions')!={'text':768,'audio':768,'vision':384}:
        raise ValueError('Encoder provenance does not describe the pinned common backbone')
    if not all(v.get('verified') is True for v in provenance.get('encoders',{}).values()):
        raise ValueError('Unverified encoder in global provenance')
    destination.mkdir(parents=True)
    (destination/'records').mkdir()
    merged=[];source_archives=[]
    for index,entry in enumerate(listed):
        if entry.get('shard_number')!=index:raise ValueError('Shard numbers not contiguous')
        rel=entry.get('archive_path')
        if not isinstance(rel,str) or rel!=f'shards/shard_{index:05d}.zip':raise ValueError('Unsafe archive path')
        path=source/rel
        if not path.is_file() or path.stat().st_size!=entry['archive_size_bytes'] or file_sha(path)!=entry['archive_sha256']:
            raise ValueError('Archive size/SHA mismatch: '+rel)
        side=source/(rel+'.sha256')
        expected_side=entry['archive_sha256']+'  '+path.name+'\n'
        if not side.is_file() or side.read_text(encoding='ascii')!=expected_side:
            raise ValueError('Archive SHA sidecar mismatch: '+rel)
        with zipfile.ZipFile(path) as zip_file:
            if zip_file.testzip() is not None:raise ValueError('Archive CRC mismatch: '+rel)
            internal=read_json_from_bytes(zip_file.read('archive_manifest.json'))
            index_bytes=zip_file.read('index.jsonl');rows=records_jsonl(index_bytes)
            if (internal.get('index_sha256')!=sha(index_bytes)
                or internal.get('source_meta_csv_sha256')!=META_SHA
                or internal.get('input_manifest_sha256')!=ident['input_manifest_sha256']
                or internal.get('record_start_index')!=index*64
                or internal.get('record_count')!=len(rows)
                or len(rows)!=entry['record_count']):
                raise ValueError('Archive internal identity mismatch: '+rel)
            if internal.get('records')!=[
                {key:row[key] for key in ('id','split','artifact','artifact_sha256','source_video_sha256')}
                for row in rows]:
                raise ValueError('Archive index differs from record manifest: '+rel)
            for offset,row in enumerate(rows):
                expected=inputs[index*64+offset]
                sid=row['id'];name=row['artifact']
                if sid!=expected['id'] or row['split']!=expected['split'] or row['transcript_sha256']!=expected['transcript_sha256']:
                    raise ValueError('Archive record order/split/transcript mismatch: '+sid)
                if not isinstance(name,str) or not RECORD_PATH.fullmatch(name) or name not in zip_file.namelist():
                    raise ValueError('Unsafe/missing record path: '+str(name))
                raw=zip_file.read(name)
                if sha(raw)!=row['artifact_sha256']:raise ValueError('Record SHA mismatch: '+sid)
                target=destination/name
                if target.exists():raise ValueError('Duplicate extracted record')
                target.write_bytes(raw)
                details=dict(row['metadata'])
                decoder=details.get('decode',{})
                if (details.get('fixture') is not False or details.get('model_results') is not True
                    or decoder.get('source_sha256')!=row['source_video_sha256']):
                    raise ValueError('Source record is not a verified model/media result: '+sid)
                details.update({'id':sid,'encoder_provenance':provenance,
                                'source_video_sha256':row['source_video_sha256'],
                                'transcript':expected['transcript'],'prediction':None})
                merged.append({'id':sid,'artifact':name,'artifact_sha256':row['artifact_sha256'],'metadata':details})
        source_archives.append({'archive':rel,'sha256':entry['archive_sha256'],'record_count':len(rows)})
    if len(merged)!=expected_count:raise ValueError('Missing merged records')
    index_bytes=b''.join(canonical_json(row)+b'\n' for row in merged)
    (destination/'index.jsonl').write_bytes(index_bytes)
    meta={'schema_version':1,'cache_kind':'raw_video_feature_records','labels_included':False,
          'canonical_ids_and_splits_joined':False,'fixture_results':False,'all_records_are_fixtures':False,
          'model_results':True,'sample_count':expected_count,'encoder_provenance':provenance,
          'input_manifest_sha256':ident['input_manifest_sha256'],
          'files':{'index.jsonl':sha(index_bytes),'records':[x['artifact'] for x in merged]},
          'source_archives':source_archives,'cohort':'full_official_train_valid'}
    write_json(destination/'meta.json',meta)
    (destination/'meta.sha256').write_text(file_sha(destination/'meta.json')+'  meta.json\n',encoding='ascii')
    manifest=[]
    for row in merged:
        sid=row['id'];source_row=official[sid]
        manifest.append({'ID':sid,'split':source_row['mode'],'dataset':'CH-SIMS-v2.0','language':'zh',
                         'transcript_condition':'author_provided','text':source_row['text'],
                         'raw_label':source_row['label'],'source_video':source_row['video_id'],
                         'sha256':row['metadata']['source_video_sha256'],'source_csv_sha256':META_SHA,
                         'video_path':inputs_by_id[sid]['video_path']})
    manifest_bytes=b''.join(canonical_json(row)+b'\n' for row in manifest)
    # Raw bundle remains label-free. This explicit join manifest lives beside
    # it, and the cache builder later verifies every row against official CSV.
    join_manifest=destination.parent/(destination.name+'_train_valid_manifest.jsonl')
    join_manifest.write_bytes(manifest_bytes)
    result={'records':expected_count,'split_counts':{'train':2722,'valid':647},'test_encoded':False,
            'source_archives':len(source_archives),'official_csv_sha256':META_SHA,
            'raw_bundle_meta_sha256':file_sha(destination/'meta.json'),
            'train_valid_manifest_path':str(join_manifest),
            'train_valid_manifest_sha256':sha(manifest_bytes),'ready_for_cache_builder':True}
    write_json(destination/'merge_summary.json',result)
    return result

def read_json_from_bytes(data:bytes)->dict:
    value=json.loads(data.decode('utf-8'))
    if not isinstance(value,dict):raise ValueError('Archive manifest must be a JSON object')
    return value

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--official-csv',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(merge(args.source,args.official_csv,args.output),indent=2))
if __name__=='__main__':main()
