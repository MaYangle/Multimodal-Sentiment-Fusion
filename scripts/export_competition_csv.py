"""Turn complete raw-video predictions into the competition's exact ID,Label CSV.

This finalizes predictions; it does not run a model or use test labels. Every
MP4 in Test_Data must have a prediction, including low-confidence items.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from jevaffect.protocol import export_predictions

def main(argv:list[str]|None=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--test-data',type=Path,required=True)
    p.add_argument('--predictions-jsonl',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,default=Path('Test_Results'))
    p.add_argument('--confidence-threshold',type=float,default=0.6)
    args=p.parse_args(argv)
    if not args.test_data.is_dir():raise FileNotFoundError(args.test_data)
    videos=sorted(path for path in args.test_data.rglob('*') if path.is_file() and path.suffix.lower()=='.mp4')
    if not videos:raise ValueError('Test_Data has no MP4 videos')
    ids=[path.name for path in videos]
    if len(set(ids))!=len(ids):raise ValueError('Two Test_Data videos have the same filename/competition ID')
    records=[]
    with args.predictions_jsonl.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():records.append(json.loads(line))
    if not records:raise ValueError('No prediction records supplied')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    result=export_predictions(records,ids,args.output_dir/'label_prediction.csv',
                              args.output_dir/'low_confidence.jsonl',
                              low_confidence_threshold=args.confidence_threshold)
    print(json.dumps({**result,'test_video_count':len(videos),'csv':str(args.output_dir/'label_prediction.csv'),
                      'all_low_confidence_rows_retained_in_csv':True,'test_labels_read':False}))
    return 0
if __name__=='__main__':raise SystemExit(main())
