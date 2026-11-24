"""Audit first-trigger acquisition stamps against labeled trial starts.
This checks logged snapshots; it does not certify every frame or RGB/depth alignment.
Defaults to the preserved historical 84-trial input snapshot, not edited live files.
"""
from pathlib import Path
import argparse
import collections
import hashlib
import json

root=Path(__file__).resolve().parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input-dir',type=Path,default=root/'documents/ieee_semantic_navigation/data')
parser.add_argument('--output',type=Path,default=Path.home()/'hexapod_results/yolo_world_summary/historical_first_trigger_audit.json')
args=parser.parse_args()
rows=[];inputs=[]
for path in sorted(args.input_dir.glob('*.jsonl')):
    raw=path.read_bytes();inputs.append({'file':str(path),'sha256':hashlib.sha256(raw).hexdigest()})
    groups=collections.defaultdict(list)
    for line in raw.decode().splitlines():
        if line.strip():
            e=json.loads(line);groups[e['session_id'],e['trial_id']].append(e)
    for (session,trial),events in groups.items():
        starts=[e for e in events if e['status']=='trial_started' and e.get('scenario_id')]
        detections=[e for e in events if e['status']=='detected']
        if len(starts)!=1 or not detections:continue
        start=starts[0];d=detections[0]
        row={'pipeline':start['pipeline'],'scenario_id':start['scenario_id'],
             'trial_id':trial,'start_sim_s':start['sim_time'],'trigger_sim_s':d['sim_time']}
        for name in ('rgb','depth'):
            stamp=d.get(name+'_stamp_s');row[name+'_stamp_s']=stamp
            row[name+'_post_reset']=(stamp>start['sim_time']) if stamp is not None else None
        rows.append(row)
result={'scope':'First-trigger RGB/depth header stamps in the supplied snapshot; not all frames',
        'inputs':inputs,'trials_checked':len(rows),
        'trials_with_pre_reset_or_equal_stamp':sum(r['rgb_post_reset'] is False or r['depth_post_reset'] is False for r in rows),
        'trials_with_missing_stamp':sum(r['rgb_post_reset'] is None or r['depth_post_reset'] is None for r in rows),
        'trials':rows}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print({k:v for k,v in result.items() if k not in ('inputs','trials')})
