"""Compare complete replay rows (including selected spans) with historical validation."""
import json,gzip,math
from pathlib import Path
from datetime import datetime,timezone
from guard import digest
S=Path(__file__).resolve().parent;R=S.parent
manifest=json.loads((S/'replay_manifest.json').read_text());reports=[]
def compare(a,b):
 if isinstance(a,dict):
  assert set(a)==set(b),(set(a)^set(b));return max((compare(a[k],b[k]) for k in a),default=0.)
 if isinstance(a,list):
  assert len(a)==len(b);return max((compare(x,y) for x,y in zip(a,b)),default=0.)
 if isinstance(a,float):
  assert math.isfinite(a) and math.isfinite(b);err=abs(a-b);assert err<=1e-4+1e-5*abs(b),(a,b);return err
 assert a==b,(a,b);return 0.
for ds in ['MOSI','MOSEI']:
 for v in ['main_only','main_pair','concat']:
  for seed in [2026,2027,2028]:
   sub=Path(ds)/v/f'seed{seed}';out=S/'replay_explanations'/sub;old=R/'expanded_locked_validation_20261002/results'/sub;chosen=set(manifest[ds]['indices'])
   report=dict(dataset=ds,variant=v,seed=seed)
   for name,opener in [('paired.jsonl.gz',gzip.open),('games.jsonl',open)]:
    with opener(old/name,'rt') as f:prior=[x for line in f if (x:=json.loads(line))['index'] in chosen]
    with opener(out/name,'rt') as f:current=[json.loads(line) for line in f]
    report[name]=dict(rows=len(current),max_numeric_error=compare(current,prior))
   status=json.loads((out/'status.json').read_text());assert status['status']=='completed' and not status['test_evaluated']
   pred=json.loads((S/'replay_predictions'/sub/'status.json').read_text());assert pred['status']=='completed' and not pred['test_inference'];report['prediction']=pred
   reports.append(report)
result=dict(status='passed',models=len(reports),test_inference=False,reports=reports)
(S/'REPLAY_REPORT.json').write_text(json.dumps(result,indent=2));print('REPLAY PASSED',len(reports),flush=True)
