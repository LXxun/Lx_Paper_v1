from pathlib import Path
import json,gzip,hashlib,sys
import numpy as np
from collections import defaultdict
out=Path(sys.argv[3]);out.mkdir(exist_ok=False)
D=Path(sys.argv[1])
done=json.loads((D/'results/COMPLETE.json').read_text());samples=json.loads((D/'samples.json').read_text());records={}
for name,h in done['files'].items():
 p=D/'results'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==h
 with gzip.open(p,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);matrix={tuple(q[k] for k in ['substitution','ranking','evaluation','operation']):q['utility'] for q in r['matrix']};assert len(matrix)==16
   for q in r['matrix']:
    u=-abs(r['base']-q['after']) if q['operation']=='retain' else r['direction']*(r['base']-q['after']);assert abs(u-q['utility'])<1e-12
   records[r['dataset'],r['id'],r['seed']]=matrix
result={}
for ds in ['MOSI','MOSEI']:
 rr=samples[ds]['rows'];ids=[q['id'] for q in rr];vids=[q['video'] for q in rr];vs=sorted(set(vids));mm={}
 for rank in ['Oi','Oe']:
  for op in ['delete','retain']:
   a=np.array([[records[ds,sid,seed][sub,rank,'evidence',op] for seed in [2026,2027,2028] for sub in ['MASK','UNK']] for sid in ids]);mm[rank,op]=a.mean(1)
 x=np.stack([mm['Oe',op]-mm['Oi',op] for op in ['delete','retain']],axis=1)
 result[ds]=[dict(group=f'g{j+1:04d}',n=vids.count(v),sums=x[np.array(vids)==v].sum(0).tolist()) for j,v in enumerate(vs)]
(out/'recurrent.json').write_text(json.dumps(dict(family='recurrent',columns=['delete_matching','retain_matching'],datasets=result),indent=2))
D=Path(sys.argv[2]);details=json.loads((D/'PAIRED_DIFFERENCES.json').read_text());groups=defaultdict(list)
for q in details:
 if q['modality']=='T' and q['space']=='representation':groups[q['dataset'],q['video'],q['id'],q['operation']].append(q['difference'])
video=defaultdict(list)
for (ds,vid,sid,op),vv in groups.items():
 assert len(vv)==6;video[ds,vid,op].append(float(np.mean(vv)))
result={}
for ds in ['MOSI','MOSEI']:
 vs=sorted({v for d,v,o in video if d==ds});result[ds]=[dict(group=f'g{j+1:04d}',n=len(video[ds,v,'delete']),sums=[float(np.sum(video[ds,v,op])) for op in ['delete','retain']]) for j,v in enumerate(vs)]
(out/'mult.json').write_text(json.dumps(dict(family='mult',columns=['delete_matching','retain_matching'],datasets=result),indent=2))
print('Exported anonymous recurrent and MulT video sums/counts')
