"""Local reconstruction of locked validation contrasts; requires complete raw results."""
import json,gzip,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
P=Path(__file__).resolve().parents[1]/'results/mult_space_v1'
done=json.loads((P/'results/COMPLETE.json').read_text());assert done['status']=='passed' and done['samples']==1587
samples=json.loads((P/'samples.json').read_text());groups=defaultdict(list);seen=set();rows_count=0
for filename,h in done['files'].items():
 path=P/'results'/filename;assert hashlib.sha256(path.read_bytes()).hexdigest()==h
 with gzip.open(path,'rt',encoding='utf-8') as f:
  for line in f:
   r=json.loads(line);ds=r['dataset'];sid=r['id'];seed=r['seed'];key=(ds,sid,seed);assert key not in seen;seen.add(key);rows_count+=1
   source=next(q for q in samples[ds] if q['id']==sid);assert source['units']==r['units'] and source['video']==r['video']
   matrix={tuple(c[k] for k in ['substitution','modality','ranking','evaluation','operation']):c for c in r['matrix']}
   assert len(matrix)==sum(bool(u) for u in r['units'])*16
   for sub in ['mask_mean','unk_zero']:
    for m,units in zip('TAV',r['units']):
     if not units:continue
     for operation in ['delete','retain']:
      for space in ['input','representation']:
       cells=[matrix[(sub,m,rank,space,operation)] for rank in ['Oi','Oe']]
       val=[]
       for c in cells:
        v=r['direction']*(r['original']-c['value'])*(-1 if operation=='retain' else 1);assert abs(v-c['utility'])<1e-12;val.append(v)
       contrast=val[1]-val[0] if space=='representation' else val[0]-val[1]
       groups[(ds,m,operation,space,sid,r['video'])].append((seed,sub,contrast))
assert rows_count==1587
assert seen=={(ds,q['id'],seed) for ds,v in samples.items() for q in v for seed in [2026,2027,2028]}
video=defaultdict(lambda:defaultdict(list));details=[]
for (ds,m,op,space,sid,vid),values in groups.items():
 assert {(s,t) for s,t,_ in values}=={(s,t) for s in [2026,2027,2028] for t in ['mask_mean','unk_zero']}
 sample_mean=float(np.mean([x[2] for x in values]));video[(ds,m,op,space)][vid].append(sample_mean)
 for seed,sub,value in values:details.append({'dataset':ds,'modality':m,'operation':op,'space':space,'id':sid,'video':vid,'seed':seed,'substitution':sub,'difference':value})
summary=[]
for (ds,m,op,space),v in sorted(video.items()):
 means=np.array([np.mean(v[k]) for k in sorted(v)]);rng=np.random.default_rng(20261005);boot=[]
 for lo in range(0,20000,1000):boot.extend(means[rng.integers(len(means),size=(min(1000,20000-lo),len(means)))].mean(1).tolist())
 primary=m=='T' and space=='representation'
 row=dict(dataset=ds,modality=m,operation=op,space=space,primary=primary,mean=float(means.mean()),samples=sum(len(x) for x in v.values()),videos=len(v),ci95=np.quantile(boot,[.025,.975]).tolist())
 if primary:row['ci98_75_bonferroni4']=np.quantile(boot,[.00625,.99375]).tolist()
 summary.append(row)
dump={'protocol':'MULT_SPACE_VALIDATION_V1','n_records':rows_count,'summary':summary,'limitations':['validation only','shared BERT family','shallow prefusion representation','MOSI only 10 independent video clusters']}
(P/'ANALYSIS.json').write_text(json.dumps(dump,indent=2));(P/'PAIRED_DIFFERENCES.json').write_text(json.dumps(details));print(json.dumps([r for r in summary if r['primary']],indent=2))
