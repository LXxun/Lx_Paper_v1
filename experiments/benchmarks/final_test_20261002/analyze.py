"""Locked six primary comparisons; no selection based on observed effects."""
import json,gzip,argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from guard import verify
S=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','test'],required=True);a=p.parse_args();verify(a.phase)
manifest=json.loads((S/('replay_manifest.json' if a.phase=='replay' else 'sample_manifest.json')).read_text())
root=S/('replay_explanations' if a.phase=='replay' else 'results');out=S/('replay_analysis' if a.phase=='replay' else 'analysis');out.mkdir(exist_ok=False)
summary=[];sensitivity=[]
for ds in ['MOSI','MOSEI']:
 rows_by_sample=defaultdict(list);videos={};descriptive=defaultdict(lambda:defaultdict(list))
 for v in ['main_only','main_pair','concat']:
  for seed in [2026,2027,2028]:
   folder=root/ds/v/f'seed{seed}';status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
   picked={};n=0
   with gzip.open(folder/'paired.jsonl.gz','rt') as f:
    for line in f:
     r=json.loads(line);n+=1;i=r['index'];videos[i]=r['video']
     k=(v,r['substitution'],r['reference'],r['target'],r['budget'],r['modality'],r['method'],r['operation'])
     # Average random repeats within seed, then seeds per segment: all have equal repetitions.
     for space,field in [('input','directional_drop' if r['operation']=='delete' else 'absolute_change'),('evidence','representation_directional_drop' if r['operation']=='delete' else 'representation_absolute_change')]:descriptive[k+(space,)][i].append(r[field])
     if r['modality']=='T' and r['budget']==.2 and r['method']!='random':
      key=(i,r['substitution'],r['reference'],r['target'],r['method'],r['operation']);assert key not in picked;picked[key]=r
   assert n==status['records']
   for i in manifest[ds]['indices']:
    if 'T' not in manifest[ds]['eligible_modalities'][str(i)]:continue
    for sub in ['mask_mean','unk_zero']:
     contrasts=[]
     for op in ['delete','retain']:
      b=picked[i,sub,'zero','intensity','group_occlusion_input',op];e=picked[i,sub,'zero','intensity','group_occlusion_evidence',op]
      field='representation_directional_drop' if op=='delete' else 'representation_absolute_change';sign=1 if op=='delete' else -1
      contrasts.append(sign*(e[field]-b[field]))
     pos=lambda ref,target:set(picked[i,sub,ref,target,'coalition','delete']['positions'])
     change_ref=int(pos('zero','intensity')!=pos('encoded','intensity'));change_target=int(pos('zero','intensity')!=pos('zero','raw_intensity'))
     rows_by_sample[i].append(contrasts+[change_ref-change_target,change_ref,change_target])
 for key,samples in descriptive.items():
  sensitivity.append(dict(dataset=ds,configuration=key,samples=len(samples),mean=float(np.mean([np.mean(x) for x in samples.values()]))))
 ids=sorted(rows_by_sample);assert all(len(rows_by_sample[i])==18 for i in ids)
 x=np.array([np.mean(rows_by_sample[i],axis=0) for i in ids]);vv=sorted({videos[i] for i in ids})
 sums=np.array([x[[videos[i]==v for i in ids]].sum(0) for v in vv]);counts=np.array([sum(videos[i]==v for i in ids) for v in vv])
 rng=np.random.default_rng({'MOSI':2026100201,'MOSEI':2026100202}[ds]);boot=[]
 for start in range(0,50000,500):
  idx=rng.integers(0,len(vv),size=(min(500,50000-start),len(vv)));boot.append(sums[idx].sum(1)/counts[idx].sum(1)[:,None])
 boot=np.concatenate(boot);names=['delete_matching','retain_matching','reference_vs_target']
 for j,name in enumerate(names):
  ci=np.quantile(boot[:,j],[.05/12,1-.05/12]);summary.append(dict(id=ds+'_'+name,mean=float(x[:,j].mean()),adjusted_ci=ci.tolist(),pointwise_ci=np.quantile(boot[:,j],[.025,.975]).tolist(),samples=len(ids),videos=len(vv),decision='supports_direction' if ci[0]>0 else 'opposite_direction' if ci[1]<0 else 'insufficient',reference_change_rate=float(x[:,3].mean()),target_change_rate=float(x[:,4].mean())))
assert len(summary)==6
(out/'primary_six.json').write_text(json.dumps(summary,indent=2));(out/'sensitivity_means.json').write_text(json.dumps(sensitivity,indent=2));(out/'status.json').write_text(json.dumps(dict(status='completed',phase=a.phase,test_inference=False,primary_count=6)))
print('ANALYSIS COMPLETE',a.phase)
