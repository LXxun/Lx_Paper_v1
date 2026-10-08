"""Read archived final-test records only; emit anonymous video sufficient statistics."""
import gzip,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
R=Path(sys.argv[1])  # Existing archive root; read-only.
manifest=json.loads((R/'sample_manifest.json').read_text());result={};hashes={}
for ds in ['MOSI','MOSEI']:
 rows=defaultdict(list);videos={}
 for head in ['main_only','main_pair','concat']:
  for seed in [2026,2027,2028]:
   folder=R/'results'/ds/head/f'seed{seed}';p=folder/'paired.jsonl.gz';status=json.loads((folder/'status.json').read_text());assert status['status']=='completed'
   hashes[str(p.relative_to(R))]=hashlib.sha256(p.read_bytes()).hexdigest()
   picked={};n=0
   with gzip.open(p,'rt') as f:
    for line in f:
     r=json.loads(line);n+=1
     if r['modality']=='T' and r['budget']==.2 and r['method'] in ['group_occlusion_input','group_occlusion_evidence','coalition']:
      key=(r['index'],r['substitution'],r['reference'],r['target'],r['method'],r['operation']);assert key not in picked;picked[key]=r;videos[r['index']]=r['video']
   assert n==status['records']
   for i in manifest[ds]['indices']:
    if 'T' not in manifest[ds]['eligible_modalities'][str(i)]:continue
    for sub in ['mask_mean','unk_zero']:
     vals=[]
     for op in ['delete','retain']:
      b=picked[i,sub,'zero','intensity','group_occlusion_input',op];e=picked[i,sub,'zero','intensity','group_occlusion_evidence',op];field='representation_directional_drop' if op=='delete' else 'representation_absolute_change'
      vals.append((1 if op=='delete' else -1)*(e[field]-b[field]))
     pos=lambda ref,target:set(picked[i,sub,ref,target,'coalition','delete']['positions'])
     cr=int(pos('zero','intensity')!=pos('encoded','intensity'));ct=int(pos('zero','intensity')!=pos('zero','raw_intensity'));rows[i].append(vals+[cr-ct,cr,ct])
   print(ds,head,seed,'read verified',file=sys.stderr,flush=True)
 ids=sorted(rows);assert all(len(rows[i])==18 for i in ids);x=np.array([np.mean(rows[i],axis=0) for i in ids]);vv=sorted(set(videos[i] for i in ids))
 result[ds]=[dict(group=f'g{j+1:04d}',n=sum(videos[i]==v for i in ids),sums=x[[videos[i]==v for i in ids]].sum(0).tolist()) for j,v in enumerate(vv)]
print(json.dumps(dict(family='original',columns=['delete_matching','retain_matching','reference_vs_target','reference_change','target_change'],datasets=result,raw_file_hashes=hashes)))
