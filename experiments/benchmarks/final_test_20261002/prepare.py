import os
"""Prepare IDs/cache and audit assets. Never constructs a model or runs inference."""
import sys,json,hashlib,platform,subprocess,ast
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,torch,transformers
S=Path(__file__).resolve().parent;R=S.parent;sys.path.insert(0,str(R))
from prepare_standard import Restricted,digest,HASHES
expected=json.loads((S/'expected_artifacts.json').read_text())
actual={}
for rel,h in expected['artifacts'].items():
 actual[str(R/rel)]=digest(R/rel);assert actual[str(R/rel)]==h,rel
print('18 checkpoints and historical dependencies verified',flush=True)
for folder in ['source','compact_stage/source']:
 for f in (R/folder).rglob('*.py'):actual[str(f)]=digest(f)
for f in [R/'prepare_standard.py',R/'compact_stage/explain.py',R/'run_controlled.py',R/'run_compact_stage.py',R/'run_finetuned.py']:actual[str(f)]=digest(f)
bert=Path(os.environ.get('BERT_MODEL_DIR',str(R/'models/bert-base-uncased')))
for f in bert.iterdir():
 if f.is_file():actual[str(f)]=digest(f)
for f in (S/'protocol.json',S/'FINAL_TEST_PROTOCOL.md',S/'PROTOCOL_LOCK.json',S/'checkpoint_manifest.json',S/'expected_artifacts.json'):actual[str(f)]=digest(f)
manifest={};replay={};audit={}
for ds in ['MOSI','MOSEI']:
 path=R/'data'/ds/'aligned_50.pkl';h=digest(path);assert h==HASHES[ds]['aligned_50.pkl'];actual[str(path)]=h
 with path.open('rb') as f:raw=Restricted(f).load()
 ids={split:np.asarray(raw[split]['id']).astype(str).reshape(-1) for split in ['train','valid','test']}
 videos={sp:{v.split('$_$')[0] for v in ii} for sp,ii in ids.items()}
 for sp,ii in ids.items():assert len(set(ii))==len(ii)
 overlaps={}
 for x,y in [('train','valid'),('train','test'),('valid','test')]:
  overlaps[x+'_'+y]={'ids':sorted(set(ids[x])&set(ids[y])),'videos':sorted(videos[x]&videos[y])};assert not overlaps[x+'_'+y]['ids'] and not overlaps[x+'_'+y]['videos']
 foldpath=R/'repos/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets'/('CMU_'+ds)/('cmu_'+ds.lower()+'_std_folds.py')
 folds={n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(foldpath.read_text()).body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)}
 exceptions={}
 for sp in ids:
  outside=videos[sp]-set(folds['standard_'+sp+'_fold']);allowed={'-NFrJFQijFE'} if ds=='MOSEI' and sp=='train' else set();assert outside<=allowed
  exceptions[sp]=sorted(outside)
 actual[str(foldpath)]=digest(foldpath)
 d=raw['test'];tokens=np.asarray(d['text_bert'],np.int64);audio=np.asarray(d['audio'],np.float32);vision=np.asarray(d['vision'],np.float32)
 assert len(tokens)=={'MOSI':686,'MOSEI':4659}[ds] and tokens.shape[1:]==(3,50)
 assert audio.shape[:2]==vision.shape[:2]==(len(tokens),50)
 assert np.isfinite(audio).all() and np.isfinite(vision).all()
 mask=np.stack([(tokens[:,1]==1)&~np.isin(tokens[:,0],[0,101,102]),np.any(audio!=0,axis=-1),np.any(vision!=0,axis=-1)],-1)
 out=S/'cache'/ds;out.mkdir(parents=True,exist_ok=False)
 # Labels are copied for future evaluation, never summarized or used in selection.
 np.savez(out/'test.npz',tokens=tokens,audio=audio,vision=vision,mask=mask,ids=ids['test'],yr=np.asarray(d['regression_labels'],np.float32).reshape(-1))
 cp=out/'test.npz';actual[str(cp)]=digest(cp)
 hh=lambda x:hashlib.sha256(x.encode()).hexdigest()
 if ds=='MOSI':indices=list(range(len(tokens)))
 else:
  vv=sorted(videos['test'],key=lambda v:(hh('final-test-v1:video:'+v),v))[:300]
  indices=[min((i for i,x in enumerate(ids['test']) if x.split('$_$')[0]==v),key=lambda i:(hh('final-test-v1:sample:'+ids['test'][i]),ids['test'][i],i)) for v in vv]
 ck={v:{str(seed):expected['artifacts'][f'compact_stage/runs/{ds}/tei_mlp/{v}/seed{seed}/best.pt'] for seed in [2026,2027,2028]} for v in ['main_only','main_pair','concat']}
 rows=[dict(index=i,id=str(ids['test'][i]),video=str(ids['test'][i]).split('$_$')[0],eligible_modalities=['TAV'[m] for m in range(3) if mask[i,:,m].any()]) for i in range(len(tokens))]
 manifest[ds]=dict(cache_path=str(cp),cache_sha256=digest(cp),indices=indices,checkpoints=ck,all_prediction_samples=rows,eligible_modalities={str(i):rows[i]['eligible_modalities'] for i in indices})
 vp=R/'cache'/ds/'valid.npz';actual[str(vp)]=digest(vp)
 with np.load(vp) as v:
  old=json.loads((R/'expanded_locked_validation_20261002/sample_manifest.json').read_text())[ds]
  candidates=old['indices'];chosen=[candidates[0]]
  zero=[i for i in candidates if not v['mask'][i,:,2].any()]
  second=zero[0] if zero else candidates[-1]
  if second not in chosen:chosen.append(second)
  replay[ds]=dict(cache_path=str(vp),cache_sha256=digest(vp),indices=chosen,checkpoints=ck,eligible_modalities={str(i):['TAV'[m] for m in range(3) if v['mask'][i,:,m].any()] for i in chosen})
 audit[ds]=dict(splits={sp:dict(samples=len(ids[sp]),videos=len(videos[sp])) for sp in ids},overlaps=overlaps,sdk_unlisted=exceptions,explanation_samples=len(indices),explanation_videos=len({rows[i]['video'] for i in indices}),eligible_segments={m:sum(m in rows[i]['eligible_modalities'] for i in indices) for m in 'TAV'},eligible_videos={m:len({rows[i]['video'] for i in indices if m in rows[i]['eligible_modalities']}) for m in 'TAV'},replay_indices=chosen)
 print(ds,json.dumps(audit[ds]),flush=True)
 del raw
for name,obj in [('sample_manifest.json',manifest),('replay_manifest.json',replay),('DATA_AUDIT.json',audit)]:
 (S/name).write_text(json.dumps(obj,indent=2));actual[str(S/name)]=digest(S/name)
env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,gpus=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],text=True),packages=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
(S/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2));actual[str(S/'ENVIRONMENT.json')]=digest(S/'ENVIRONMENT.json')
(S/'ASSET_LOCK.json').write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),files=actual,test_inference=False),indent=2))
print('PREPARATION COMPLETE: no inference',flush=True)
