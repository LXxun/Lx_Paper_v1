"""Recurrent extension. prepare metadata -> validation replay -> seal -> test."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,json,time,math,gzip,traceback
from pathlib import Path
import numpy as np,torch
import run_recurrent as r
from transformers import BertTokenizerFast
ROOT=r.ROOT;BASE=r.BASE;OUT=ROOT/'explanation_fp32';OLD=BASE/'final_test_20261002'
JOBS=[(d,s) for d in ['MOSI','MOSEI'] for s in [2026,2027,2028]]
def read(p):return json.loads(p.read_text())
def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
def load(p):
 with np.load(p,allow_pickle=False) as a:return {k:a[k].copy() for k in ['tokens','audio','vision','mask','ids']}
def scalar(q):return 3*torch.tanh(q[:,3]/3)
def checkfiles(lock):
 for p,h in lock['files'].items():assert r.digest(Path(p))==h,('changed',p)
 assert torch.__version__==lock['torch']
def independent_select(units,values):
 n=max(1,math.ceil(.2*len(units)));sums=[sum(values[j:j+n]) for j in range(len(units)-n+1)];j=max(range(len(sums)),key=lambda i:(sums[i],-i));return sum(units[j:j+n],[])
def prepare():
 OUT.mkdir(exist_ok=False);lock=read(ROOT/'TRAINING_LOCK.json');files=dict(lock['sources']);manifest=read(OLD/'sample_manifest.json');samples={};audits={}
 assert r.digest(OLD/'sample_manifest.json')=='0dc4fd3d09ef5cc1bd252fc32deeab9e2175904ba323cfa83a04c519ad334aaa'
 tok=BertTokenizerFast.from_pretrained(r.b.BERT,local_files_only=True)
 for ds in ['MOSI','MOSEI']:
  m=manifest[ds];cache=Path(m['cache_path']);assert r.digest(cache)==m['cache_sha256'];d=load(cache);files[str(cache)]=r.digest(cache)
  splits={sp:load(BASE/'cache'/ds/(sp+'.npz'))['ids'].tolist() for sp in ['train','valid']};splits['test']=d['ids'].tolist();overlaps={}
  for sp,ids in splits.items():assert len(ids)==len(set(ids))
  for x,y in [('train','valid'),('train','test'),('valid','test')]:
   ids=set(splits[x])&set(splits[y]);vid={v.split('$_$')[0] for v in splits[x]}&{v.split('$_$')[0] for v in splits[y]};assert not ids and not vid;overlaps[x+'_'+y]={'ids':len(ids),'videos':len(vid)}
  if ds=='MOSI':expected=list(range(len(d['ids'])))
  else:
   import hashlib
   h=lambda x:hashlib.sha256(x.encode()).hexdigest()
   videos={v.split('$_$')[0] for v in splits['test']};vs=sorted(videos,key=lambda v:(h('final-test-v1:video:'+v),v))[:300]
   expected=[min([i for i,v in enumerate(splits['test']) if v.split('$_$')[0]==vid],key=lambda i:(h('final-test-v1:sample:'+splits['test'][i]),splits['test'][i],i)) for vid in vs]
  assert expected==m['indices'];rows=[]
  for i in expected:
   sid=str(d['ids'][i]);assert m['all_prediction_samples'][i]['id']==sid
   mask=d['mask'][i,:,0];assert not np.any(mask&np.isin(d['tokens'][i,0],[0,101,102]));units=r.evidence_units(mask,d['tokens'][i,0],tok);assert units
   assert all(all(mask[u]) for u in units);assert sorted(sum(units,[]))==np.flatnonzero(mask).tolist()
   rows.append(dict(index=i,id=sid,video=sid.split('$_$')[0],units=units))
  samples[ds]=dict(cache=str(cache),rows=rows);audits[ds]=dict(samples=len(rows),videos=len({q['video'] for q in rows}),overlaps=overlaps,eligible_text=len(rows))
  assert len(rows)==(686 if ds=='MOSI' else 300)
  assert audits[ds]['videos']==(31 if ds=='MOSI' else 300)
  for seed in [2026,2027,2028]:
   p=ROOT/'runs'/ds/f'seed{seed}';st=read(p/'COMPLETE.json');assert st['predictor_usable'] and not st['test_accessed']
   for name in ['best.pt','normalization.npz','valid_predictions.npz','COMPLETE.json','config.json']:files[str(p/name)]=r.digest(p/name)
 for p in [Path(__file__),ROOT/'TRAINING_LOCK.json',OLD/'sample_manifest.json',ROOT/'PROTOCOL_V1.txt',ROOT/'IMPLEMENTATION_CLARIFICATION.txt',ROOT/'NUMERICAL_EXECUTION_AMENDMENT.txt']+list(Path(r.b.BERT).glob('*.json'))+list(Path(r.b.BERT).glob('*.txt')):files[str(p)]=r.digest(p)
 save(OUT/'samples.json',samples);files[str(OUT/'samples.json')]=r.digest(OUT/'samples.json');save(OUT/'DATA_AUDIT.json',audits);files[str(OUT/'DATA_AUDIT.json')]=r.digest(OUT/'DATA_AUDIT.json')
 obj=dict(stage='prepared_no_new_test_inference',files=files,torch=torch.__version__,numpy=np.__version__,jobs=JOBS,audit=audits,protocol='PROTOCOL_V1.txt',budget=.2,reference='zero',output='3*tanh(raw/3)',retention='negative absolute reconstruction error');checkfiles(obj);save(OUT/'PREPARED.json',obj);print(json.dumps(audits),flush=True)

def compute(model,d,i,tok,smoke):
 device='cuda:0';orig={k:torch.as_tensor(d[k][i:i+1],device=device) for k in ['tokens','audio','vision','mask']};ix=torch.tensor([0],device=device);o=model(orig,ix);e=o['evidence'];raw=float(o['scores'][0,3]);base=float(scalar(o['scores'])[0]);zero=float(model.head.fuse(torch.zeros_like(e).sum(2))['scores'][0,3]);direction=1 if raw-zero>=0 else -1
 mask=d['mask'][i,:,0];units=r.evidence_units(mask,d['tokens'][i,0],tok);L=len(mask);errors=[]
 def positions(pos,keep):
  chosen=np.zeros(L,bool);chosen[list(pos)]=True;return np.flatnonzero(mask&(~chosen if keep else chosen))
 def evaluate(space,jobs,sub):
  values={}
  for start in range(0,len(jobs),32):
   chunk=jobs[start:start+32];n=len(chunk)+1
   if space=='input':
    b={k:v.repeat(n,*([1]*(v.ndim-1))).clone() for k,v in orig.items()}
    for j,(pos,keep) in enumerate(chunk,1):b['tokens'][j,0,positions(pos,keep)]=tok.mask_token_id if sub=='MASK' else tok.unk_token_id
    assert torch.equal(b['mask'],orig['mask'].expand_as(b['mask'])) and torch.equal(b['tokens'][:,1:],orig['tokens'][:,1:].expand(n,-1,-1))
    protected=np.isin(d['tokens'][i,0],[0,101,102]);assert torch.equal(b['tokens'][:,0,protected],orig['tokens'][:,0,protected].expand(n,-1))
    q=model(b,torch.arange(n,device=device))['scores']
   else:
    ee=e.repeat(n,1,1,1).clone()
    for j,(pos,keep) in enumerate(chunk,1):ee[j,0,positions(pos,keep)]=0
    assert torch.equal(ee[:,1:],e[:,1:].expand(n,-1,-1,-1));q=model.head.fuse(ee.sum(2))['scores']
   err=abs(float(q[0,3])-raw);errors.append(err);assert err<1e-4
   v=scalar(q).cpu().numpy();assert np.isfinite(v).all()
   for j,key in enumerate(chunk,1):values[key]=float(v[j])
  return values
 singles=[(tuple(u),False) for u in units];ev=evaluate('evidence',singles,'MASK');rankings=[];matrix=[]
 for sub in ['MASK','UNK']:
  iv=evaluate('input',singles,sub)
  ranks={}
  for name,vv in [('Oi',iv),('Oe',ev)]:
   vals=[direction*(base-vv[(tuple(u),False)]) for u in units];scores=np.zeros(L)
   for u,v in zip(units,vals):scores[u]=v/len(u)
   chosen=r.select_span(units,scores,.2,None);independent=independent_select(units,vals);assert chosen==independent
   ranks[name]=tuple(chosen);rankings.append(dict(substitution=sub,ranking=name,singleton_outputs=[vv[(tuple(u),False)] for u in units],positions=chosen))
  jobs=sorted({(p,keep) for p in ranks.values() for keep in [False,True]})
  for space in ['input','evidence']:
   vv=evaluate(space,jobs,sub)
   for ranking,pos in ranks.items():
    for keep in [False,True]:
     val=vv[(pos,keep)];utility=-abs(base-val) if keep else direction*(base-val)
     matrix.append(dict(substitution=sub,ranking=ranking,evaluation=space,operation='retain' if keep else 'delete',positions=list(pos),after=val,utility=utility))
     if smoke:
      b={k:v.clone() for k,v in orig.items()};ee=e.clone();changed=positions(pos,keep)
      if space=='input':
       b['tokens'][0,0,changed]=tok.mask_token_id if sub=='MASK' else tok.unk_token_id;q=model(b,ix)['scores']
      else:ee[0,0,changed]=0;q=model.head.fuse(ee.sum(2))['scores']
      err=abs(float(scalar(q)[0])-val);errors.append(err);assert err<1e-4
 # Engineering no-op and complete-retention identity checks.
 for space in ['input','evidence']:
  vals=evaluate(space,[((),False),(tuple(np.flatnonzero(mask)),True)],'MASK');assert max(abs(v-base) for v in vals.values())<1e-4
 return dict(index=i,id=str(d['ids'][i]),video=str(d['ids'][i]).split('$_$')[0],units=units,raw=raw,base=base,zero_raw=zero,direction=direction,rankings=rankings,matrix=matrix,max_replay_error=max(errors))

def run(smoke):
 lock=read(OUT/('PREPARED.json' if smoke else 'EVALUATION_LOCK.json'));checkfiles(lock)
 if not smoke:assert lock['stage']=='sealed_after_validation_replay'
 dest=OUT/('smoke' if smoke else 'results');dest.mkdir(exist_ok=False);r.init();torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;tok=BertTokenizerFast.from_pretrained(r.b.BERT,local_files_only=True);manifest=read(OUT/'samples.json');start=time.time();total=0;maxsec=0;statuses=[]
 with torch.inference_mode():
  for ds,seed in JOBS:
   p=ROOT/'runs'/ds/f'seed{seed}';stats=dict(np.load(p/'normalization.npz',allow_pickle=False));model=r.Model(stats,ds).cuda();model.load_state_dict(torch.load(p/'best.pt',map_location='cpu',weights_only=True)['model_state']);model.eval().requires_grad_(False)
   d=load(BASE/'cache'/ds/'valid.npz' if smoke else Path(manifest[ds]['cache']));saved=np.load(p/'valid_predictions.npz',allow_pickle=False) if smoke else None
   canonical_error=0.;precision_drift=None
   if smoke:
    # Exact original batch64/TF32 setting for checkpoint replay, separate from stable evaluation arithmetic.
    vd={k:torch.as_tensor(d[k],device='cuda:0') for k in ['tokens','audio','vision','mask']};allidx=torch.arange(len(d['ids']),device='cuda:0');predictions={}
    for flag in [True,False]:
     torch.backends.cudnn.allow_tf32=flag;predictions[flag]=torch.cat([model(vd,ix)['reg'] for ix in allidx.split(64)]).cpu().numpy()
    assert np.array_equal(saved['ids'],d['ids']);canonical_error=float(np.max(np.abs(predictions[True]-saved['prediction'])));assert canonical_error<1e-4
    drift=np.abs(predictions[False]-predictions[True]);precision_drift=dict(maximum=float(drift.max()),mean=float(drift.mean()),samples=len(drift),reason='TF32 disabled for batch-invariant explanation inference; no retraining or outcome selection')
    save(dest/f'{ds}_{seed}_precision.json',dict(canonical_replay_error=canonical_error,precision_drift=precision_drift));del vd
   if smoke:indices=[0,len(d['ids'])-1]
   else:indices=[q['index'] for q in manifest[ds]['rows']]
   maxerror=0;jobstart=time.time()
   with gzip.open(dest/f'{ds}_{seed}.jsonl.gz','wt',encoding='utf-8') as f:
    for i in indices:
     t=time.time();row=compute(model,d,i,tok,smoke);row.update(dataset=ds,seed=seed)
     if smoke:
      assert str(saved['ids'][i])==row['id'];err=abs(float(predictions[False][i])-row['base']);assert err<1e-4;row['checkpoint_replay_error']=canonical_error;row['fp32_batch_replay_error']=err;row['precision_drift']=precision_drift
     else:
      expected=next(q for q in manifest[ds]['rows'] if q['index']==i);assert row['id']==expected['id'] and row['units']==expected['units']
     f.write(json.dumps(row)+'\n');f.flush();total+=1;maxerror=max(maxerror,row['max_replay_error']);maxsec=max(maxsec,time.time()-t)
     save(dest/'STATUS.json',dict(dataset=ds,seed=seed,index=i,samples=total,seconds=time.time()-start))
   statuses.append(dict(dataset=ds,seed=seed,samples=len(indices),seconds=time.time()-jobstart,max_error=maxerror));print(json.dumps(statuses[-1]),flush=True);del model;torch.cuda.empty_cache()
 checkfiles(lock);obj=dict(status='passed',mode='validation_replay' if smoke else 'test',samples=total,jobs=statuses,seconds=time.time()-start,max_sample_seconds=maxsec,files={p.name:r.digest(p) for p in dest.glob('*.gz')},precision_files={p.name:r.digest(p) for p in dest.glob('*_precision.json')});save(dest/'COMPLETE.json',obj)
def seal():
 p=read(OUT/'PREPARED.json');checkfiles(p);sm=read(OUT/'smoke/COMPLETE.json');local=read(OUT/'LOCAL_SMOKE_VERIFIED.json');assert sm['status']=='passed' and sm['samples']==12 and local['passed'];assert local['complete_sha256']==r.digest(OUT/'smoke/COMPLETE.json')
 for file in [OUT/'smoke/COMPLETE.json',OUT/'LOCAL_SMOKE_VERIFIED.json']:p['files'][str(file)]=r.digest(file)
 for name,h in {**sm['files'],**sm.get('precision_files',{})}.items():assert r.digest(OUT/'smoke'/name)==h;p['files'][str(OUT/'smoke'/name)]=h
 p.update(stage='sealed_after_validation_replay',test_inference_already_run=False,followup_after_original_test=True)
 assert not (OUT/'EVALUATION_LOCK.json').exists();save(OUT/'EVALUATION_LOCK.json',p);print('SEALED',flush=True)
if __name__=='__main__':
 try:
  mode=sys.argv[1]
  if mode=='prepare':prepare()
  elif mode=='smoke':run(True)
  elif mode=='seal':seal()
  elif mode=='test':run(False)
  else:raise ValueError(mode)
 except Exception:
  (ROOT/'EVALUATION_FAILURE.txt').write_text(traceback.format_exc());raise
