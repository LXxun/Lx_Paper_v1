"""Fixed validation-only MulT crossing; prepare -> smoke -> seal -> run."""
import os
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
import sys,json,time,hashlib,platform,traceback,argparse,gzip,math
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast
import check_mult_interface as a
R=a.ROOT
S=Path(__file__).resolve().parent/'mult_space_v1'
JOBS=[(d,s) for d in ['MOSI','MOSEI'] for s in [2026,2027,2028]]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def read(p):return json.loads(p.read_text())
def data(ds):
 with np.load(R/'cache'/ds/'valid.npz',allow_pickle=False) as f:return {k:f[k].copy() for k in ['tokens','audio','vision','mask','ids']}
def paths(ds,seed):return R/'structural_stage/runs'/ds/'mult/full'/f'seed{seed}'
def prepare():
 S.mkdir(exist_ok=False)
 old=read(R/'expanded_locked_validation_20261002/sample_manifest.json');accept=read(a.OUT/'status.json');assert accept['status']=='passed' and len(accept['runs'])==6
 tok=BertTokenizerFast.from_pretrained(a.r.BERT,local_files_only=True)
 files={};samples={}
 src=[Path(__file__),Path(a.__file__),Path(a.r.__file__),Path(a.r.c.__file__),R/'expanded_locked_validation_20261002/sample_manifest.json',a.OUT/'status.json',S.parent/'MULT_SPACE_VALIDATION_V1.md']
 src+=list((R/'repos/MMSA/src/MMSA/models/subNets').rglob('*.py'))+list((R/'source/tei_kan').rglob('*.py'))+[R/'repos/MMSA/src/MMSA/models/singleTask/MULT.py',R/'repos/MMSA/src/MMSA/config/config_regression.json',R/'structural_stage/source/model.py']
 src+=list(Path(a.r.BERT).glob('*.json'))+list(Path(a.r.BERT).glob('*.txt'))
 for p in src:files[str(p)]=a.digest(p)
 for ds in ['MOSI','MOSEI']:
  d=data(ds);p=R/'cache'/ds/'valid.npz';assert a.digest(p)==old[ds]['validation_sha256'];files[str(p)]=a.digest(p)
  rows=[]
  assert len(set(old[ds]['indices']))==len(old[ds]['indices'])
  for i,sid,vid in zip(old[ds]['indices'],old[ds]['ids'],old[ds]['videos'],strict=True):
   assert str(d['ids'][i])==sid and sid.split('$_$')[0]==vid
   units=[a.evidence_units(d['mask'][i,:,m],d['tokens'][i,0] if m==0 else None,tok) for m in range(3)]
   rows.append(dict(index=i,id=sid,video=vid,units=units,batch_start=i//64*64,batch_length=min(64,len(d['ids'])-i//64*64)))
  assert len(rows)==(229 if ds=='MOSI' else 300)
  samples[ds]=rows
  for seed in [2026,2027,2028]:
   p=paths(ds,seed)
   for f in ['best.pt','normalization.npz','manifest.json','valid_predictions.npz']:files[str(p/f)]=a.digest(p/f)
   prev=next(q for q in accept['runs'] if q['dataset']==ds and q['seed']==seed);assert files[str(p/'best.pt')]==prev['checkpoint_sha256']
 dump(S/'samples.json',samples);files[str(S/'samples.json')]=a.digest(S/'samples.json')
 lock={'stage':'prepared','files':files,'environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'batch':64,'budget':.2,'target':'native_raw','reference':'zero_prefusion','order':JOBS,'test_access':False,'sample_counts':{d:len(v) for d,v in samples.items()},'videos':{d:len(set(q['video'] for q in v)) for d,v in samples.items()},'eligible':{d:{'TAV'[m]:sum(bool(q['units'][m]) for q in v) for m in range(3)} for d,v in samples.items()}}
 dump(S/'PREPARED.json',lock);print('PREPARED',lock['sample_counts'],lock['videos'],lock['eligible'],flush=True)
def verify(lock):
 for p,h in lock['files'].items():assert a.digest(p)==h,('source changed',p)
 assert torch.__version__==lock['environment']['torch']
def run(smoke):
 lock=read(S/('PREPARED.json' if smoke else 'LOCK.json'));verify(lock)
 if not smoke:
  assert lock['stage']=='sealed' and a.digest(S/'smoke/COMPLETE.json')==lock['smoke_sha256']
  assert a.digest(S/'LOCAL_SMOKE_VERIFIED.json')==lock['local_verification_sha256']
 out=S/('smoke' if smoke else 'results');out.mkdir(exist_ok=False);start=time.monotonic();count=0;timings=[]
 torch.set_num_threads(4);torch.cuda.set_device(0);device='cuda:0';samples=read(S/'samples.json');tok=BertTokenizerFast.from_pretrained(a.r.BERT,local_files_only=True)
 with torch.inference_mode():
  for ds,seed in JOBS:
   d=data(ds);p=paths(ds,seed);stats=dict(np.load(p/'normalization.npz',allow_pickle=False));a.r.c.seed_all(seed)
   model=a.r.EndToEnd('mult',stats,ds,'full').to(device);ck=torch.load(p/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);del ck;model.eval().requires_grad_(False)
   saved=np.load(p/'valid_predictions.npz',allow_pickle=False);assert np.array_equal(d['ids'],saved['ids'])
   rows=samples[ds]
   if smoke:rows=[rows[0],max(rows,key=lambda q:(q['batch_start'],-q['index']))]
   batchid=None
   with gzip.open(out/f'{ds}_{seed}.jsonl.gz','wt',encoding='utf-8') as f:
    for row in rows:
     t=time.monotonic();i=row['index'];lo=row['batch_start'];local=i-lo
     if batchid!=lo:
      batch={k:torch.as_tensor(d[k][lo:lo+64],device=device) for k in ['tokens','audio','vision','mask']};x=a.encode(model,batch);base=a.head(model,x)
      a.require(float(np.max(np.abs(base.cpu().numpy()-saved['prediction'][lo:lo+64]))),'canonical stored output')
      z=a.head(model,[torch.zeros_like(v) for v in x]);batchid=lo
     original=float(base[local]);zero=float(z[local]);direction=1 if original-zero>=0 else -1;mask=d['mask'][i];units=row['units'];cache={}
     for m in range(3):assert units[m]==a.evidence_units(mask[:,m],d['tokens'][i,0] if m==0 else None,tok)
     def evaluate(space,sub,m,pos,retain):
      key=(space,sub if space=='input' else 'zero',m,tuple(pos),retain)
      if key in cache:return cache[key]
      chosen=np.zeros(50,bool);chosen[list(pos)]=True;ix=torch.as_tensor(np.flatnonzero(mask[:,m]&(~chosen if retain else chosen)),device=device)
      xx=[v.clone() for v in x]
      if space=='representation':xx[m][local,ix]=0
      elif m==0:
       b={k:v.clone() for k,v in batch.items()};b['tokens'][local,0,ix]=tok.mask_token_id if sub=='mask_mean' else tok.unk_token_id
       assert torch.equal(b['tokens'][:,1:],batch['tokens'][:,1:]);assert torch.equal(b['mask'],batch['mask'])
       xx=a.encode(model,b)
      else:
       name=['text','audio','vision'][m];val=torch.zeros_like(xx[m][local,ix])
       if sub=='unk_zero':val=(-torch.as_tensor(stats[name+'_mean'],device=device)/torch.as_tensor(stats[name+'_std'],device=device)).clamp(-8,8).expand_as(val)
       xx[m][local,ix]=val
      value=float(a.head(model,xx)[local]);assert math.isfinite(value);cache[key]=value;return value
     record={**row,'dataset':ds,'seed':seed,'original':original,'zero':zero,'direction':direction,'rankings':[],'matrix':[]}
     for sub in ['mask_mean','unk_zero']:
      for m in range(3):
       if not units[m]:continue
       for ranking,space in [('Oi','input'),('Oe','representation')]:
        vals=[evaluate(space,sub,m,u,False) for u in units[m]];scores=np.zeros(50)
        for u,v in zip(units[m],vals):scores[u]=direction*(original-v)/len(u)
        pos=a.select_span(units[m],scores,.2,None)
        rr=dict(substitution=sub,modality='TAV'[m],ranking=ranking,unit_outputs=vals,positions=pos);record['rankings'].append(rr)
        for evaluation in ['input','representation']:
         for keep in [False,True]:
          v=evaluate(evaluation,sub,m,pos,keep);drop=direction*(original-v)
          record['matrix'].append(dict(substitution=sub,modality='TAV'[m],ranking=ranking,evaluation=evaluation,operation='retain' if keep else 'delete',positions=pos,value=v,utility=-drop if keep else drop))
     # Smoke direct full-forward and separate selection checks: no reported scientific estimates.
     if smoke:
      errors=[]
      for sub in ['mask_mean','unk_zero']:
       for m in range(3):
        if not units[m]:continue
        rr=next(q for q in record['rankings'] if q['substitution']==sub and q['modality']=='TAV'[m] and q['ranking']=='Oi')
        for keep in [False,True]:
         b={k:v.clone() for k,v in batch.items()};chosen=np.zeros(50,bool);chosen[rr['positions']]=True;ix=torch.as_tensor(np.flatnonzero(mask[:,m]&(~chosen if keep else chosen)),device=device)
         if m==0:b['tokens'][local,0,ix]=tok.mask_token_id if sub=='mask_mean' else tok.unk_token_id
         else:b[['text','audio','vision'][m]][local,ix]=torch.as_tensor(stats[['text','audio','vision'][m]+'_mean'],device=device) if sub=='mask_mean' else 0
         full=float(model(b,torch.arange(len(b['tokens']),device=device))['reg'][local]);errors.append(a.require(abs(full-evaluate('input',sub,m,rr['positions'],keep)),'independent full forward'))
      record['smoke_full_forward_max_error']=max(errors,default=0.)
     record['elapsed_seconds']=time.monotonic()-t;timings.append(record['elapsed_seconds']);f.write(json.dumps(record)+'\n');f.flush();count+=1
     dump(out/'status.json',{'status':'running','dataset':ds,'seed':seed,'last_index':i,'completed_samples':count,'elapsed_seconds':time.monotonic()-start})
     if not smoke:assert time.monotonic()-start<lock['runtime_limit_seconds'],'frozen time budget exceeded'
   del model,x,base,z;torch.cuda.empty_cache();print(ds,seed,'completed',flush=True)
 verify(lock)
 dump(out/'COMPLETE.json',{'status':'passed','mode':'smoke' if smoke else 'validation','samples':count,'elapsed_seconds':time.monotonic()-start,'max_sample_seconds':max(timings),'files':{p.name:a.digest(p) for p in out.glob('*.gz')}})
def seal():
 p=read(S/'PREPARED.json');verify(p);s=read(S/'smoke/COMPLETE.json');local=read(S/'LOCAL_SMOKE_VERIFIED.json');assert s['status']=='passed' and s['samples']==12 and local['passed'] and local['smoke_sha256']==a.digest(S/'smoke/COMPLETE.json')
 assert not (S/'LOCK.json').exists()
 p.update(stage='sealed',smoke_sha256=a.digest(S/'smoke/COMPLETE.json'),local_verification_sha256=a.digest(S/'LOCAL_SMOKE_VERIFIED.json'),runtime_limit_seconds=max(3600,math.ceil(s['max_sample_seconds']*1587*3+900)),command='python -B -u scripts/run_mult_space.py run')
 dump(S/'LOCK.json',p);print('SEALED',p['runtime_limit_seconds'],flush=True)
if __name__=='__main__':
 mode=sys.argv[1]
 try:
  if mode=='prepare':prepare()
  elif mode=='seal':seal()
  elif mode in ['smoke','run']:run(mode=='smoke')
  else:raise ValueError(mode)
 except Exception:
  if S.exists():(S/f'FAILURE_{mode}.txt').write_text(traceback.format_exc())
  raise
