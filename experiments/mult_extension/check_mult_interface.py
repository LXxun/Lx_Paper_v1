"""Validation-only engineering acceptance. No optimizer, test access or scientific ranking."""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import sys,json,hashlib,time,platform,traceback
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizerFast
ROOT=Path(__file__).resolve().parents[1]/'benchmarks'
OUT=Path(__file__).resolve().parent/'mult_interface_v1'
sys.path.insert(0,str(ROOT))
import run_structural_stage as r
from tei_kan.faithfulness import evidence_units,select_span
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def err(a,b):return float(torch.max(torch.abs(a-b)).item())
def require(v,label):
 assert np.isfinite(v) and v<1e-4,(label,v)
 return v
def encode(m,d):
 t=d['tokens'];am=t[:,1].clone();am[am.sum(1)==0,0]=1
 text=m.bert(input_ids=t[:,0],attention_mask=am,token_type_ids=t[:,2]).last_hidden_state
 xs=[text,d['audio'],d['vision']]
 for j,k in enumerate(['text','audio','vision']):
  if j:xs[j]=((xs[j]-torch.as_tensor(m.stats[k+'_mean'],device=text.device))/torch.as_tensor(m.stats[k+'_std'],device=text.device)).clamp(-8,8)
  xs[j]=torch.where(d['mask'][:,:,j,None],xs[j],torch.zeros_like(xs[j]))
 return xs
def head(m,x):return m.head(*x)['M'].reshape(-1)
def main():
 OUT.mkdir(exist_ok=False);start=time.monotonic();torch.set_num_threads(4);torch.cuda.set_device(0);device='cuda:0'
 tok=BertTokenizerFast.from_pretrained(r.BERT,local_files_only=True)
 result={'status':'running','scope':'validation-only engineering','environment':{'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda},'script_sha256':digest(__file__),'runs':[]}
 (OUT/'status.json').write_text(json.dumps(result,indent=2))
 for ds in ['MOSI','MOSEI']:
  cache=ROOT/'cache'/ds/'valid.npz'
  with np.load(cache,allow_pickle=False) as f:arr={k:f[k].copy() for k in ['tokens','audio','vision','mask','ids']}
  cachehash=digest(cache)
  for seed in [2026,2027,2028]:
   assert time.monotonic()-start<1800,'engineering budget exceeded'
   p=ROOT/'structural_stage/runs'/ds/'mult/full'/f'seed{seed}'
   stats=dict(np.load(p/'normalization.npz',allow_pickle=False));r.c.seed_all(seed)
   m=r.EndToEnd('mult',stats,ds,'full').to(device);ck=torch.load(p/'best.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['model_state']);del ck;m.eval().requires_grad_(False)
   rec={'dataset':ds,'seed':seed,'checkpoint_sha256':digest(p/'best.pt'),'validation_sha256':cachehash,'normalization_sha256':digest(p/'normalization.npz'),'manifest_sha256':digest(p/'manifest.json'),'canonical_batch':64,'checks':[]}
   saved=np.load(p/'valid_predictions.npz',allow_pickle=False);assert np.array_equal(saved['ids'],arr['ids']);pred=[];identity=0.;first=None
   with torch.inference_mode():
    for lo in range(0,len(arr['ids']),64):
     d={k:torch.as_tensor(arr[k][lo:lo+64],device=device) for k in ['tokens','audio','vision','mask']}
     full=m(d,torch.arange(len(d['tokens']),device=device))['reg'];x=encode(m,d);y=head(m,x)
     identity=max(identity,require(err(y,full),'interface identity'));pred.extend(full.cpu().tolist())
     if first is None:first=({k:v.clone() for k,v in d.items()},[z.clone() for z in x],full.clone())
    diff=float(np.max(np.abs(np.asarray(pred)-saved['prediction'])));require(diff,'full saved replay')
    rec.update(n=len(pred),saved_replay_max_error=diff,interface_identity_max_error=identity)
    d,x,y=first;mask=arr['mask'][0];units=[evidence_units(mask[:,j],arr['tokens'][0,0] if j==0 else None,tok) for j in range(3)]
    structural=np.isin(arr['tokens'][0,0],[tok.cls_token_id,tok.sep_token_id,tok.pad_token_id]);assert all(not structural[t] for u in units[0] for t in u)
    for sub in ['mask_mean','unk_zero']:
     for j in range(3):
      if not units[j]:rec['checks'].append({'modality':j,'substitution':sub,'unavailable':True});continue
      pos=select_span(units[j],np.zeros(50),.2,None);count=int(np.ceil(.2*len(units[j])));assert any([t for u in units[j][k:k+count] for t in u]==pos for k in range(len(units[j])-count+1))
      chosen=np.zeros(50,bool);chosen[pos]=True;delete=mask[:,j]&chosen;retain=mask[:,j]&~chosen;assert np.array_equal(delete|retain,mask[:,j]) and not np.any(delete&retain)
      for keep,selection in [(False,delete),(True,retain)]:
       ix=torch.as_tensor(np.flatnonzero(selection),device=device);b={k:v.clone() for k,v in d.items()}
       if j==0:b['tokens'][0,0,ix]=tok.mask_token_id if sub=='mask_mean' else tok.unk_token_id
       else:b[['text','audio','vision'][j]][0,ix]=torch.as_tensor(stats[['text','audio','vision'][j]+'_mean'],device=device) if sub=='mask_mean' else 0
       assert torch.equal(b['tokens'][:,1:],d['tokens'][:,1:]) and torch.equal(b['mask'],d['mask'])
       assert torch.equal(b['tokens'][0,:,torch.as_tensor(structural,device=device)],d['tokens'][0,:,torch.as_tensor(structural,device=device)])
       z=encode(m,b);out=head(m,z);repeat=head(m,encode(m,b));check={'modality':j,'substitution':sub,'operation':'retain' if keep else 'delete','positions_changed':len(ix),'repeat_error':require(err(out,repeat),'repeat'),'reencode_vs_forward_error':require(err(out,m(b,torch.arange(len(b['tokens']),device=device))['reg']),'modified interface')}
       zz=[v.clone() for v in x];zz[j][0,ix]=z[j][0,ix]
       if j>0:check['av_local_equivalence_error']=require(err(head(m,zz),out),'AV local encoded equivalence')
       if j>0 and sub=='mask_mean':
        zz[j][0,ix]=0;check['av_zero_equivalence_error']=require(err(head(m,zz),out),'AV mean zero equivalence')
       rec['checks'].append(check)
   rec['status']='passed';result['runs'].append(rec);result['elapsed_seconds']=time.monotonic()-start
   (OUT/'status.json').write_text(json.dumps(result,indent=2));np.savez(OUT/f'{ds}_{seed}_replay.npz',prediction=np.asarray(pred),ids=arr['ids']);print(ds,seed,'PASS',diff,flush=True)
   del m,first,d,x,y;torch.cuda.empty_cache()
 result['status']='passed';result['source_hashes']={str(p):digest(p) for p in [Path(r.__file__),Path(r.c.__file__),ROOT/'repos/MMSA/src/MMSA/models/singleTask/MULT.py',ROOT/'source/tei_kan/faithfulness.py']};(OUT/'status.json').write_text(json.dumps(result,indent=2));(OUT/'COMPLETE').write_text('engineering only\n')
if __name__=='__main__':
 try:main()
 except Exception:
  if OUT.exists():(OUT/'FAILURE.txt').write_text(traceback.format_exc())
  raise
