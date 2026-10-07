"""Fixed-seed validation pilot; fresh finetuned BERT; no training/test access."""
import sys,json,time,argparse,importlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import numpy as np,torch
from transformers import BertTokenizerFast
from prepare_standard import digest
import run_finetuned
from tei_kan.explain import explain_output,target_values
from tei_kan.faithfulness import evidence_units,select_span

def ig(head,evidence,target,steps):
 z=evidence.sum(2).detach();total=torch.zeros_like(z)
 for lo in range(0,steps+1,32):
  alpha=torch.arange(lo,min(lo+32,steps+1),device=z.device,dtype=z.dtype)/steps
  p=(alpha[:,None,None]*z[0]).detach().requires_grad_(True)
  v=target_values(head.fuse(p)['scores'],target['kind'],target['weights'].expand(len(p),-1))
  g=torch.autograd.grad(v.sum(),p)[0];w=torch.ones_like(alpha);w[alpha==0]=.5;w[alpha==1]=.5
  total+=(g*w[:,None,None]).sum(0,keepdim=True)/steps
 return (evidence.detach()*total[:,:,None]).sum(-1)

def main(a):
 torch.set_num_threads(4);torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}'
 manifest=json.loads((R/'compact_stage/explanation/sample_manifest.json').read_text())[a.dataset]
 ckpath=R/'compact_stage/runs'/a.dataset/'tei_mlp'/a.model/f'seed{a.seed}'/'best.pt';assert digest(ckpath)==manifest['checkpoints'][a.model][str(a.seed)]
 cache=R/'cache'/a.dataset/'valid.npz';assert digest(cache)==manifest['validation_sha256']
 with np.load(cache,allow_pickle=False) as x:d={k:x[k].copy() for k in ['tokens','audio','vision','mask','ids','yc']}
 stats=dict(np.load(ckpath.parent/'normalization.npz'))
 module=importlib.import_module('run_compact_stage')
 model=module.EndToEnd('tei_mlp',stats,a.dataset,a.model).to(device);ck=torch.load(ckpath,map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);del ck;model.eval().requires_grad_(False)
 tokenizer=BertTokenizerFast.from_pretrained(module.BERT,local_files_only=True)
 outdir=R/'compact_stage/explanation'/('smoke' if a.smoke else 'results')/a.dataset/a.model/f'seed{a.seed}'/a.target/a.substitution/str(a.budget);outdir.mkdir(parents=True,exist_ok=False)
 indices=manifest['indices'][:1] if a.smoke else manifest['indices'];allchecks=[];start=time.time()
 for i in indices:
  orig={k:torch.as_tensor(d[k][i:i+1],device=device) for k in ['tokens','audio','vision','mask']};mask=d['mask'][i]
  torch.cuda.synchronize();baseline_start=time.perf_counter()
  with torch.no_grad():o=model(orig,torch.tensor([0],device=device))
  torch.cuda.synchronize();baseline_seconds=time.perf_counter()-baseline_start
  torch.cuda.synchronize();attribution_start=time.perf_counter()
  exp=explain_output(model.head,o,kind=a.target,steps=32,max_steps=2048);target=exp['target'];value=float(exp['value'][0]);delta=float(exp['value'][0]-exp['baseline'][0]);direction=1 if a.target=='margin' or delta>=0 else -1
  torch.cuda.synchronize();coalition_seconds=time.perf_counter()-attribution_start;ig_start=time.perf_counter()
  steps=32
  while True:
   attribution=ig(model.head,o['evidence'],target,steps);residual=abs(float(attribution.sum())-delta)
   if residual<=1e-4+1e-3*abs(delta) or steps==2048:break
   steps*=2
  torch.cuda.synchronize();ig_seconds=time.perf_counter()-ig_start
  units=[evidence_units(mask[:,m],d['tokens'][i,0] if m==0 else None,tokenizer) for m in range(3)]
  scores={'coalition':exp['local'][0].detach().cpu().numpy()*direction,'ig_evidence':attribution[0].detach().cpu().numpy()*direction}
  controls=[]
  def evaluate(jobs):
   results=[]
   for lo in range(0,len(jobs),32):
    chunk=jobs[lo:lo+32];b={k:v.repeat(len(chunk)+1,*([1]*(v.ndim-1))).clone() for k,v in orig.items()}
    for j,(m,pos,retain) in enumerate(chunk,1):
     chosen=np.zeros(50,bool);chosen[pos]=True;replace=mask[:,m]&(~chosen if retain else chosen);ii=torch.as_tensor(np.flatnonzero(replace),device=device)
     if m==0:b['tokens'][j,0,ii]=(tokenizer.mask_token_id if a.substitution=='mask_mean' else tokenizer.unk_token_id)
     else:
      name=['text','audio','vision'][m];b[name][j,ii]=getattr(model.head,name+'_mean') if a.substitution=='mask_mean' else 0
    with torch.no_grad():q=model(b,torch.arange(len(chunk)+1,device=device));v=target_values(q['scores'],a.target,target['weights'].expand(len(chunk)+1,-1))
    controls.append(abs(float(v[0])-value));assert controls[-1]<1e-4
    results.extend(v[1:].cpu().tolist())
   return results
  torch.cuda.synchronize();occ_start=time.perf_counter()
  single=[(m,u,False) for m in range(3) for u in units[m]];vs=evaluate(single);occ=np.zeros((3,50))
  for (m,u,_),v in zip(single,vs):occ[m,u]=direction*(value-v)/len(u)
  torch.cuda.synchronize();occ_seconds=time.perf_counter()-occ_start
  scores['group_occlusion']=occ
  noop=evaluate([(m,[],False) for m in range(3) if units[m]]);assert max(abs(v-value) for v in noop)<1e-4
  jobs=[];metadata=[];rng=np.random.default_rng(20260930+i)
  for m in range(3):
   if not units[m]:continue
   for method in ['coalition','ig_evidence','group_occlusion','random']:
    for repeat in range(10 if method=='random' else 1):
     pos=select_span(units[m],np.zeros(50) if method=='random' else scores[method][m],a.budget,rng if method=='random' else None)
     for retain in [False,True]:jobs.append((m,pos,retain));metadata.append(dict(modality='TAV'[m],method=method,repeat=repeat,operation='retain' if retain else 'delete',positions=pos))
  values=evaluate(jobs)
  with (outdir/'interventions.jsonl').open('a') as f:
   for meta,v in zip(metadata,values):f.write(json.dumps(dict(index=i,sample_id=str(d['ids'][i]),target=a.target,value_before=value,value_after=v,directional_drop=direction*(value-v),absolute_change=abs(value-v),**meta))+'\n')
  check=dict(index=i,baseline_forward_seconds=baseline_seconds,coalition_seconds=coalition_seconds,ig_seconds=ig_seconds,occlusion_seconds=occ_seconds,coalition_converged=bool(exp['integration_converged'].all()),coalition_residual=float(exp['local_residual'].abs().max()),shapley_residual=float(exp['shapley_residual'].abs().max()),ig_steps=steps,ig_total_residual=residual,ig_converged=residual<=1e-4+1e-3*abs(delta),ig_modality_shapley_gap=float((attribution.sum(-1)-exp['phi']).abs().max()),batch_control_error=max(controls),trials=len(jobs))
  assert all(np.isfinite(v).all() for v in scores.values());allchecks.append(check)
  (outdir/f'attribution_{i}.json').write_text(json.dumps(dict(scores={k:v.tolist() for k,v in scores.items()},phi=exp['phi'][0].cpu().tolist(),direction=direction,units=units)))
  (outdir/'checks.json').write_text(json.dumps(allchecks,indent=2));print(json.dumps(check),flush=True)
 (outdir/'status.json').write_text(json.dumps(dict(status='completed',seed=a.seed,dataset=a.dataset,model=a.model,target=a.target,samples=len(indices),smoke_only=a.smoke,elapsed_seconds=time.time()-start,test_evaluated=False,checkpoint_sha256=manifest['checkpoints'][a.model][str(a.seed)],script_sha256=digest(Path(__file__))),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['MOSI','MOSEI'],required=True);p.add_argument('--model',choices=['main_only','main_pair','concat'],required=True);p.add_argument('--target',choices=['margin','intensity'],default='intensity');p.add_argument('--gpu',type=int,default=0);p.add_argument('--seed',type=int,choices=[2026,2027,2028],default=2026);p.add_argument('--budget',type=float,choices=[.1,.2,.3],default=.2);p.add_argument('--substitution',choices=['mask_mean','unk_zero'],default='mask_mean');p.add_argument('--smoke',action='store_true');main(p.parse_args())
