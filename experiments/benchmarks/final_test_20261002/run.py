import sys,json,time,argparse,gzip
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
import run_compact_stage as r
import numpy as np,torch
from transformers import BertTokenizerFast
from prepare_standard import digest
from tei_kan.explain import explain_output
from tei_kan.faithfulness import evidence_units,select_span
from compact_stage.explain import ig
S=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--phase',choices=['replay','test'],required=True);a=p.parse_args()
from guard import verify
verify(a.phase)
torch.set_num_threads(4);torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}'
manifest=json.loads((S/('replay_manifest.json' if a.phase=='replay' else 'sample_manifest.json')).read_text());tok=BertTokenizerFast.from_pretrained(r.BERT,local_files_only=True);struct_ids=[tok.cls_token_id,tok.sep_token_id,tok.pad_token_id]
class Shifted:
 training=False
 def __init__(self,head,ref):self.head=head;self.ref=ref.sum(2)
 def fuse(self,z):return dict(scores=self.head.fuse(z+self.ref)['scores'],main=None,interactions=None)
def scalar(scores,kind):return scores[:,3] if kind=='raw_intensity' else 3*torch.tanh(scores[:,3]/3)
def metrics_game(vals):
 terms=[]
 for subset in range(1,8):terms.append(sum((-1)**((subset^s).bit_count())*vals[s] for s in range(8) if s&subset==s))
 return dict(moebius=terms,pair_l1=sum(abs(terms[s-1]) for s in [3,5,6]),triple=terms[6])
jobs=[(ds,v,seed) for ds in ['MOSI','MOSEI'] for v in ['main_only','main_pair','concat'] for seed in [2026,2027,2028]]

for ds,v,seed in jobs[a.gpu::2]:
 out=S/('replay_explanations' if a.phase=='replay' else 'results')/ds/v/f'seed{seed}';out.mkdir(parents=True,exist_ok=False);start=time.time()
 lock=json.loads((S/'expected_artifacts.json').read_text());assert digest(Path(__file__))==json.loads((S/'CODE_LOCK.json').read_text())['run.py']
 ckpath=R/'compact_stage/runs'/ds/'tei_mlp'/v/f'seed{seed}'/'best.pt';assert digest(ckpath)==manifest[ds]['checkpoints'][v][str(seed)]
 assert all(digest(R/rel)==h for rel,h in lock['artifacts'].items() if rel.endswith('.py'))
 cache=Path(manifest[ds]['cache_path']);assert digest(cache)==manifest[ds]['cache_sha256']
 with np.load(cache) as x:d={k:x[k].copy() for k in ['tokens','audio','vision','mask','ids']}
 structural=np.isin(d['tokens'][:,0],struct_ids);assert not np.any(structural&d['mask'][:,:,0])
 assert digest(ckpath.parent/'normalization.npz')==lock['artifacts'][str((ckpath.parent/'normalization.npz').relative_to(R))]
 stats=dict(np.load(ckpath.parent/'normalization.npz'));model=r.EndToEnd('tei_mlp',stats,ds,v).to(device);ck=torch.load(ckpath,map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);del ck;model.eval().requires_grad_(False)
 indices=manifest[ds]['indices'];checks=[];controls=[];records=0;game_count=0;replay_errors=[]
 with gzip.open(out/'paired.jsonl.gz','wt',encoding='utf-8',compresslevel=1) as file,(out/'games.jsonl').open('w') as games:
  for i in indices:
   orig={k:torch.as_tensor(d[k][i:i+1],device=device) for k in ['tokens','audio','vision','mask']};mask=d['mask'][i].copy();mask[:,0]&=~structural[i];assert np.array_equal(mask,d['mask'][i])
   with torch.no_grad():o=model(orig,torch.tensor([0],device=device))
   e=o['evidence'].detach();score=o['scores'].detach();raw=float(score[0,3]);zero=torch.zeros_like(e)
   with torch.no_grad():zscore=model.head.fuse(zero.sum(2))['scores'];direction=1 if raw-float(zscore[0,3])>=0 else -1
   units=[evidence_units(mask[:,m],d['tokens'][i,0] if m==0 else None,tok) for m in range(3)];assert all(not structural[i,t] for u in units[0] for t in u)
   def replacement(m,pos,retain):
    chosen=np.zeros(50,bool);chosen[list(pos)]=True;return np.flatnonzero(mask[:,m]&(~chosen if retain else chosen))
   def raw_evaluate(jobs,sub):
    vals={}
    with torch.no_grad():
     for lo in range(0,len(jobs),32):
      chunk=jobs[lo:lo+32];n=len(chunk)+1;b={k:val.repeat(n,*([1]*(val.ndim-1))).clone() for k,val in orig.items()}
      for j,(m,pos,retain) in enumerate(chunk,1):
       ii=torch.as_tensor(replacement(m,pos,retain),device=device)
       if m==0:b['tokens'][j,0,ii]=tok.mask_token_id if sub=='mask_mean' else tok.unk_token_id
       else:b[['text','audio','vision'][m]][j,ii]=getattr(model.head,['text','audio','vision'][m]+'_mean') if sub=='mask_mean' else 0
      assert torch.equal(b['tokens'][:,:,structural[i]],orig['tokens'][:,:,structural[i]].expand(n,-1,-1))
      assert torch.equal(b['tokens'][:,1:],orig['tokens'][:,1:].expand(n,-1,-1))
      q=model(b,torch.arange(n,device=device));err=abs(float(q['scores'][0,3])-raw);controls.append(err);assert err<1e-4
      for j,key in enumerate(chunk,1):vals[key]=(q['scores'][j:j+1].detach(),q['evidence'][j:j+1].detach())
    return vals
   def evidence_outputs(jobs,ref):
    vals={}
    with torch.no_grad():
     for lo in range(0,len(jobs),128):
      chunk=jobs[lo:lo+128];ee=e.repeat(len(chunk),1,1,1).clone()
      for j,(m,pos,retain) in enumerate(chunk):
       ii=torch.as_tensor(replacement(m,pos,retain),device=device);ee[j,m,ii]=ref[0,m,ii]
      ss=model.head.fuse(ee.sum(2))['scores']
      for j,key in enumerate(chunk):vals[key]=ss[j:j+1].detach()
    return vals
   for sub in ['mask_mean','unk_zero']:
    rr={k:val.clone() for k,val in orig.items()}
    for m in range(3):
     ii=torch.as_tensor(np.flatnonzero(mask[:,m]),device=device)
     if m==0:rr['tokens'][0,0,ii]=tok.mask_token_id if sub=='mask_mean' else tok.unk_token_id
     else:rr[['text','audio','vision'][m]][0,ii]=getattr(model.head,['text','audio','vision'][m]+'_mean') if sub=='mask_mean' else 0
    with torch.no_grad():ro=model(rr,torch.tensor([0],device=device))
    reference=ro['evidence'].detach().clone()
    for m in range(3):reference[0,m,torch.as_tensor(np.flatnonzero(~mask[:,m]),device=device)]=e[0,m,torch.as_tensor(np.flatnonzero(~mask[:,m]),device=device)]
    assert torch.equal(rr['tokens'][:,:,structural[i]],orig['tokens'][:,:,structural[i]])
    single=[(m,tuple(u),False) for m in range(3) for u in units[m]];raw_values=raw_evaluate(single,sub);configs=[]
    for refname,ref in [('zero',zero),('encoded',reference)]:
     head=Shifted(model.head,ref);de=e-ref;deout=dict(scores=score,z=de.sum(2),evidence=de)
     evid_single=evidence_outputs(single,ref)
     for kind in ['raw_intensity','intensity']:
      exp=explain_output(head,deout,kind=kind,steps=32,max_steps=2048);assert bool(exp['integration_converged'].all());delta=float(exp['value'][0]-exp['baseline'][0]);steps=32
      while True:
       attribution=ig(head,de,exp['target'],steps);igres=abs(float(attribution.sum())-delta)
       if igres<=1e-4+1e-3*abs(delta) or steps==2048:break
       steps*=2
      assert igres<=1e-4+1e-3*abs(delta)
      base=float(scalar(score,kind)[0]);assert abs(float(exp['value'][0])-base)<1e-4
      scores={'coalition':exp['local'][0].cpu().numpy()*direction,'ig_evidence':attribution[0].cpu().numpy()*direction}
      for method,values in [('group_occlusion_input',raw_values),('group_occlusion_evidence',evid_single)]:
       occ=np.zeros((3,50))
       for (m,u,_),vv in values.items():
        ss=vv[0] if method=='group_occlusion_input' else vv;value=float(scalar(ss,kind)[0]);occ[m,list(u)]=direction*(base-value)/len(u)
       scores[method]=occ
      assert all(np.isfinite(s).all() for s in scores.values())
      common_ig=ig(head,de,exp['target'],exp['integration_steps']);common_grid_difference=float((exp['local']-common_ig).abs().max())
      if v=='main_only' and kind=='raw_intensity':assert common_grid_difference<1e-5
      gv=exp['coalition_values'][0].cpu().tolist();check=dict(common_grid_coalition_ig_difference=common_grid_difference,index=i,substitution=sub,reference=refname,target=kind,coalition_steps=exp['integration_steps'],local_residual=float(exp['local_residual'].abs().max()),shapley_residual=float(exp['shapley_residual'].abs().max()),ig_residual=igres,ig_steps=steps,coalition_ig_local_max_difference=float((exp['local']-attribution).abs().max()))
      game=dict(dataset=ds,variant=v,seed=seed,index=i,video=str(d['ids'][i]).split('$_$')[0],substitution=sub,reference=refname,target=kind,direction=direction,value=base,baseline=float(exp['baseline'][0]),delta=delta,phi=exp['phi'][0].cpu().tolist(),coalition_values=gv,modality_ig=attribution[0].sum(-1).cpu().tolist(),**metrics_game(gv),check=check)
      games.write(json.dumps(game)+'\n');game_count+=1;checks.append(check)
      configs.append(dict(reference=refname,ref=ref,target=kind,base=base,scores=scores,baseline=float(exp['baseline'][0])))
      if a.phase=='replay' and refname=='zero' and kind=='intensity' and (R/'compact_stage/explanation/results'/ds/v/f'seed{seed}'/'intensity'/sub/'0.2'/f'attribution_{i}.json').exists():
       prior=R/'compact_stage/explanation/results'/ds/v/f'seed{seed}'/'intensity'/sub/'0.2'/f'attribution_{i}.json';old=json.loads(prior.read_text());assert old['units']==units and old['direction']==direction
       err=max(np.max(np.abs(scores[new]-np.asarray(old['scores'][previous]))) for new,previous in [('coalition','coalition'),('ig_evidence','ig_evidence'),('group_occlusion_input','group_occlusion')]);replay_errors.append(float(err));assert err<1e-4
    alljobs=set();metas=[]
    for conf in configs:
     for budget in [.1,.2,.3]:
      rng=np.random.default_rng(20260930+i)
      for m in range(3):
       if not units[m]:continue
       for method in ['coalition','ig_evidence','group_occlusion_input','group_occlusion_evidence','random']:
        for repeat in range(10 if method=='random' else 1):
         pos=select_span(units[m],np.zeros(50) if method=='random' else conf['scores'][method][m],budget,rng if method=='random' else None);count=int(np.ceil(budget*len(units[m])));assert any([t for unit in units[m][j:j+count] for t in unit]==pos for j in range(len(units[m])-count+1))
         for retain in [False,True]:
          key=(m,tuple(pos),retain);alljobs.add(key);metas.append((conf,budget,method,repeat,key))
    alljobs=sorted(alljobs);missing=[j for j in alljobs if j not in raw_values];raw_values.update(raw_evaluate(missing,sub))
    erefs={refname:evidence_outputs(alljobs,ref) for refname,ref in [('zero',zero),('encoded',reference)]}
    hybrid={}
    with torch.no_grad():
     for lo in range(0,len(alljobs),128):
      chunk=alljobs[lo:lo+128];eh=e.repeat(len(chunk),1,1,1).clone()
      for j,key in enumerate(chunk):
       m,pos,retain=key;ii=torch.as_tensor(replacement(m,pos,retain),device=device);eh[j,m,ii]=raw_values[key][1][0,m,ii]
      ss=model.head.fuse(eh.sum(2))['scores']
      for j,key in enumerate(chunk):hybrid[key]=ss[j:j+1].detach()
    for conf,budget,method,repeat,key in metas:
     m,pos,retain=key;kind=conf['target'];base=conf['base'];av=float(scalar(raw_values[key][0],kind)[0]);ev=float(scalar(erefs[conf['reference']][key],kind)[0]);hv=float(scalar(hybrid[key],kind)[0]);zv=float(scalar(erefs['zero'][key],kind)[0]);outside=av-hv;inside=hv-ev;refshift=ev-zv
     assert np.isfinite([av,ev,hv,zv]).all();assert abs((av-zv)-(outside+inside+refshift))<1e-6
     row=dict(dataset=ds,variant=v,seed=seed,index=i,video=str(d['ids'][i]).split('$_$')[0],substitution=sub,reference=conf['reference'],target=kind,budget=budget,modality='TAV'[m],method=method,repeat=repeat,operation='retain' if retain else 'delete',positions=list(pos),value_before=base,value_after=av,directional_drop=direction*(base-av),absolute_change=abs(base-av),representation_after=ev,representation_directional_drop=direction*(base-ev),representation_absolute_change=abs(base-ev),outside_position_signed_gap=outside,changed_position_signed_gap=inside,reference_shift=refshift,input_minus_zero_evidence=av-zv)
     file.write(json.dumps(row)+'\n');records+=1
    noop=evidence_outputs([(m,(),False) for m in range(3)],reference);assert max(float((vv-score).abs().max()) for vv in noop.values())<1e-4
   (out/'checks.json').write_text(json.dumps(checks,indent=2));print(ds,v,seed,i,'games',game_count,'records',records,flush=True)
 expected=sum(len(manifest[ds]['eligible_modalities'][str(i)]) for i in indices)*224*3;assert records==expected and game_count==len(indices)*8
 (out/'status.json').write_text(json.dumps(dict(status='completed',dataset=ds,variant=v,seed=seed,samples=len(indices),games=game_count,records=records,max_batch_error=max(controls),max_prior_attribution_replay_error=max(replay_errors,default=0.0),prior_replay_count=len(replay_errors),structural_tokens_observed_valid=0,protected_and_legacy_masks_identical=True,elapsed_seconds=time.time()-start,checkpoint_sha256=digest(ckpath),validation_sha256=digest(cache),normalization_sha256=digest(ckpath.parent/'normalization.npz'),script_sha256=digest(Path(__file__)),phase=a.phase,test_evaluated=(a.phase=='test')),indent=2));print('COMPLETE',ds,v,seed,time.time()-start,flush=True)
 del model;torch.cuda.empty_cache()
