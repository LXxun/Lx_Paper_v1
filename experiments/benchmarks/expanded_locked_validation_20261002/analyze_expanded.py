import sys,json,gzip,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
S=Path(__file__).resolve().parent;R=S.parent;sys.path.insert(0,str(R))
from prepare_standard import digest
manifest=json.loads((S/'sample_manifest.json').read_text());lock=json.loads((S/'LOCK.json').read_text());assert digest(S/'sample_manifest.json')==lock['manifest_sha256'];assert digest(S/'run.py')==lock['files']['run.py']
A=S/'analysis';A.mkdir(exist_ok=True)
subs=['mask_mean','unk_zero'];refs=['zero','encoded'];targets=['raw_intensity','intensity'];budgets=[.1,.2,.3];methods=['coalition','ig_evidence','group_occlusion_input','group_occlusion_evidence','random'];operations=['delete','retain'];seeds=[2026,2027,2028]
lookup=[{v:i for i,v in enumerate(x)} for x in [subs,refs,targets,budgets,list('TAV'),operations,methods]]
metadata={};diagnostics=[];total=0
for ds in ['MOSI','MOSEI']:
 ids=manifest[ds]['indices'];index={i:j for j,i in enumerate(ids)};N=len(ids);eligible=np.array([[m in manifest[ds]['eligible_modalities'][str(i)] for m in 'TAV'] for i in ids]);expected=sum(eligible.flatten())*224*3
 for variant in ['main_only','main_pair','concat']:
  cache=A/f'{ds}_{variant}.npz';shape=(3,N,2,2,2,3,3,2,5);stlist=[]
  for seed in seeds:
   folder=S/'results'/ds/variant/f'seed{seed}';st=json.loads((folder/'status.json').read_text());assert st['status']=='completed' and not st['test_evaluated'];assert st['records']==expected and st['games']==N*8;assert st['script_sha256']==lock['files']['run.py'];assert st['checkpoint_sha256']==manifest[ds]['checkpoints'][variant][str(seed)];assert st['validation_sha256']==manifest[ds]['validation_sha256'];assert st['max_batch_error']<1e-4 and st['max_prior_attribution_replay_error']<1e-4;stlist.append(st)
  sums=np.zeros(shape+(2,),np.float64);counts=np.zeros(shape,np.uint8);bits=np.zeros(shape,np.uint16);spans=np.zeros((3,N,2,2,2,3,3,4),np.uint64);gaps=np.zeros((3,N,2,2,2,3,3,2,4),np.float64)
  for si,seed in enumerate(seeds):
   folder=S/'results'/ds/variant/f'seed{seed}';count=0;start=time.time()
   with gzip.open(folder/'paired.jsonl.gz','rt') as f:
    for line in f:
     x=json.loads(line);j=index[x['index']];assert x['dataset']==ds and x['variant']==variant and x['seed']==seed
     su,re,ta,bu,mo,op,me=[table[x[name]] for table,name in zip(lookup,['substitution','reference','target','budget','modality','operation','method'])];assert eligible[j,mo];key=(si,j,su,re,ta,bu,mo,op,me);bit=1<<x['repeat'];assert not int(bits[key])&bit;bits[key]|=bit;counts[key]+=1
     metric='directional_drop' if op==0 else 'absolute_change';sums[key]+=[x[metric],x['representation_'+metric]]
     assert abs(x['input_minus_zero_evidence']-x['outside_position_signed_gap']-x['changed_position_signed_gap']-x['reference_shift'])<1e-6
     if op==0 and me<4:spans[si,j,su,re,ta,bu,mo,me]=sum(1<<p for p in x['positions'])
     if me==0:gaps[si,j,su,re,ta,bu,mo,op]=np.abs([x['outside_position_signed_gap'],x['changed_position_signed_gap'],x['reference_shift'],x['input_minus_zero_evidence']])
     count+=1
   assert count==stlist[si]['records'];total+=count
   gg=[json.loads(t) for t in (folder/'games.jsonl').read_text().splitlines()];assert len(gg)==N*8;diagnostics+=gg
   print('parsed',ds,variant,seed,count,round(time.time()-start,1),flush=True)
  wanted=np.broadcast_to(eligible[None,:,None,None,None,None,:,None,None],shape).copy();expected_counts=np.where(wanted,np.array([1,1,1,1,10]),0);expected_bits=np.where(wanted,np.array([1,1,1,1,1023]),0);assert np.array_equal(counts,expected_counts) and np.array_equal(bits,expected_bits);assert np.isfinite(sums).all()
  means=np.divide(sums,counts[...,None],out=np.zeros_like(sums),where=counts[...,None]>0);np.savez_compressed(cache,means=means,spans=spans,gaps=gaps,eligible=eligible);metadata[ds+'_'+variant]=stlist
(A/'parse_summary.json').write_text(json.dumps(dict(records=total,models=18,statuses=metadata,all_expected_cells_present=True,test_evaluated=False),indent=2))
(A/'games_all.json').write_text(json.dumps(diagnostics))
print('PARSE COMPLETE',total,flush=True)
