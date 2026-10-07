import json,itertools
from pathlib import Path
from collections import defaultdict
import numpy as np
S=Path(__file__).resolve().parent;A=S/'analysis';manifest=json.loads((S/'sample_manifest.json').read_text());parse=json.loads((A/'parse_summary.json').read_text());assert parse['records']==9543744 and parse['all_expected_cells_present']
subs=['mask_mean','unk_zero'];refs=['zero','encoded'];targets=['raw_intensity','intensity'];budgets=[.1,.2,.3];methods=['coalition','ig_evidence','group_occlusion_input','group_occlusion_evidence','random'];ops=['delete','retain'];cells=[];overlap=[];bootcache={}
for ds in ['MOSI','MOSEI']:
 vids=np.array(manifest[ds]['videos'])
 for variant in ['main_only','main_pair','concat']:
  with np.load(A/f'{ds}_{variant}.npz') as z:means=z['means'];spans=z['spans'];gaps=z['gaps'];eligible=z['eligible']
  for su,re,ta,bu,mo,op in itertools.product(range(2),range(2),range(2),range(3),range(3),range(2)):
   valid=eligible[:,mo];vvids=vids[valid];clusterids=sorted(set(vvids));cachekey=(ds,tuple(np.flatnonzero(valid)))
   if cachekey not in bootcache:
    membership=np.array([vvids==v for v in clusterids],float);nc=membership.sum(1);rng=np.random.default_rng(20261002);bc=rng.multinomial(len(clusterids),np.full(len(clusterids),1/len(clusterids)),size=20000).astype(float);den=bc@nc;bootcache[cachekey]=(membership,bc,den)
   membership,bc,den=bootcache[cachekey];values=means[:,valid,su,re,ta,bu,mo,op];sign=1 if op==0 else -1;dd=sign*(values[:,:,0:1,:]-values[:,:,1:,:]);samples=dd.mean(0);ct=membership@samples.reshape(len(vvids),8);draw=(bc@ct)/den[:,None];ci=np.quantile(draw,[.025,.975],axis=0).reshape(2,4,2);mu=samples.mean(0);perseed=dd.mean(1)
   out=dict(dataset=ds,variant=variant,substitution=subs[su],reference=refs[re],target=targets[ta],budget=budgets[bu],modality='TAV'[mo],operation=ops[op],samples_per_seed=int(valid.sum()),videos=len(clusterids),comparisons={})
   for j,method in enumerate(methods[1:]):
    out['comparisons'][method]={space:dict(mean=float(mu[j,k]),ci95=ci[:,j,k].tolist(),per_seed=perseed[:,j,k].tolist(),positive_seeds=int((perseed[:,j,k]>0).sum())) for k,space in enumerate(['input','evidence'])}
   rr=out['comparisons']['random']['input'];out['stability_gate_input']=rr['mean']>0 and rr['positive_seeds']>=2;out['coalition_mean_absolute_gap']=dict(zip(['outside_reencoding','changed_position_encoding','reference_shift','input_minus_zero_evidence'],gaps[:,valid,su,re,ta,bu,mo,op].mean((0,1)).tolist()));cells.append(out)
  for mo,me,bu in itertools.product(range(3),range(4),range(3)):
   valid=eligible[:,mo]
   for change in ['target_change','reference_change','substitution_change']:
    pairs=[]
    if change=='target_change':
     for su,re in itertools.product(range(2),range(2)):pairs.append((spans[:,valid,su,re,0,bu,mo,me],spans[:,valid,su,re,1,bu,mo,me]))
    elif change=='reference_change':
     for su,ta in itertools.product(range(2),range(2)):pairs.append((spans[:,valid,su,0,ta,bu,mo,me],spans[:,valid,su,1,ta,bu,mo,me]))
    else:
     for re,ta in itertools.product(range(2),range(2)):pairs.append((spans[:,valid,0,re,ta,bu,mo,me],spans[:,valid,1,re,ta,bu,mo,me]))
    aa=np.concatenate([x.ravel() for x,y in pairs]);bb=np.concatenate([y.ravel() for x,y in pairs]);ja=[(int(x)&int(y)).bit_count()/(int(x)|int(y)).bit_count() for x,y in zip(aa,bb)];overlap.append(dict(dataset=ds,variant=variant,budget=budgets[bu],modality='TAV'[mo],method=methods[me],change=change,model_segment_pairs=len(aa),same_span_rate=float((aa==bb).mean()),mean_jaccard=float(np.mean(ja))))
  print('statistics',ds,variant,flush=True)
games=json.loads((A/'games_all.json').read_text());diagnostics=[]
for variant,target in itertools.product(['main_only','main_pair','concat'],targets):
 g=[x for x in games if x['variant']==variant and x['target']==target];diagnostics.append(dict(variant=variant,target=target,games=len(g),mean_pair_l1=float(np.mean([x['pair_l1'] for x in g])),max_triple_abs=max(abs(x['triple']) for x in g),max_common_grid_coalition_ig=max(x['check']['common_grid_coalition_ig_difference'] for x in g)))
summary=dict(cells=cells,selection_overlap=overlap,game_diagnostics=diagnostics,records=parse['records'],attribution_games=len(games),models=18,cohorts={ds:dict(segments=len(x['indices']),videos=len(set(x['videos'])),modality_segments={m:sum(m in mm for mm in x['eligible_modalities'].values()) for m in 'TAV'}) for ds,x in manifest.items()},bootstrap_draws=20000,uncertainty='Pointwise whole-video bootstrap conditional on fixed three trained seeds; no simultaneous significance claim.',test_evaluated=False)
(S/'summary_expanded.json').write_text(json.dumps(summary,indent=2));print('SUMMARY COMPLETE',flush=True)
