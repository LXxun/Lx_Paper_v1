"""Local independent audit and fixed video-cluster inference; no model/data access."""
from pathlib import Path
import json,gzip,hashlib,math,csv,datetime
import numpy as np
R=Path(__file__).resolve().parent;D=R/'explanation_fp32';A=D/'analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 complete=json.loads((D/'results/COMPLETE.json').read_text());assert complete['status']=='passed' and complete['samples']==2958
 samples=json.loads((D/'samples.json').read_text());audit=json.loads((D/'DATA_AUDIT.json').read_text());records={};count=0;maxerr=0
 for name,h in complete['files'].items():
  p=D/'results'/name;assert sha(p)==h
  with gzip.open(p,'rt',encoding='utf-8') as f:
   for line in f:
    row=json.loads(line);ds=row['dataset'];sid=row['id'];seed=row['seed'];key=(ds,sid,seed);assert key not in records
    expected=next(q for q in samples[ds]['rows'] if q['id']==sid);assert row['index']==expected['index'] and row['units']==expected['units'] and row['video']==expected['video']
    assert abs(row['base']-3*np.tanh(row['raw']/3))<1e-6 and row['direction']==(1 if row['raw']-row['zero_raw']>=0 else -1)
    maxerr=max(maxerr,row['max_replay_error']);assert row['max_replay_error']<1e-4
    units=row['units'];n=math.ceil(.2*len(units));assert n>=1 and len(row['rankings'])==4 and len(row['matrix'])==16
    rankdict={}
    for q in row['rankings']:
     assert len(q['singleton_outputs'])==len(units);v=[row['direction']*(row['base']-x) for x in q['singleton_outputs']];assert np.isfinite(v).all()
     j=max(range(len(units)-n+1),key=lambda i:(sum(v[i:i+n]),-i));assert q['positions']==sum(units[j:j+n],[]);rankdict[q['substitution'],q['ranking']]=q['positions']
    md={}
    for q in row['matrix']:
     mk=(q['substitution'],q['ranking'],q['evaluation'],q['operation']);assert mk not in md;assert q['positions']==rankdict[q['substitution'],q['ranking']]
     u=-abs(row['base']-q['after']) if q['operation']=='retain' else row['direction']*(row['base']-q['after']);assert np.isfinite(u) and abs(u-q['utility'])<1e-12;md[mk]=u
    assert set(md)=={(s,r,e,o) for s in ['MASK','UNK'] for r in ['Oi','Oe'] for e in ['input','evidence'] for o in ['delete','retain']}
    records[key]=md;count+=1
 assert count==2958;summary={};individual=[];cells=[]
 for ds,rngseed in [('MOSI',2026100701),('MOSEI',2026100702)]:
  rows=samples[ds]['rows'];ids=[q['id'] for q in rows];vids=[q['video'] for q in rows];videos=sorted(set(vids));assert len(ids)==audit[ds]['samples']
  mean={}
  for ranking in ['Oi','Oe']:
   for space in ['input','evidence']:
    for op in ['delete','retain']:
     arr=np.array([[records[ds,sid,seed][sub,ranking,space,op] for seed in [2026,2027,2028] for sub in ['MASK','UNK']] for sid in ids]);assert arr.shape==(len(ids),6)
     mean[ranking,space,op]=arr.mean(1);cells.append(dict(dataset=ds,ranking=ranking,evaluation=space,operation=op,mean=float(arr.mean()),samples=len(ids),videos=len(videos)))
  deltas=np.stack([mean['Oe','evidence',op]-mean['Oi','evidence',op] for op in ['delete','retain']],axis=1)
  counts=np.array([vids.count(v) for v in videos]);sums=np.array([deltas[np.array(vids)==v].sum(0) for v in videos]);rng=np.random.default_rng(rngseed);boot=[]
  for lo in range(0,50000,1000):
   ix=rng.integers(len(videos),size=(min(1000,50000-lo),len(videos)));boot.append(sums[ix].sum(1)/counts[ix].sum(1)[:,None])
  boot=np.concatenate(boot);ci=np.quantile(boot,[.00625,.99375],axis=0)
  primary=[]
  for j,op in enumerate(['delete','retain']):primary.append(dict(operation=op,contrast='Ue(Oe)-Ue(Oi)',mean=float(deltas[:,j].mean()),ci_lower=float(ci[0,j]),ci_upper=float(ci[1,j]),interpretation='positive' if ci[0,j]>0 else 'contrary' if ci[1,j]<0 else 'uncertain'))
  for seed in [2026,2027,2028]:
   for sub in ['MASK','UNK']:
    for op in ['delete','retain']:
     diff=[records[ds,sid,seed][sub,'Oe','evidence',op]-records[ds,sid,seed][sub,'Oi','evidence',op] for sid in ids];individual.append(dict(dataset=ds,seed=seed,substitution=sub,operation=op,mean=float(np.mean(diff))))
  summary[ds]=dict(samples=len(ids),videos=len(videos),primary=primary,reverse_descriptive={op:float((mean['Oi','input',op]-mean['Oe','input',op]).mean()) for op in ['delete','retain']},bootstrap_seed=rngseed)
 A.mkdir(exist_ok=False)
 report=dict(status='locally_verified',samples_across_seeds=count,matrix_records=count*16,max_control_error=maxerr,bootstrap_replicates=50000,ci='98.75% percentile intervals; Bonferroni family of four',aggregation='mean over 3 seeds and 2 substitutions per segment, segment-weighted mean, whole-video cluster bootstrap',summary=summary,source_hashes={n:sha(D/'results'/n) for n in complete['files']},analysis_sha256=sha(Path(__file__)))
 (A/'RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 for name,data in [('matrix.csv',cells),('individual_descriptive.csv',individual)]:
  with (A/name).open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 lines=['# Independent recurrent-encoder extension','', 'All six runs retained; same tokenizer and downstream head, randomly initialized recurrent text encoder. Post-original-test follow-up; not independent confirmation of all design choices. MASK/UNK reference semantics and unequal pretraining remain limitations.','', 'Training selected bounded-output validation MAE; see IMPLEMENTATION_CLARIFICATION.txt.','', 'Primary contrasts are Ue(Oe)-Ue(Oi), text, 20% consecutive whole-WordPiece units, bounded output, zero evidence reference. Four comparisons have 98.75% video-cluster percentile intervals.','']
 for ds,q in summary.items():
  lines.append(f"{ds}: {q['samples']} segments / {q['videos']} videos")
  for p in q['primary']:lines.append(f"- {p['operation']}: {p['mean']:.6f} [{p['ci_lower']:.6f}, {p['ci_upper']:.6f}] ({p['interpretation']})")
 lines+=['','These contrasts evaluate matched intervention-space sensitivity, not superiority of a new attribution algorithm or human semantic faithfulness. All signs, including contrary/uncertain findings, must be reported.']
 (A/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8');print(json.dumps(report['summary'],indent=2))
 log=R/'EXECUTION_LOG.md'
 with log.open('a',encoding='utf-8') as f:f.write('\n'+datetime.datetime.now().isoformat()+' — Explanation extension complete and locally verified: '+json.dumps(summary)+'; full report in '+str(A)+'. All seeds retained; independent four-comparison adjustment. No new method-superiority claim.\n')
if __name__=='__main__':main()
