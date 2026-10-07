import sys,json,argparse
from pathlib import Path
import numpy as np,torch
S=Path(__file__).resolve().parent;R=S.parent;sys.path.insert(0,str(R))
import run_compact_stage as r
from guard import verify
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','test'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args();verify(a.phase)
torch.set_num_threads(4);torch.cuda.set_device(a.gpu);device=f'cuda:{a.gpu}'
manifest=json.loads((S/('replay_manifest.json' if a.phase=='replay' else 'sample_manifest.json')).read_text())
jobs=[(ds,v,s) for ds in ['MOSI','MOSEI'] for v in ['main_only','main_pair','concat'] for s in [2026,2027,2028]]
for ds,v,seed in jobs[a.gpu::2]:
 path=R/'compact_stage/runs'/ds/'tei_mlp'/v/f'seed{seed}'
 data=dict(np.load(manifest[ds]['cache_path']));stats=dict(np.load(path/'normalization.npz'))
 m=r.EndToEnd('tei_mlp',stats,ds,v).to(device);ck=torch.load(path/'best.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['model_state']);del ck;m.eval().requires_grad_(False)
 d={k:torch.as_tensor(data[k],device=device) for k in ['tokens','audio','vision','mask']};scores=[]
 with torch.inference_mode():
  for idx in torch.arange(len(data['ids']),device=device).split(64):scores.append(m(d,idx)['scores'].cpu().numpy())
 scores=np.concatenate(scores);prediction=3*np.tanh(scores[:,3]/3);assert np.isfinite(scores).all()
 out=S/('replay_predictions' if a.phase=='replay' else 'predictions')/ds/v/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
 np.savez(out/'predictions.npz',ids=data['ids'],scores=scores,prediction=prediction,labels=data['yr'])
 result=dict(status='completed',samples=len(prediction),phase=a.phase,test_inference=a.phase=='test')
 if a.phase=='replay':
  prior=dict(np.load(path/'valid_predictions.npz'));assert np.array_equal(prior['ids'],data['ids']);assert np.array_equal(prior['labels'],data['yr'])
  error=float(np.max(np.abs(prediction-prior['prediction'])));assert error<1e-4,error;result['max_saved_prediction_error']=error
 else:
  from sklearn.metrics import accuracy_score,f1_score
  y=data['yr'];nz=y!=0
  result.update(MAE=float(np.abs(prediction-y).mean()),Pearson=float(np.corrcoef(prediction,y)[0,1]) if np.std(prediction)>0 and np.std(y)>0 else None,nonzero_samples=int(nz.sum()),Acc2=float(accuracy_score(y[nz]>0,prediction[nz]>0)),weighted_F1=float(f1_score(y[nz]>0,prediction[nz]>0,average='weighted',zero_division=0)))
 (out/'status.json').write_text(json.dumps(result,indent=2));print(ds,v,seed,json.dumps(result),flush=True)
 del m,d;torch.cuda.empty_cache()
