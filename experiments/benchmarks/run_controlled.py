"""Controlled aligned50 frozen-BERT pilot. This is not a native-paper reproduction."""
import os,sys,json,time,argparse,types,importlib.util,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(ROOT/'source'))
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from tei_kan.model import TEIKAN,ModelConfig
from tei_kan.losses import loss_components
from tei_kan.training import make_optimizer
from tei_kan.data import seed_all
def namespace(name,path):
 m=types.ModuleType(name);m.__path__=[str(path)];sys.modules[name]=m
def load_mult():
 # Load just reviewed architecture modules, avoiding optional CLI/legacy-model dependencies.
 base=ROOT/'repos/MMSA/src/MMSA'
 for name,path in [('MMSA',base),('MMSA.models',base/'models'),('MMSA.models.singleTask',base/'models/singleTask')]:namespace(name,path)
 from MMSA.models.singleTask.MULT import MULT
 return MULT
def metrics(pred,y):
 p=np.asarray(pred).reshape(-1);y=np.asarray(y).reshape(-1);keep=y!=0
 return dict(MAE=float(np.abs(p-y).mean()),Corr=float(np.corrcoef(p,y)[0,1]) if np.std(p)>1e-12 else None,
  Mult_acc_7=float(np.mean(np.round(np.clip(p,-3,3))==np.round(np.clip(y,-3,3)))),
  Has0_acc_2=float(accuracy_score(y>=0,p>=0)),Has0_F1_score=float(f1_score(y>=0,p>=0,average='weighted',zero_division=0)),
  Non0_acc_2=float(accuracy_score(y[keep]>0,p[keep]>0)),
  Non0_F1_score=float(f1_score(y[keep]>0,p[keep]>0,average='weighted',zero_division=0)))
def main(args):
 seed_all(args.seed);torch.cuda.set_device(args.gpu);device=f'cuda:{args.gpu}'
 out=ROOT/'runs'/args.dataset/args.model/f'seed{args.seed}';out.mkdir(parents=True,exist_ok=False)
 cache=ROOT/'cache'/args.dataset
 arrays={s:dict(np.load(cache/(s+'.npz'),allow_pickle=False)) for s in ['train','valid']}
 dims=tuple(arrays['train'][k].shape[-1] for k in ['text','audio','vision'])
 stats={'text_mean':np.zeros(dims[0],np.float32),'text_std':np.ones(dims[0],np.float32)}
 for m,k in [(1,'audio'),(2,'vision')]:
  values=arrays['train'][k][arrays['train']['mask'][:,:,m]].astype(np.float64)
  mean=values.mean(0);std=values.std(0);std[std<1e-3]=1
  stats[k+'_mean']=mean.astype(np.float32);stats[k+'_std']=std.astype(np.float32)
 np.savez(out/'normalization.npz',**stats)
 data={}
 for split,a in arrays.items():
  data[split]={k:torch.as_tensor(a[k],device=device) for k in ['text','audio','vision','mask','yc','yr']}
 if args.model.startswith('tei'):
  conf=ModelConfig(input_dims=dims,bottleneck_dim=32,fusion='structured_mlp' if args.model=='tei_mlp' else 'structured_kan')
  model=TEIKAN(conf,stats).to(device)
  opt=make_optimizer(model)
  config=model.config_dict()
 else:
  allconf=json.loads((ROOT/'repos/MMSA/src/MMSA/config/config_regression.json').read_text())
  c=dict(allconf['mult']['commonParams']);c.update(allconf['mult']['datasetParams'][args.dataset.lower()])
  c.update(feature_dims=dims,train_mode='regression',num_classes=3,use_bert=False,need_data_aligned=True)
  model=load_mult()(types.SimpleNamespace(**c)).to(device)
  opt=torch.optim.Adam(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
  config=c
  # Match train-only input scaling/masks with TEI; this is a controlled aligned adaptation.
  for split in data:
   for m,k in enumerate(['text','audio','vision']):
    x=data[split][k].float()
    if m:x=((x-torch.as_tensor(stats[k+'_mean'],device=device))/torch.as_tensor(stats[k+'_std'],device=device)).clamp(-8,8)
    data[split][k]=torch.where(data[split]['mask'][:,:,m,None],x,torch.zeros_like(x))
  scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.1,patience=c['patience'])
 manifest=dict(model=args.model,dataset=args.dataset,seed=args.seed,config=config,
  parameter_count=sum(p.numel() for p in model.parameters()),regime='controlled aligned50 frozen local FP32 BERT; train-only AV normalization',
  selection='minimum validation raw regression MAE; max30epochs earlystop8; no class consistency postprocessing',
  loss='original TEI CE+0.75Huber+0.1aux+1e-4KL' if args.model.startswith('tei') else 'L1 regression',
  test_evaluated=False,source_manifest=json.loads((ROOT/'reports/repository_versions.json').read_text()))
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 def forward(d,idx):
  xs=[d[k][idx] for k in ['text','audio','vision']]
  if args.model.startswith('tei'):return model(*xs,d['mask'][idx])
  return {'reg':model(*xs)['M'].reshape(-1)}
 best=float('inf');bad=0;history=[];start=time.perf_counter()
 for epoch in range(1,args.epochs+1):
  model.train();permutation=torch.randperm(len(data['train']['yr']),device=device)
  losses=[]
  for idx in permutation.split(32):
   opt.zero_grad(set_to_none=True);pred=forward(data['train'],idx)
   loss=loss_components(pred,data['train']['yc'][idx],data['train']['yr'][idx])['total'] if args.model.startswith('tei') else torch.nn.functional.l1_loss(pred['reg'],data['train']['yr'][idx])
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1. if args.model.startswith('tei') else config['grad_clip']);opt.step()
   losses.append(float(loss.detach()))
  model.eval();preds=[]
  with torch.inference_mode():
   for idx in torch.arange(len(data['valid']['yr']),device=device).split(128):preds.append(forward(data['valid'],idx)['reg'].cpu().numpy())
  p=np.concatenate(preds);report=metrics(p,arrays['valid']['yr'])
  if not args.model.startswith('tei'):scheduler.step(report['MAE'])
  improved=report['MAE']<best-1e-5
  if improved:
   best=report['MAE'];bad=0
   torch.save({'model_state':model.state_dict(),'model_config':config,'epoch':epoch,'seed':args.seed},out/'best.pt')
   np.savez(out/'valid_predictions.npz',prediction=p,labels=arrays['valid']['yr'],ids=arrays['valid']['ids'])
   (out/'best_metrics.json').write_text(json.dumps(dict(epoch=epoch,**report),indent=2))
  else:bad+=1
  row=dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start,**report)
  history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2))
  print(json.dumps(dict(model=args.model,seed=args.seed,**row)),flush=True)
  if bad>=8:break
 (out/'status.json').write_text(json.dumps(dict(status='completed',epochs=len(history),best_validation_MAE=best,elapsed_seconds=time.perf_counter()-start,test_evaluated=False),indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--dataset',choices=['MOSI','MOSEI'],required=True)
 ap.add_argument('--model',choices=['tei_kan','tei_mlp','mult'],required=True)
 ap.add_argument('--seed',type=int,default=2026);ap.add_argument('--gpu',type=int,default=0);ap.add_argument('--epochs',type=int,default=30)
 main(ap.parse_args())
