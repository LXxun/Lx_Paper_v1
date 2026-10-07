"""Matched MOSI TEI fusion ablation with end-to-end BERT; no test data."""
import os,json,time,argparse
import run_controlled as c
import numpy as np,torch
from transformers import BertModel
ROOT=c.ROOT
import sys,types,importlib.util
spec=importlib.util.spec_from_file_location("structural_model",ROOT/"structural_stage/source/model.py")
struct=importlib.util.module_from_spec(spec);sys.modules[spec.name]=struct;spec.loader.exec_module(struct)
BERT=os.environ.get('BERT_MODEL_DIR',str(ROOT/'models/bert-base-uncased'))
class EndToEnd(torch.nn.Module):
 def __init__(self,kind,stats,dataset="MOSI",variant="full"):
  dims=(768,5,20) if dataset=="MOSI" else (768,74,35)
  super().__init__()
  self.bert=BertModel.from_pretrained(BERT,local_files_only=True,attn_implementation='sdpa')
  self.kind=kind
  self.stats=stats
  if kind=='mult':
   cfg=json.loads((ROOT/'repos/MMSA/src/MMSA/config/config_regression.json').read_text())
   conf=dict(cfg['mult']['commonParams']);conf.update(cfg['mult']['datasetParams'][dataset.lower()]);conf.update(feature_dims=dims,train_mode='regression',num_classes=3,use_bert=False,need_data_aligned=True)
   self.head=c.load_mult()(types.SimpleNamespace(**conf));self.mult_config=conf
  else:
   self.head=struct.TEIKAN(struct.ModelConfig(input_dims=dims,bottleneck_dim=32,fusion='structured_kan' if kind=='tei_kan' else 'structured_mlp',interaction=variant!='no_interaction',deterministic=variant=='deterministic',evidence_pool=variant!='mean_pool'),stats)
 def forward(self,d,idx):
  t=d['tokens'][idx];am=t[:,1].clone();am[am.sum(1)==0,0]=1
  text=self.bert(input_ids=t[:,0],attention_mask=am,token_type_ids=t[:,2]).last_hidden_state
  if self.kind=='mult':
   xs=[text,d['audio'][idx],d['vision'][idx]]
   for m,k in enumerate(['text','audio','vision']):
    if m:xs[m]=((xs[m]-torch.as_tensor(self.stats[k+'_mean'],device=text.device))/torch.as_tensor(self.stats[k+'_std'],device=text.device)).clamp(-8,8)
    xs[m]=torch.where(d['mask'][idx,:,m,None],xs[m],torch.zeros_like(xs[m]))
   return {'reg':self.head(*xs)['M'].reshape(-1)}
  return self.head(text,d['audio'][idx],d['vision'][idx],d['mask'][idx])
def load_data(device,dataset="MOSI"):
 arrays={s:dict(np.load(ROOT/'cache'/dataset/f'{s}.npz',allow_pickle=False)) for s in ['train','valid']}
 stats={'text_mean':np.zeros(768,np.float32),'text_std':np.ones(768,np.float32)}
 for m,k in [(1,'audio'),(2,'vision')]:
  v=arrays['train'][k][arrays['train']['mask'][:,:,m]].astype(np.float64)
  mean=v.mean(0);std=v.std(0);std[std<1e-3]=1
  stats[k+'_mean']=mean.astype(np.float32);stats[k+'_std']=std.astype(np.float32)
 data={s:{k:torch.as_tensor(a[k],device=device) for k in ['tokens','audio','vision','mask','yc','yr']} for s,a in arrays.items()}
 return arrays,data,stats
def main(a):
 c.seed_all(a.seed);torch.cuda.set_device(a.gpu);torch.set_num_threads(4);device=f'cuda:{a.gpu}'
 out=ROOT/'structural_stage/runs'/a.dataset/a.model/a.variant/f'seed{a.seed}'
 if a.smoke:out=ROOT/'structural_stage/smoke'/a.dataset/a.model/a.variant
 out.mkdir(parents=True,exist_ok=False)
 arrays,data,stats=load_data(device,a.dataset);np.savez(out/'normalization.npz',**stats)
 model=EndToEnd(a.model,stats,a.dataset,a.variant).to(device)
 opt=c.make_optimizer(model.head) if a.model!='mult' else torch.optim.AdamW(model.head.parameters(),lr=model.mult_config['learning_rate'],weight_decay=model.mult_config['weight_decay'])
 criterion=lambda pred,yc,yr: torch.nn.functional.l1_loss(pred['reg'],yr) if a.model=='mult' else c.loss_components(pred,yc,yr)['total']
 config=model.mult_config if a.model=='mult' else model.head.config_dict()
 opt.add_param_group({'params':list(model.bert.parameters()),'lr':5e-5,'weight_decay':.001})
 # Dedicated sample-order generator makes order independent of model parameter count/dropout RNG.
 order=torch.Generator(device=device).manual_seed(a.seed)
 manifest=dict(model=a.model,seed=a.seed,dataset=a.dataset,variant=a.variant,regime='aligned50; full FP32 trainable same local BERT; train-only AV normalization',
  epochs=a.epochs,early_stop=8,batch_size=32,bert_lr=5e-5,bert_weight_decay=.001,head_encoder_lr=3e-4,head_fusion_lr=5e-4,
  loss='CE+0.75Huber(delta0.5)+0.1aux+1e-4KL',selection='minimum global validation raw MAE; tolerance1e-5',
  test_evaluated=False,parameter_count=sum(p.numel() for p in model.parameters()),head_config=config,
  data_manifest=json.loads((ROOT/'cache'/a.dataset/'manifest.json').read_text()),smoke_only=a.smoke)
 manifest['loss']='L1 regression' if a.model=='mult' else manifest['loss']
 manifest['regime_note']='MulT aligned input adaptation with trainable external BERT, AdamW head with author lr/wd, fixed common budget; not native paper reproduction' if a.model=='mult' else 'one structural component changed; other training settings preserved'
 manifest['head_encoder_lr']=model.mult_config['learning_rate'] if a.model=='mult' else 3e-4
 manifest['head_fusion_lr']=model.mult_config['learning_rate'] if a.model=='mult' else 5e-4
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if a.smoke:
  model.train();idx=torch.arange(4,device=device);pred=model(data['train'],idx)
  loss=criterion(pred,data['train']['yc'][idx],data['train']['yr'][idx]);loss.backward()
  g=model.bert.embeddings.word_embeddings.weight.grad
  assert torch.isfinite(loss) and g is not None and torch.isfinite(g).all() and g.abs().sum()>0
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  (out/'status.json').write_text(json.dumps(dict(status='forward_backward_update_passed',bert_gradient_norm=float(g.norm()),loss=float(loss.detach()))));return
 best=float('inf');bad=0;history=[];start=time.perf_counter()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for idx in torch.randperm(len(data['train']['yr']),device=device,generator=order).split(32):
   opt.zero_grad(set_to_none=True);pred=model(data['train'],idx)
   loss=criterion(pred,data['train']['yc'][idx],data['train']['yr'][idx])
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise RuntimeError('nonfinite gradients')
   opt.step();losses.append(float(loss.detach()))
  model.eval();preds=[]
  with torch.inference_mode():
   for idx in torch.arange(len(data['valid']['yr']),device=device).split(64):preds.append(model(data['valid'],idx)['reg'].cpu().numpy())
  p=np.concatenate(preds);metrics=c.metrics(p,arrays['valid']['yr'])
  if metrics['MAE']<best-1e-5:
   best=metrics['MAE'];bad=0
   torch.save({'model_state':model.state_dict(),'epoch':epoch,'seed':a.seed,'head_config':config},out/'best.pt')
   np.savez(out/'valid_predictions.npz',prediction=p,labels=arrays['valid']['yr'],ids=arrays['valid']['ids'])
   (out/'best_metrics.json').write_text(json.dumps(dict(epoch=epoch,**metrics),indent=2))
  else:bad+=1
  history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start,**metrics))
  (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(model=a.model,seed=a.seed,**history[-1])),flush=True)
  if bad>=8:break
 (out/'status.json').write_text(json.dumps(dict(status='completed',epochs=len(history),best_validation_MAE=best,test_evaluated=False,elapsed_seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['tei_kan','tei_mlp','mult'],required=True);p.add_argument('--seed',type=int,default=2026);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=30);p.add_argument('--dataset',choices=['MOSI','MOSEI'],required=True);p.add_argument('--variant',choices=['full','no_interaction','deterministic','mean_pool'],required=True);p.add_argument('--smoke',action='store_true');main(p.parse_args())
