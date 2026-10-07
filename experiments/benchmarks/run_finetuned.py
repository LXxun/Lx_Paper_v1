"""Matched MOSI TEI fusion ablation with end-to-end BERT; no test data."""
import os,json,time,argparse
import run_controlled as c
import numpy as np,torch
from transformers import BertModel
ROOT=c.ROOT
BERT=os.environ.get('BERT_MODEL_DIR',str(ROOT/'models/bert-base-uncased'))
class EndToEnd(torch.nn.Module):
 def __init__(self,kind,stats):
  super().__init__()
  self.bert=BertModel.from_pretrained(BERT,local_files_only=True,attn_implementation='sdpa')
  self.head=c.TEIKAN(c.ModelConfig(input_dims=(768,5,20),bottleneck_dim=32,fusion='structured_kan' if kind=='tei_kan' else 'structured_mlp'),stats)
 def forward(self,d,idx):
  t=d['tokens'][idx];am=t[:,1].clone();am[am.sum(1)==0,0]=1
  text=self.bert(input_ids=t[:,0],attention_mask=am,token_type_ids=t[:,2]).last_hidden_state
  return self.head(text,d['audio'][idx],d['vision'][idx],d['mask'][idx])
def load_data(device):
 arrays={s:dict(np.load(ROOT/'cache/MOSI'/f'{s}.npz',allow_pickle=False)) for s in ['train','valid']}
 stats={'text_mean':np.zeros(768,np.float32),'text_std':np.ones(768,np.float32)}
 for m,k in [(1,'audio'),(2,'vision')]:
  v=arrays['train'][k][arrays['train']['mask'][:,:,m]].astype(np.float64)
  mean=v.mean(0);std=v.std(0);std[std<1e-3]=1
  stats[k+'_mean']=mean.astype(np.float32);stats[k+'_std']=std.astype(np.float32)
 data={s:{k:torch.as_tensor(a[k],device=device) for k in ['tokens','audio','vision','mask','yc','yr']} for s,a in arrays.items()}
 return arrays,data,stats
def main(a):
 c.seed_all(a.seed);torch.cuda.set_device(a.gpu);torch.set_num_threads(4);device=f'cuda:{a.gpu}'
 out=ROOT/'runs_finetuned/MOSI'/a.model/f'seed{a.seed}'
 if a.smoke:out=ROOT/'smoke_finetuned'/a.model
 out.mkdir(parents=True,exist_ok=False)
 arrays,data,stats=load_data(device);np.savez(out/'normalization.npz',**stats)
 model=EndToEnd(a.model,stats).to(device)
 opt=c.make_optimizer(model.head)
 opt.add_param_group({'params':list(model.bert.parameters()),'lr':5e-5,'weight_decay':.001})
 # Dedicated sample-order generator makes order independent of model parameter count/dropout RNG.
 order=torch.Generator(device=device).manual_seed(a.seed)
 manifest=dict(model=a.model,seed=a.seed,dataset='MOSI',regime='aligned50; full FP32 trainable same local BERT; train-only AV normalization',
  epochs=a.epochs,early_stop=8,batch_size=32,bert_lr=5e-5,bert_weight_decay=.001,head_encoder_lr=3e-4,head_fusion_lr=5e-4,
  loss='CE+0.75Huber(delta0.5)+0.1aux+1e-4KL',selection='minimum global validation raw MAE; tolerance1e-5',
  test_evaluated=False,parameter_count=sum(p.numel() for p in model.parameters()),head_config=model.head.config_dict(),
  data_manifest=json.loads((ROOT/'cache/MOSI/manifest.json').read_text()),smoke_only=a.smoke)
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
 if a.smoke:
  model.train();idx=torch.arange(4,device=device);pred=model(data['train'],idx)
  loss=c.loss_components(pred,data['train']['yc'][idx],data['train']['yr'][idx])['total'];loss.backward()
  g=model.bert.embeddings.word_embeddings.weight.grad
  assert torch.isfinite(loss) and g is not None and torch.isfinite(g).all() and g.abs().sum()>0
  torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
  (out/'status.json').write_text(json.dumps(dict(status='forward_backward_update_passed',bert_gradient_norm=float(g.norm()),loss=float(loss.detach()))));return
 best=float('inf');bad=0;history=[];start=time.perf_counter()
 for epoch in range(1,a.epochs+1):
  model.train();losses=[]
  for idx in torch.randperm(len(data['train']['yr']),device=device,generator=order).split(32):
   opt.zero_grad(set_to_none=True);pred=model(data['train'],idx)
   loss=c.loss_components(pred,data['train']['yc'][idx],data['train']['yr'][idx])['total']
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
   torch.save({'model_state':model.state_dict(),'epoch':epoch,'seed':a.seed,'head_config':model.head.config_dict()},out/'best.pt')
   np.savez(out/'valid_predictions.npz',prediction=p,labels=arrays['valid']['yr'],ids=arrays['valid']['ids'])
   (out/'best_metrics.json').write_text(json.dumps(dict(epoch=epoch,**metrics),indent=2))
  else:bad+=1
  history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_seconds=time.perf_counter()-start,**metrics))
  (out/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(dict(model=a.model,seed=a.seed,**history[-1])),flush=True)
  if bad>=8:break
 (out/'status.json').write_text(json.dumps(dict(status='completed',epochs=len(history),best_validation_MAE=best,test_evaluated=False,elapsed_seconds=time.perf_counter()-start),indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['tei_kan','tei_mlp'],required=True);p.add_argument('--seed',type=int,default=2026);p.add_argument('--gpu',type=int,default=0);p.add_argument('--epochs',type=int,default=30);p.add_argument('--smoke',action='store_true');main(p.parse_args())
