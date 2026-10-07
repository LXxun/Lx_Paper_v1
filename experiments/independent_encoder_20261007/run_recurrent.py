"""Fixed recurrent-encoder extension: train/valid only; no test loader."""
from pathlib import Path
import sys,json,time,hashlib,argparse,types,os
BASE=Path(__file__).resolve().parents[1]/'benchmarks'
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import run_compact_stage as b
import numpy as np, torch
from transformers import BertTokenizerFast
from tei_kan.faithfulness import evidence_units,select_span

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(p,x): p.write_text(json.dumps(x,indent=2),encoding='utf-8')
class Recurrent(torch.nn.Module):
 def __init__(self):
  super().__init__();self.embeddings=torch.nn.Module();self.embeddings.word_embeddings=torch.nn.Embedding(30522,128,padding_idx=0)
  self.rnn=torch.nn.LSTM(128,384,batch_first=True,bidirectional=True)
 def forward(self,input_ids,attention_mask,token_type_ids=None):
  lengths=attention_mask.sum(1).clamp(min=1).cpu();x=self.embeddings.word_embeddings(input_ids)
  packed=torch.nn.utils.rnn.pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
  encoded,_=self.rnn(packed);encoded,_=torch.nn.utils.rnn.pad_packed_sequence(encoded,batch_first=True,total_length=input_ids.shape[1])
  return types.SimpleNamespace(last_hidden_state=encoded)
class Model(b.EndToEnd):
 def __init__(self,stats,dataset):
  torch.nn.Module.__init__(self);self.kind='tei_mlp';self.stats=stats;self.bert=Recurrent()
  dims=(768,5,20) if dataset=='MOSI' else (768,74,35)
  self.head=b.struct.TEIKAN(b.struct.ModelConfig(input_dims=dims,bottleneck_dim=32,fusion='structured_mlp',interaction=False,deterministic=True,evidence_pool=False),stats)

def init(seed=2026):
 b.c.seed_all(seed);torch.set_num_threads(4);torch.cuda.set_device(0)

def optimizer(model):
 opt=b.c.make_optimizer(model.head);opt.add_param_group({'params':list(model.bert.parameters()),'lr':.001,'weight_decay':.001});return opt

def preflight():
 init();rows=[];sources={}
 assert ROOT.parent==BASE.parent
 tok=BertTokenizerFast.from_pretrained(b.BERT,local_files_only=True)
 assert tok.vocab_size==30522
 assert select_span([[0],[1],[2]],np.array([-3.,-2.,-1.]),.2)==[2]
 assert select_span([[0],[1],[2]],np.zeros(3),.5)==[0,1]
 assert select_span([[0,1],[3],[4]],np.array([1.,1.,0.,1.,1.]),.2)==[0,1]
 for ds in ['MOSI','MOSEI']:
  arrays,data,stats=b.load_data('cuda:0',ds)
  for split,a in arrays.items():
   att=a['tokens'][:,1];assert np.all(np.isin(att,[0,1]));assert np.all(np.diff(att,axis=1)<=0);assert np.all(att.sum(1)>0)
   assert not np.any(a['tokens'][:,2]);assert not np.any(a['mask'][:,:,0]&np.isin(a['tokens'][:,0],[0,101,102]))
  assert not set(arrays['train']['ids'].tolist()) & set(arrays['valid']['ids'].tolist())
  model=Model(stats,ds).cuda();idx=torch.arange(4,device='cuda:0');model.train();out=model(data['train'],idx)
  loss=b.c.loss_components(out,data['train']['yc'][idx],data['train']['yr'][idx])['total'];loss.backward()
  grad=model.bert.embeddings.word_embeddings.weight.grad;assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum()>0
  assert torch.isfinite(loss);opt=optimizer(model);torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();model.eval()
  with torch.no_grad():
   a=model(data['valid'],idx);single=torch.cat([model(data['valid'],i[None])['scores'] for i in idx]);replay=model.head.fuse(a['evidence'].sum(2))['scores']
  batcherr=float((a['scores']-single).abs().max());fuseerr=float((a['scores']-replay).abs().max());assert batcherr<1e-4 and fuseerr<1e-4
  rows.append({'dataset':ds,'loss':float(loss.detach()),'batch_error':batcherr,'fuse_error':fuseerr,'train':len(arrays['train']['yr']),'valid':len(arrays['valid']['yr']),'test_accessed':False})
  for sp in ['train','valid']:sources[str(BASE/'cache'/ds/(sp+'.npz'))]=digest(BASE/'cache'/ds/(sp+'.npz'))
  del model,data,arrays,opt;torch.cuda.empty_cache()
 for m in list(sys.modules.values()):
  f=getattr(m,'__file__',None)
  if f and str(f).startswith(str(BASE)) and Path(f).suffix=='.py':sources[f]=digest(f)
 for f in [Path(__file__),ROOT/'PROTOCOL_V1.txt',Path(b.BERT)/'vocab.txt']:sources[str(f)]=digest(f)
 record={'status':'passed','checks':rows,'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'sources':sources,'scope':'train-validation implementation lock only; no test authorization'}
 save(ROOT/'TRAINING_LOCK.json',record);print(json.dumps(record),flush=True)

def train(ds,seed):
 lock=json.loads((ROOT/'TRAINING_LOCK.json').read_text());assert lock['status']=='passed'
 for f,h in lock['sources'].items():assert digest(f)==h,('changed',f)
 init(seed);out=ROOT/'runs'/ds/f'seed{seed}';out.mkdir(parents=True,exist_ok=False)
 arrays,data,stats=b.load_data('cuda:0',ds);model=Model(stats,ds).cuda();opt=optimizer(model);np.savez(out/'normalization.npz',**stats)
 save(out/'config.json',{'dataset':ds,'seed':seed,'encoder':'one-layer bidirectional LSTM 128->384x2; random embedding; no pretrained weights','head':model.head.config_dict(),'epochs':30,'patience':8,'batch':32,'selection':'validation raw MAE','training_lock_sha256':digest(ROOT/'TRAINING_LOCK.json'),'test_accessed':False})
 order=torch.Generator(device='cuda:0').manual_seed(seed);best=float('inf');bad=0;history=[];start=time.time()
 for epoch in range(1,31):
  model.train();losses=[]
  for idx in torch.randperm(len(data['train']['yr']),device='cuda:0',generator=order).split(32):
   opt.zero_grad(set_to_none=True);pred=model(data['train'],idx);loss=b.c.loss_components(pred,data['train']['yc'][idx],data['train']['yr'][idx])['total']
   assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
  model.eval();preds=[]
  with torch.no_grad():
   for idx in torch.arange(len(data['valid']['yr']),device='cuda:0').split(64):preds.append(model(data['valid'],idx)['reg'].cpu().numpy())
  p=np.concatenate(preds);assert np.isfinite(p).all();metrics=b.c.metrics(p,arrays['valid']['yr'])
  if metrics['MAE']<best-1e-5:
   best=metrics['MAE'];bad=0;torch.save({'model_state':model.state_dict(),'epoch':epoch,'seed':seed,'head_config':model.head.config_dict()},out/'best.pt');np.savez(out/'valid_predictions.npz',prediction=p,labels=arrays['valid']['yr'],ids=arrays['valid']['ids']);save(out/'best_metrics.json',dict(epoch=epoch,**metrics))
  else:bad+=1
  history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),elapsed_seconds=time.time()-start,**metrics));save(out/'history.json',history);print(json.dumps(history[-1]),flush=True)
  if bad>=8:break
 ck=torch.load(out/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model_state']);model.eval();preds=[]
 with torch.no_grad():
  for idx in torch.arange(len(data['valid']['yr']),device='cuda:0').split(64):preds.append(model(data['valid'],idx)['reg'].cpu().numpy())
 p=np.concatenate(preds);saved=np.load(out/'valid_predictions.npz');err=float(np.max(np.abs(p-saved['prediction'])));assert err<1e-4
 baseline=float(np.mean(np.abs(arrays['valid']['yr']-np.median(arrays['train']['yr']))));corr=float(np.corrcoef(p,arrays['valid']['yr'])[0,1]);mae=float(np.mean(np.abs(p-arrays['valid']['yr'])))
 status={'status':'completed','epochs':len(history),'seconds':time.time()-start,'best_validation_MAE':mae,'validation_Pearson':corr,'training_median_baseline_MAE':baseline,'replay_max_error':err,'predictor_usable':bool(mae<baseline and np.isfinite(corr) and corr>0 and np.std(p)>0),'test_accessed':False,'next':'Local verification before any next training or explanation run'};save(out/'COMPLETE.json',status);print(json.dumps(status),flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['preflight','train']);ap.add_argument('--dataset',choices=['MOSI','MOSEI'],default='MOSI');ap.add_argument('--seed',type=int,default=2026);a=ap.parse_args()
 try:
  if a.action=='preflight':preflight()
  else:train(a.dataset,a.seed)
 except Exception as e:
  save(ROOT/'FAILURE.json',{'action':a.action,'dataset':a.dataset,'seed':a.seed,'error':repr(e)});raise
