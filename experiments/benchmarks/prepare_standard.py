"""Hash-gated standard data audit; preserve SDK video folds; prepare train/valid only."""
import os,sys,json,pickle,ast,hashlib,importlib.util,argparse
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.path.insert(0,str(ROOT/'source'))
HASHES={'MOSI':{'aligned_50.pkl':'d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9','unaligned_50.pkl':'78e0f8b5ef8ff71558e7307848fc1fa929ecb078203f565ab22b9daab2e02524'},
 'MOSEI':{'aligned_50.pkl':'45eccfb748a87c80ecab9bfac29582e7b1466bf6605ff29d3b338a75120bf791'}}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
class Restricted(pickle.Unpickler):
 def find_class(self,module,name):
  allowed={('numpy.core.multiarray','_reconstruct'),('numpy','ndarray'),('numpy','dtype'),('numpy.core.multiarray','scalar')}
  if (module,name) not in allowed:raise ValueError('Unapproved pickle global '+module+'.'+name)
  return super().find_class(module,name)
def main(dataset):
 path=ROOT/'data'/dataset/'aligned_50.pkl'
 if not path.exists():
  part=path.with_suffix('.pkl.part')
  assert part.exists()
  assert digest(part)==HASHES[dataset]['aligned_50.pkl'],'official SHA256 mismatch or incomplete download'
  part.rename(path)
 assert digest(path)==HASHES[dataset]['aligned_50.pkl']
 with path.open('rb') as f:raw=Restricted(f).load()
 folds_path=ROOT/'repos/CMU-MultimodalSDK/mmsdk/mmdatasdk/dataset/standard_datasets'/('CMU_'+dataset)/('cmu_'+dataset.lower()+'_std_folds.py')
 folds={}
 for node in ast.parse(folds_path.read_text()).body:
  if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
   folds[node.targets[0].id]=ast.literal_eval(node.value)
 records={};ids_by={};videos_by={}
 for split in ['train','valid','test']:
  d=raw[split];ids=np.asarray(d['id']).astype(str).reshape(-1);videos=np.array([v.split('$_$')[0] for v in ids])
  ids_by[split]=set(ids);videos_by[split]=set(videos)
  assert len(ids_by[split])==len(ids)
  outside=videos_by[split]-set(folds['standard_'+split+'_fold'])
  # Official MMSA hash-verified MOSEI contains this single training video absent from the pinned SDK lists.
  permitted={'-NFrJFQijFE'} if dataset=='MOSEI' and split=='train' else set()
  assert outside<=permitted,(split,'unexpected SDK fold mismatch',outside)
  for other in ['train','valid','test']:
   if other!=split:assert not(videos_by[split]&set(folds['standard_'+other+'_fold'])),(split,other,'cross-fold overlap')
  records[split]={'samples':len(ids),'videos':len(set(videos)),'official_fold_match':not bool(outside),'sdk_unlisted_videos':sorted(outside),'sdk_unlisted_samples':int(np.isin(videos,list(outside)).sum()),'fields':{k:list(np.asarray(v).shape) for k,v in d.items()}}
 for x,y in [('train','valid'),('train','test'),('valid','test')]:
  assert not(ids_by[x]&ids_by[y]) and not(videos_by[x]&videos_by[y])
 cache=ROOT/'cache'/dataset;cache.mkdir(parents=True,exist_ok=False)
 # Standard file includes test labels, but only test IDs/shapes are audited; no test predictions or scores.
 from transformers import BertModel
 bert=Path(os.environ.get('BERT_MODEL_DIR',str(ROOT/'models/bert-base-uncased')))
 enc=BertModel.from_pretrained(bert,local_files_only=True,attn_implementation='sdpa').eval().to('cuda:0')
 enc.requires_grad_(False);torch.set_num_threads(4)
 for split in ['train','valid']:
  d=raw[split];tokens=np.asarray(d['text_bert'],dtype=np.int64)
  audio=np.asarray(d['audio'],dtype=np.float32);vision=np.asarray(d['vision'],dtype=np.float32)
  assert tokens.shape[1:]==(3,50) and audio.shape[:2]==vision.shape[:2]==(len(tokens),50)
  assert np.isfinite(audio).all() and np.isfinite(vision).all()
  mask=np.stack([(tokens[:,1]==1)&~np.isin(tokens[:,0],[0,101,102]),np.any(audio!=0,axis=-1),np.any(vision!=0,axis=-1)],-1)
  features=[]
  with torch.inference_mode():
   for lo in range(0,len(tokens),128):
    t=torch.as_tensor(tokens[lo:lo+128],device='cuda:0')
    am=t[:,1].clone();am[am.sum(1)==0,0]=1
    h=enc(input_ids=t[:,0],attention_mask=am,token_type_ids=t[:,2]).last_hidden_state
    features.append(h.cpu().numpy())
  text=np.concatenate(features).astype(np.float32)
  yr=np.asarray(d['regression_labels'],np.float32).reshape(-1)
  assert np.isfinite(yr).all() and np.abs(yr).max()<=3
  yc=(np.sign(yr)+1).astype(np.int64)
  np.savez(cache/(split+'.npz'),text=text,audio=audio,vision=vision,tokens=tokens,mask=mask,yr=yr,yc=yc,ids=np.asarray(d['id']).astype(str).reshape(-1))
  records[split]['all_zero_observed_modalities']=[int((~mask[:,:,m].any(1)).sum()) for m in range(3)]
  records[split]['cache_sha256']=digest(cache/(split+'.npz'))
  print(split,records[split]['samples'],flush=True)
 report={'dataset':dataset,'data_sha256':digest(path),'official_hash_verified':True,'fold_source':str(folds_path),'splits':records,
  'features':'same local frozen bert-base-uncased FP32 for every controlled model; original supplied audio/vision; aligned50',
  'test_status':'metadata fold/shape check only; no test predictions, metrics, model selection',
  'bert_weight_sha256':digest(bert/'model.safetensors') if (bert/'model.safetensors').exists() else digest(bert/'pytorch_model.bin')}
 (cache/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 (ROOT/'reports').mkdir(exist_ok=True)
 (ROOT/'reports'/(dataset+'_data_audit.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2))
 print('AUDIT AND CACHE COMPLETE',dataset)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['MOSI','MOSEI'],required=True);a=p.parse_args();main(a.dataset)
