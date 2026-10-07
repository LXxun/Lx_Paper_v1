import json,hashlib
from pathlib import Path
S=Path(__file__).resolve().parent

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def verify(phase):
 assert phase in ['replay','test']
 for rel,h in json.loads((S/'CODE_LOCK.json').read_text()).items():assert digest(S/rel)==h,rel
 for path,h in json.loads((S/'ASSET_LOCK.json').read_text())['files'].items():assert digest(path)==h,path
 if phase=='test':
  lock=json.loads((S/'EXECUTION_LOCK.json').read_text());assert lock['status']=='locked_before_test_inference'
  for rel,h in lock['files'].items():assert digest(S/rel)==h,rel
 print('All code and asset hashes verified:',phase,flush=True)
