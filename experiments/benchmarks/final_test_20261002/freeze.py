import json
from pathlib import Path
from datetime import datetime,timezone
from guard import verify,digest
S=Path(__file__).resolve().parent
verify('replay');report=json.loads((S/'REPLAY_REPORT.json').read_text());assert report['status']=='passed' and report['models']==18
assert not (S/'EXECUTION_LOCK.json').exists()
files={f.name:digest(f) for f in S.iterdir() if f.is_file() and f.suffix in ['.py','.json','.md'] and f.name!='EXECUTION_LOCK.json'}
for dirname in ['replay_explanations','replay_predictions']:
 for f in (S/dirname).rglob('*'):
  if f.is_file():files[str(f.relative_to(S))]=digest(f)
(S/'EXECUTION_LOCK.json').write_text(json.dumps(dict(status='locked_before_test_inference',created_utc=datetime.now(timezone.utc).isoformat(),test_inference=False,files=files,asset_lock='ASSET_LOCK.json',replay='REPLAY_REPORT.json'),indent=2))
print('EXECUTION LOCK CREATED; TEST NOT STARTED')
