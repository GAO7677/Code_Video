from pathlib import Path
import json,numpy as np,hashlib,urllib.request
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
checked=[]
for name in ['lighting','slope']:
 r=D.parent.parent/('20261003_test70_v2_'+name+'_revision');state=json.loads((r/'revision_status.json').read_text());assert state['stage']=='PUBLISHED',state
 for c in state['cases']:
  old=r/'previous_version'/c;new=D/'samples'/c
  a=np.load(old/'raw/trajectories.npz');b=np.load(new/'raw/trajectories.npz');assert set(a.files)==set(b.files)
  assert all(np.array_equal(a[k],b[k]) for k in a.files),c
  assert json.loads((old/'camera.json').read_text())==json.loads((new/'camera.json').read_text()),c
  assert (old/'contacts.json').read_bytes()==(new/'contacts.json').read_bytes(),c
  record=json.loads((new/'complete.json').read_text());assert record['qa_warning_count']==0
  assert hashlib.sha256((new/'videos/rgb_cycles.mp4').read_bytes()).hexdigest()==record['video_sha256']
  checked.append(c)
manifest=json.loads((D/'manifest.json').read_text());assert manifest['completed_count']==70
page=urllib.request.urlopen('http://127.0.0.1:8844/test70_v2/',timeout=10).read().decode();assert '暖色素墙木地板住宅' in page and '?v=' in page
report=dict(status='PASS',revised_cases=checked,total_complete=70,trajectories_unchanged=True,cameras_unchanged=True,contacts_unchanged=True,visibility_warnings=0,http_page_updated=True)
(D/'reports/scene_revision_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
