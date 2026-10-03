from pathlib import Path
import json,time,subprocess
r=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2_slope_revision')
for _ in range(480):
 state=json.loads((r/'revision_status.json').read_text())
 if state['stage']=='PUBLISHED':
  subprocess.run(['/home/gaoya/miniconda3/envs/sam/bin/python','/home/gaoya/Code_Video/geometry_preview_tools/validate_test70_scene_revision.py'],check=True);break
 time.sleep(15)
else:raise RuntimeError('Revision did not publish within 2 hours; inspect renderer/controller log')
