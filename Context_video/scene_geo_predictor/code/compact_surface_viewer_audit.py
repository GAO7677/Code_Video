"""Keep large per-pixel evidence in frozen mesh, not video startup payloads."""
import json
from pathlib import Path
import sys
from context_rgb_pybullet_common import dump_json
root=Path(sys.argv[1])
data=json.loads((root/'viewer_data.json').read_text())
for row in data['records']:
    audit=row.get('completion_audit')
    if audit:
        for component in audit.get('components',[]):
            component.pop('source_pixels_xy',None)
        audit['full_audit_url']=f'estimates/{row["id"]}/collision_primitive.json'
    dump_json(root/'cases'/f'{row["id"]}.json',row)
dump_json(root/'viewer_data.json',data)
print('clip_000 bytes:',(root/'cases/clip_000.json').stat().st_size)
