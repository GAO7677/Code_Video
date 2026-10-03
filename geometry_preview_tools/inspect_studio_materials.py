import bpy,json
from pathlib import Path
bpy.ops.wm.open_mainfile(filepath='/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes/7d04c6e1-3f00-47ae-9296-3ae7e1af6361/scene.blend')
for name in ['Main Room','Floor','Floor thing','Ceiling']:
 o=bpy.data.objects[name];print('STRUCTURE',name,[(s.material.name if s.material else None) for s in o.material_slots],flush=True)
