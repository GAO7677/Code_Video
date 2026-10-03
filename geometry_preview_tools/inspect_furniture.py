import bpy,json
from pathlib import Path
from mathutils import Vector
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');cfg=json.loads((R/'config.json').read_text())
for family in ['F12_RAMP_LENGTH','V2V_DOMINO','V2V_PENDULUM','V2V_SEESAW']:
 p=cfg['scene_by_family'][family];bpy.ops.wm.open_mainfile(filepath=p['scene_blend']);bpy.context.scene.frame_set(bpy.context.scene.frame_start);bpy.context.view_layer.update();rows=[]
 for o in bpy.context.scene.objects:
  if o.type not in ['MESH','CURVE','FONT','LIGHT']:continue
  vs=[o.matrix_world@Vector(v) for v in o.bound_box];lo=[round(min(v[i] for v in vs),3) for i in range(3)];hi=[round(max(v[i] for v in vs),3) for i in range(3)];rows.append(dict(name=o.name,type=o.type,lo=lo,hi=hi,parent=o.parent.name if o.parent else None,materials=[m.name for m in o.data.materials if m] if o.type in ['MESH','CURVE','FONT'] else [],collections=[c.name for c in o.users_collection]))
 (R/'inspections'/(family+'_furniture.json')).write_text(json.dumps(rows,indent=2));print('READY',family,len(rows),flush=True)
