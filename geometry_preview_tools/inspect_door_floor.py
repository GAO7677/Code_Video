import bpy,json,math
from pathlib import Path
from mathutils import Matrix,Vector
r=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261005_test70_v2_approved_motion_render');p=json.loads((r/'config.json').read_text())['scene_by_family']['SCENE_DOOR_FRAME_BALL']
bpy.ops.wm.open_mainfile(filepath=p['scene_blend']);T=Matrix.Rotation(p['fixed_room_yaw'],4,'Z')@Matrix.Translation(-Vector(p['fixed_anchor']))
for o in bpy.context.scene.objects:
 if o.parent is None:o.matrix_world=T@o.matrix_world
bpy.context.view_layer.update();o=bpy.data.objects['Floor'];vs=[o.matrix_world@v.co for v in o.data.vertices];print('FLOOR',len(vs),len(o.data.polygons), 'bounds',[(min(v[i] for v in vs),max(v[i] for v in vs)) for i in range(3)],flush=True)
for poly in o.data.polygons:
 pts=[vs[i] for i in poly.vertices];lo=[min(v[i] for v in pts) for i in range(3)];hi=[max(v[i] for v in pts) for i in range(3)]
 print('FACE',poly.index,'bounds',lo,hi,'material',poly.material_index,flush=True)
eye=Vector((-3,-.65,1.95));target=Vector((3.30553,.1,.17999))
def hit():
 inv=o.matrix_world.inverted();a=inv@eye;b=inv@target;ok,loc,n,face=o.ray_cast(a,(b-a).normalized(),distance=(b-a).length);return dict(hit=ok,face=face,world=list(o.matrix_world@loc) if ok else None)
print('BEFORE',hit(),flush=True)
inv=o.matrix_world.inverted()
for v in o.data.vertices:
 w=o.matrix_world@v.co
 if w.x>2:w.x=2+3*(w.x-2);v.co=inv@w
o.data.update();bpy.context.view_layer.update();print('AFTER',hit(),flush=True)
