import bpy,json,sys,math
from pathlib import Path
from mathutils import Vector
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');cache=Path('/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes');rows=json.loads((R/'assets_catalog.json').read_text());out=R/'inspections';out.mkdir(exist_ok=True)
for a in rows:
 if '--' in sys.argv and a['assetBaseId'] not in sys.argv[sys.argv.index('--')+1:]:continue
 p=cache/a['assetBaseId']/'scene.blend'
 if not p.exists():continue
 bpy.ops.wm.open_mainfile(filepath=str(p));s=bpy.context.scene;s.frame_set(s.frame_start);bpy.context.view_layer.update();cams=[];objs=[]
 for o in s.objects:
  if o.type=='CAMERA':
   f=o.matrix_world.to_quaternion()@Vector((0,0,-1));cams.append(dict(name=o.name,position=list(o.matrix_world.translation),forward=list(f),lens=o.data.lens,active=o==s.camera))
  if o.type=='MESH' and not o.hide_render:
   vs=[o.matrix_world@Vector(v) for v in o.bound_box];lo=[min(v[i] for v in vs) for i in range(3)];hi=[max(v[i] for v in vs) for i in range(3)];size=[hi[i]-lo[i] for i in range(3)]
   if size[0]*size[1]>3:objs.append(dict(name=o.name,lo=lo,hi=hi,size=size))
 missing=[im.filepath for im in bpy.data.images if im.source=='FILE' and not im.packed_file and not Path(bpy.path.abspath(im.filepath)).exists()]
 d=dict(id=a['assetBaseId'],name=a['displayName'],cameras=cams,objects=sorted(objs,key=lambda o:o['size'][0]*o['size'][1],reverse=True)[:60],object_count=len(s.objects),missing=missing,world=s.world.name if s.world else None,exposure=s.view_settings.exposure)
 (out/(a['assetBaseId']+'.json')).write_text(json.dumps(d,indent=2));print('INSPECTED',a['displayName'],len(s.objects),'missing',len(missing),'cameras',len(cams),flush=True)
