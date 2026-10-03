from pathlib import Path
import json
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');p=R/'config.json';cfg=json.loads(p.read_text());d=cfg['scene_by_family']
d['SCENE_DOOR_FRAME_BALL'].update(name='Minimal Interior Wall Art Mockup',label='暖色挂画客厅',asset_id='c675b544-6e32-42fd-b840-f2fdd964f610',scene_blend='/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes/c675b544-6e32-42fd-b840-f2fdd964f610/scene.blend')
d['SCENE_DOOR_FRAME'].update(exposure=.65,fill_light_watts=650,fill_light_size=4.5,fill_light_color=[1,.85,.68])
cfg['actor_materials']['SCENE_DOOR_FRAME']['support_wood']='oak_veneer_01'
cfg['actor_materials']['SCENE_DOOR_FRAME_BALL']['support_wood']='wood_table_001'
d['V2V_DOMINO']['task_origin_offset']=[0,0,.8]
d['V2V_DOMINO']['camera_adjustment']={'position_add':[0,-.6,0],'target_add':[0,0,-.25],'yfov':43}
d['V2V_DOMINO']['tabletop']={'height':.8,'thickness':.055,'bounds_xy':[-2.365,-.90,1.996,.90],'legs':6,'material':'oak_veneer_01','coverage':'All 5 source domino cases, full 90-frame actor bounding spheres plus >=0.35 m margins'}
d['V2V_PENDULUM']['fill_light_watts']=500;d['V2V_PENDULUM']['exposure']=.3
p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
p=R/'render.py';s=p.read_text()
old="placement=cfg['scene_by_family'][family];cfg['scene_blend']=placement['scene_blend']"
new="""placement=cfg['scene_by_family'][family];cfg['scene_blend']=placement['scene_blend']
# A rigid translation maps the domino support plane onto a finite supported tabletop.
offset=Vector(placement.get('task_origin_offset',[0,0,0]))
for key,values in tr.items():
 if key.endswith('_positions'):tr[key]=[list(Vector(v)+offset) for v in values]
for key in ['position','target']:cam[key]=list(Vector(cam[key])+offset)
for key in ['position','target']:cam[key]=list(Vector(cam[key])+Vector(placement.get('camera_adjustment',{}).get(key+'_add',[0,0,0])))
cam['yfov']=placement.get('camera_adjustment',{}).get('yfov',cam['yfov'])"""
assert old in s;s=s.replace(old,new);s=s.replace("scene.view_settings.exposure=cfg['exposure']", "scene.view_settings.exposure=placement.get('exposure',cfg['exposure'])")
needle="bpy.ops.object.camera_add(location=cam['position']);"
insert="""if placement.get('tabletop'):
 table=placement['tabletop'];x0,y0,x1,y1=table['bounds_xy'];z=table['height'];th=table['thickness'];material=wood_material('domino_table',table['material'])
 def table_box(name,pos,half):
  return orig.add_actor(name,{'shape':'box','size_m':dict(zip(['hx','hy','hz'],half)),'initial_position_m':pos,'initial_orientation_wxyz':[1,0,0,0]},material,edge_clarity=False)
 table_box('Task_domino_table_top',[(x0+x1)/2,(y0+y1)/2,z-th/2],[(x1-x0)/2,(y1-y0)/2,th/2])
 for i,x in enumerate([x0+.16,(x0+x1)/2,x1-.16]):
  for j,y in enumerate([y0+.16,y1-.16]):table_box(f'Task_domino_table_leg_{i}_{j}',[x,y,(z-th)/2],[.055,.055,(z-th)/2])
"""
assert needle in s;s=s.replace(needle,insert+needle)
needle="frames=P/('calibration_frames'"
insert="""if placement.get('fill_light_watts'):
 direction=(Vector(cam['target'])-Vector(cam['position'])).normalized();side=direction.cross(Vector((0,0,1))).normalized()
 bpy.ops.object.light_add(type='AREA',location=Vector(cam['position'])-side*1.8+Vector((0,0,1.6)));fill=bpy.context.object;fill.name='Task_soft_fill';fill.data.energy=placement['fill_light_watts'];fill.data.shape='DISK';fill.data.size=placement.get('fill_light_size',4.5);fill.data.color=placement.get('fill_light_color',[1,.88,.72]);orig.point_at(fill,cam['target'])
"""
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
print('Updated replacement, warm-house lighting, and supported domino tabletop')
