from pathlib import Path
import json,shutil
B=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer');M=B/'20261005_test70_v2_approved_motion_render';R=B/'20261006_test70_v2_flat_ramp_render';F=B/'20261006_test70_v2_door_room_fix'
F.mkdir(exist_ok=True)
for d in ['samples','source_cases','logs','reports','testjsons/v2v_jsons/test70_v2_all_cycles']:(F/d).mkdir(parents=True,exist_ok=True)
for n in ['render.py','egl_guard.py','fit_room.py','run_test70_v2.py','parallel_config.json']:shutil.copy2(M/n,F/n)
shutil.copy2(R/'run_render.py',F/'run_render.py')
cases=[c for c in json.loads((M/'case_order.json').read_text()) if c.startswith('scene_door_frame_ball_')]
for c in cases:
 shutil.copytree(M/'source_cases'/c,F/'source_cases'/c,dirs_exist_ok=True);(F/'samples'/c).mkdir(exist_ok=True)
def put(n,v):(F/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
put('case_order.json',cases);cfg=json.loads((M/'config.json').read_text());cfg['qa_frames']=[1,85,90];cfg['scene_by_family']['SCENE_DOOR_FRAME_BALL']['background_mesh_extension']={'object':'Floor','axis':0,'threshold_m':2.0,'scale':3.0,'reason':'Extend native rear wall and floor beyond the unchanged ball swept volume; same correction for all five widths.'};put('config.json',cfg)
put('queue_config.json',dict(assignments={c:1 for c in cases},calibration_case=cases[-1],calibration_gpu=1,max_workers=1))
p=F/'render.py';s=p.read_text();needle='# Rotate any environment texture together with the room.'
s=s.replace(needle,'''# Extend the native room behind the task without changing physics or camera.
if placement.get('background_mesh_extension'):
 ext=placement['background_mesh_extension'];bpy.context.view_layer.update();obj=scene.objects[ext['object']];inv=obj.matrix_world.inverted();axis=ext['axis'];threshold=ext['threshold_m']
 for vertex in obj.data.vertices:
  point=obj.matrix_world@vertex.co
  if point[axis]>threshold:point[axis]=threshold+ext['scale']*(point[axis]-threshold);vertex.co=inv@point
 obj.data.update();bpy.context.view_layer.update()
'''+needle);p.write_text(s)
# Generic gallery should retain the actual family and variable.
p=F/'run_render.py';s=p.read_text().replace("family='V2V_OBSTACLE'","family=m['family_key']").replace("variable='initial_speed_mps'","variable=m['control']['variable']");p.write_text(s)
(F/'README.md').write_text('门洞球组背景远墙延伸修复。保持五例物理、轨迹、门宽、相机和材质不变；GPU1串行，先验证首/85/90帧及全部90帧可见性。\n')
print(F)
