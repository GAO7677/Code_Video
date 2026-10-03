from pathlib import Path
import json,shutil
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');p=R/'config.json';cfg=json.loads(p.read_text());sc=cfg['scene_by_family']
rows=json.loads((R/'inspections/F12_RAMP_LENGTH_furniture.json').read_text())
sc['F12_RAMP_LENGTH']['hide_background_objects']=[o['name'] for o in rows if o['type'] in ['MESH','CURVE','FONT'] and o['lo'][0]>-2.2 and o['hi'][0]<-.3 and o['hi'][2]<1.9]
sc['F12_RAMP_LENGTH']['preserve_background_objects']=['Cube.013','Cube.014','Cube.015','Cube.016','Cube.017']
sc['V2V_DOMINO']['hide_background_objects']=[f'Cube.{i:03d}' for i in range(7,19)]
sc['V2V_DOMINO']['camera_adjustment']={'position_add':[0,0,0],'target_add':[0,0,-.15],'yfov':34}
sc['V2V_PENDULUM']['keep_background_geometry']=['Main house','Window','Window.001','CTRL_Hole','Frame','Pole','Cube.002','Cube.004','Cube.005','Cube.006','Cube.039','Cube.040','Plane.007']
rows=json.loads((R/'inspections/V2V_SEESAW_furniture.json').read_text());sofa=[o['name'] for o in rows if o['lo'][0]>-3 and o['hi'][0]<1.6 and o['lo'][1]>3 and o['hi'][2]<1.2 and o['name'] not in ['Plane.001','Plane.004','Plane.005']]
sc['V2V_SEESAW']['keep_background_geometry']=['Wall','Floor']+sofa
sc['V2V_PENDULUM']['label']='简洁木墙挑高室内';sc['V2V_SEESAW']['label']='简洁深色沙发厅'
p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2));p=R/'render.py';s=p.read_text();needle='removed_furniture=[]\n';replacement="""removed_furniture=[]
keep=placement.get('keep_background_geometry');hide=set(placement.get('hide_background_objects',[]))
for obj in list(scene.objects):
 if obj.type in ['MESH','CURVE','FONT'] and (obj.name in hide or (keep is not None and obj.name not in keep)):
  obj.hide_render=True;obj.hide_set(True);removed_furniture.append(obj.name)
for name in placement.get('preserve_background_objects',[]):
 assert name in scene.objects and not scene.objects[name].hide_render,'Required sofa object missing: '+name
bpy.context.view_layer.update()
"""
assert needle in s;s=s.replace(needle,replacement);p.write_text(s)
redo=['difficulty_l2_f12_length_l140','v2v_domino_g090','v2v_pendulum_l110','v2v_seesaw_x058']
for c in redo:
 p=R/'cases'/c;archive=p/'before_user_declutter';archive.mkdir(exist_ok=True)
 for f in ['preview_complete.json','calibration_report.json','calibration_frames/frame_0001.png']:
  src=p/f
  if src.exists():shutil.copy2(src,archive/src.name)
 (p/'preview_complete.json').unlink(missing_ok=True)
print('Coffee removed',len(sc['F12_RAMP_LENGTH']['hide_background_objects']),'sofa kept; domino chairs removed; simplified final two backgrounds')
