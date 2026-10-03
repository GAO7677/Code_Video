from pathlib import Path
import json,shutil
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');p=R/'config.json';shutil.copy2(p,R/'config_before_scene_simplification_v2.json');cfg=json.loads(p.read_text());sc=cfg['scene_by_family']
sc['SCENE_DOOR_FRAME_BALL']['hide_background_objects']=['Cube','Cube.001','Cube.002','Cube.003','Cube.004','Cube.005','Cylinder','Plane.001','Plane.002','Plane.003','Plane.004','Plane.005']
sc['V2V_PENDULUM_CABINET']['keep_background_geometry']=['Main hall','Side wall','Marble','Ceilling.001','Ceilling.002'];sc['V2V_PENDULUM_CABINET']['label']='简洁石材房间';sc['V2V_PENDULUM_CABINET']['fill_light_watts']=200
sc['V2V_PENDULUM']['keep_background_geometry']=[x for x in sc['V2V_PENDULUM']['keep_background_geometry'] if x not in ['Cube.002','Cube.004','Cube.005','Plane.007']]
sc['V2V_SEESAW']['background_material_overrides']=[{'object':'Wall','slot_material':'Black concreete','texture':'white_plaster_rough_02','tint_srgb':'#D5D0C4'}];sc['V2V_SEESAW']['label']='浅暖灰墙与深色地面'
p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
p=R/'render.py';s=p.read_text();needle='actors={};dynamic='
insert="""for override in placement.get('background_material_overrides',[]):
 obj=scene.objects[override['object']];name='background_'+override['object'];mat=wood_material(name,override['texture']);material_audit[name]['kind']='textured background plaster'
 if override.get('tint_srgb'):
  nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF');source=bsdf.inputs['Base Color'].links[0].from_socket;mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MULTIPLY';mix.inputs[0].default_value=1;mix.inputs[2].default_value=(*srgb(override['tint_srgb']),1);links.new(source,mix.inputs[1]);links.new(mix.outputs[0],bsdf.inputs['Base Color']);material_audit[name]['tint_srgb']=override['tint_srgb']
 replaced=[]
 for index,slot in enumerate(obj.material_slots):
  if slot.material and slot.material.name==override['slot_material']:obj.data.materials[index]=mat;replaced.append(index)
 assert replaced,('Background material slot missing',override);material_audit[name]['replaced_slots']=replaced
"""
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
redo=['scene_door_frame_ball_w054','v2v_pendulum_cabinet_h250','v2v_pendulum_l110','v2v_seesaw_x058']
for c in redo:
 p=R/'cases'/c;archive=p/'before_scene_simplification_v2';archive.mkdir(exist_ok=True)
 for f in ['preview_complete.json','calibration_report.json','calibration_frames/frame_0001.png']:
  src=p/f
  if src.exists():shutil.copy2(src,archive/src.name)
 (p/'preview_complete.json').unlink(missing_ok=True)
p=Path('/home/gaoya/Code_Video/geometry_preview_tools/publish_multiscene.py');s=p.read_text().replace("note?.status==='PASS'?", "note?.status==='PASS'&&note.image_sha256===d.sha256?");p.write_text(s)
print('Prepared 4 background refinements; task geometry unchanged')
