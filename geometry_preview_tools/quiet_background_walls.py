from pathlib import Path
import json,shutil
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');p=R/'config.json';cfg=json.loads(p.read_text());sc=cfg['scene_by_family']
sc['V2V_SEESAW']['background_material_overrides'][0]['albedo_texture_strength']=.18
sc['V2V_PENDULUM_CABINET']['background_wall_panels']=[dict(name='left_plain_stone_wall',position=[-2.145,.14,2.76],half_size=[.025,3.62,2.75],texture='beige_wall_001',color_srgb='#CDC8BC',texture_strength=.15),dict(name='right_plain_stone_wall',position=[2.145,.14,2.76],half_size=[.025,3.62,2.75],texture='beige_wall_001',color_srgb='#CDC8BC',texture_strength=.15)]
sc['V2V_PENDULUM']['background_wall_panels']=[dict(name='plain_back_wall',position=[.2,2.44,2.5],half_size=[3.4,.025,3.6],texture='white_plaster_rough_02',color_srgb='#D5C8B0',texture_strength=.12)]
p.write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
p=R/'render.py';s=p.read_text();needle=" replaced=[]\n for index,slot in enumerate(obj.material_slots):"
insert=""" if override.get('albedo_texture_strength') is not None:
  nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF');source=bsdf.inputs['Base Color'].links[0].from_socket;mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[0].default_value=override['albedo_texture_strength'];mix.inputs[1].default_value=(*srgb(override['tint_srgb']),1);links.new(source,mix.inputs[2]);links.new(mix.outputs[0],bsdf.inputs['Base Color']);material_audit[name]['albedo_texture_strength']=override['albedo_texture_strength']
"""
assert needle in s;s=s.replace(needle,insert+needle)
needle='actors={};dynamic='
insert="""for panel in placement.get('background_wall_panels',[]):
 name='Background_'+panel['name'];mat=wood_material(name,panel['texture']);nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF');source=bsdf.inputs['Base Color'].links[0].from_socket;mix=nodes.new('ShaderNodeMixRGB');mix.blend_type='MIX';mix.inputs[0].default_value=panel['texture_strength'];mix.inputs[1].default_value=(*srgb(panel['color_srgb']),1);links.new(source,mix.inputs[2]);links.new(mix.outputs[0],bsdf.inputs['Base Color']);material_audit[name].update(kind='low-contrast textured background wall',color_srgb=panel['color_srgb'],texture_strength=panel['texture_strength'])
 obj=orig.add_actor(name,{'shape':'box','size_m':dict(zip(['hx','hy','hz'],panel['half_size'])),'initial_position_m':panel['position']},mat,edge_clarity=False);obj.matrix_world=transform@obj.matrix_world
"""
assert needle in s;s=s.replace(needle,insert+needle);p.write_text(s)
for c in ['v2v_pendulum_cabinet_h250','v2v_pendulum_l110','v2v_seesaw_x058']:
 p=R/'cases'/c;archive=p/'before_continuous_plain_walls';archive.mkdir(exist_ok=True)
 for f in ['preview_complete.json','calibration_report.json','calibration_frames/frame_0001.png']:
  src=p/f
  if src.exists():shutil.copy2(src,archive/src.name)
 (p/'preview_complete.json').unlink(missing_ok=True)
