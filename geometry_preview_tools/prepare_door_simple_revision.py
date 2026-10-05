from pathlib import Path
import json,shutil
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2');T=D.parent.parent/'20261005_test70_v2_pendulum_blue_rope';R=D.parent.parent/'20261005_test70_v2_door_simple';R.mkdir(exist_ok=True)
cfg=json.loads((T/'config.json').read_text());cfg['scene_by_family']['SCENE_DOOR_FRAME']=dict(name='Simple warm gray room',label='暖灰素墙与深色哑光地面',task_label='门洞宽度（木块）',scene_blend='procedural_room',procedural_room=True,fixed_anchor=[0,0,0],forward=[0,1],fixed_room_yaw=0,fill_light_watts=900,fill_light_size=5,fill_light_color=[1,.9,.8],fill_light_side_offset=1.8,fill_light_height_offset=2.5,exposure=.1,room_floor_srgb='#454B50',room_wall_srgb='#BEB7AC')
(R/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
for n in ['fit_room.py','egl_guard.py']:shutil.copy2(T/n,R/n)
s=(T/'render.py').read_text();old="bpy.ops.wm.open_mainfile(filepath=cfg['scene_blend']);bpy.context.scene.frame_set(bpy.context.scene.frame_start)";new="""if placement.get('procedural_room'):
 bpy.ops.wm.read_factory_settings(use_empty=True);world=bpy.data.worlds.new('Warm neutral environment');world.use_nodes=True;world.node_tree.nodes['Background'].inputs['Color'].default_value=(.65,.70,.78,1);world.node_tree.nodes['Background'].inputs['Strength'].default_value=.3;bpy.context.scene.world=world
else:
 bpy.ops.wm.open_mainfile(filepath=cfg['scene_blend']);bpy.context.scene.frame_set(bpy.context.scene.frame_start)""";assert old in s;s=s.replace(old,new)
marker='actors={};dynamic=m';code="""if placement.get('procedural_room'):
 def surface_material(name,color,roughness):
  mat=bpy.data.materials.new(name);mat.use_nodes=True;n=mat.node_tree.nodes;l=mat.node_tree.links;b=n.get('Principled BSDF');b.inputs['Base Color'].default_value=(*srgb(color),1);b.inputs['Roughness'].default_value=roughness;b.inputs['Specular'].default_value=.22;co=n.new('ShaderNodeTexCoord');noise=n.new('ShaderNodeTexNoise');noise.inputs['Scale'].default_value=90;l.new(co.outputs['Object'],noise.inputs['Vector']);bump=n.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.001;l.new(noise.outputs['Fac'],bump.inputs['Height']);l.new(bump.outputs['Normal'],b.inputs['Normal']);return mat
 def room_box(name,pos,half,mat):
  return orig.add_actor(name,{'shape':'box','size_m':dict(zip(['hx','hy','hz'],half)),'initial_position_m':pos,'initial_orientation_wxyz':[1,0,0,0]},mat,edge_clarity=False)
 floor=surface_material('Matte charcoal floor',placement['room_floor_srgb'],.9);wall=surface_material('Warm gray plaster',placement['room_wall_srgb'],.85)
 room_box('Background_continuous_floor',[0,0,-.05],[12,12,.05],floor)
 room_box('Background_rear_wall',[5.5,0,3],[.06,12,3],wall)
 room_box('Background_side_wall',[0,6,3],[12,.06,3],wall)
 bpy.ops.object.light_add(type='AREA',location=(1,-2.5,5));key=bpy.context.object;key.name='Room_window_softbox';key.data.energy=1100;key.data.shape='DISK';key.data.size=4;key.data.color=(1,.92,.84);orig.point_at(key,(0,0,.4))
 material_audit['room']=dict(floor=placement['room_floor_srgb'],wall=placement['room_wall_srgb'],furniture=False)
""";assert marker in s;s=s.replace(marker,code+marker);compile(s,'render.py','exec');(R/'render.py').write_text(s)
c='scene_door_frame_w054';shutil.copytree(D/'source_cases'/c,R/'source_cases'/c,dirs_exist_ok=True);(R/'samples'/c).mkdir(parents=True,exist_ok=True);(R/'logs').mkdir(exist_ok=True)
s=(T/'render_preview.py').read_text().replace('v2v_pendulum_l110',c).replace('rope_preview.log','door_preview.log').replace("Path('/data/gaoya/agent-data/locks/cosmos_gpu/gpu2.lock')","(r/'preview.lock')");(R/'render_preview.py').write_text(s)
s=(T/'run_revision.py').read_text().replace("['V2V_PENDULUM']","['SCENE_DOOR_FRAME']").replace("  report=json.loads((after/'render_report.json').read_text());assert report['materials']['pendulum_rope']['color_srgb']=='#123B78'", "  report=json.loads((after/'render_report.json').read_text());assert report['materials']['room']['furniture']==False")
s=s.replace('all 5 pendulum rope cases','all 5 simple door cases');compile(s,'run_revision.py','exec');(R/'run_revision.py').write_text(s)
s=(T/'refresh_links.py').read_text().replace("['V2V_PENDULUM']","['SCENE_DOOR_FRAME']");s=s.replace("p=P.with_suffix('.tmp');", "data['family_labels']['SCENE_DOOR_FRAME']='门洞宽度（木块） · 暖灰素墙与深色哑光地面'\np=P.with_suffix('.tmp');");(R/'refresh_links.py').write_text(s)
print(R)
