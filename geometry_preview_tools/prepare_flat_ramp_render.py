from pathlib import Path
import json,shutil,numpy as np
BASE=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer')
P=BASE/'20261006_flat_to_ramp_uniform_speed_preview';T=BASE/'20261005_test70_v2_approved_motion_render';R=BASE/'20261006_test70_v2_flat_ramp_render';R.mkdir(exist_ok=True)
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
cfg=json.loads((T/'config.json').read_text());cfg['source_version']=str(P);cfg['geometry_and_physics']='Exact approved flat-to-ramp trajectories, uniformly spaced speeds';cfg['scene_by_family']['V2V_OBSTACLE']['task_label']='平地转上坡／初速度';cfg['scene_by_family']['V2V_OBSTACLE'].pop('camera_adjustment',None);put(R/'config.json',cfg)
for name in ['render.py','fit_room.py','egl_guard.py','run_test70_v2.py','parallel_config.json']:shutil.copy2(T/name,R/name)
s=(R/'render.py').read_text();old=" obj=orig.add_actor('Task_'+name,a,mat,edge_clarity=False);obj.rotation_mode='QUATERNION';actors[name]=obj"
new=""" if a['shape']=='ramp_mesh':
  mesh=bpy.data.meshes.new('Task_'+name+'_mesh');mesh.from_pydata(a['vertices'],[],[a['indices'][i:i+3] for i in range(0,len(a['indices']),3)]);mesh.update();obj=bpy.data.objects.new('Task_'+name,mesh);bpy.context.collection.objects.link(obj);obj.data.materials.append(mat)
 else:obj=orig.add_actor('Task_'+name,a,mat,edge_clarity=False)
 obj.rotation_mode='QUATERNION';actors[name]=obj"""
assert old in s;s=s.replace(old,new);compile(s,'render.py','exec');(R/'render.py').write_text(s)
cases=[]
for v in [1.2,2.2,3.2,4.2,5.2]:
 src=P/('speed_'+str(v).replace('.','p'));scene=json.loads((src/'scene.json').read_text());tr=np.load(src/'trajectory.npz');ident='v2v_flat_to_ramp_v'+str(round(v*100));cases.append(ident)
 pos=tr['positions'];q=tr['rotations_xyzw'][:,[3,0,1,2]];verts=scene['ramp']['vertices'];indices=scene['ramp']['indices']
 actors={'ball':dict(shape='sphere',size_m=dict(radius=.11),initial_position_m=pos[0].tolist(),initial_orientation_wxyz=q[0].tolist(),dynamic=True,mass_kg=.49911,friction=.18,restitution=.06,collision_enabled=True),'ramp':dict(shape='ramp_mesh',size_m=dict(hx=1.8,hy=.45,hz=scene['ramp']['height_m']),vertices=verts,indices=indices,initial_position_m=[0,0,0],initial_orientation_wxyz=[1,0,0,0],dynamic=False,mass_kg=0,collision_enabled=True)}
 trajectory=dict(ball_positions=pos.tolist(),ball_rotations=q.tolist(),ramp_positions=[[0,0,0]]*90,ramp_rotations=[[1,0,0,0]]*90)
 meta=dict(sample_id=ident,dataset='test70_v2',family_key='V2V_OBSTACLE',task_type='flat_to_ramp',source_group='flat_to_ramp_initial_speed',split='controlled_physics',seed=1701,title='A ball moves from flat ground onto an uphill ramp.',gravity_mps2=[0,0,-9.81],simulation=dict(sim_hz=5688,fps=30,frame_count=90),control=dict(scene['control'],value_label=f'{v:g} m/s'),actors=actors,resimulation=dict(dynamic_names=['ball'],physics=scene['physics'],source=str(src)),input_caption='A textured ball is moving across a flat floor toward an upward sloping solid ramp.',approved_scene=scene)
 dest=R/'source_cases'/ident
 for name,data in [('metadata.json',meta),('trajectory.json',trajectory),('camera.json',json.loads((P/'camera.json').read_text())),('simulation_report.json',json.loads((src/'report.json').read_text()))]:put(dest/name,data)
 shutil.copy2(src/'contacts.json',dest/'contacts.json');(R/'samples'/ident).mkdir(parents=True,exist_ok=True)
put(R/'case_order.json',cases)
for sub in ['logs','reports','testjsons/v2v_jsons/test70_v2_all_cycles']:(R/sub).mkdir(parents=True,exist_ok=True)
put(R/'queue_config.json',dict(assignments=dict(zip(cases,[0,1,2,0,1])),calibration_case=cases[-1],calibration_gpu=2))
print(R)
