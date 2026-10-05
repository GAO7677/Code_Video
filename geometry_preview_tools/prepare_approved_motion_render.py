from pathlib import Path
import json,shutil,math,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
P=D.parent.parent/'20261005_motion_preview_uniform_obstacles'
R=D.parent.parent/'20261005_test70_v2_approved_motion_render';R.mkdir(exist_ok=True)
def put(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2))
cfg=json.loads((D/'config.json').read_text());cfg['geometry_and_physics']='Exact replay of user-approved PyBullet trajectories';cfg['source_version']=str(P);cfg['preview_only']=False
mapping={'door_ball':'SCENE_DOOR_FRAME_BALL','door_block':'SCENE_DOOR_FRAME','ball_radius':'V2V_OBSTACLE_SIZE','obstacle_position':'V2V_PENDULUM_CABINET'}
labels={'ball_radius':'固定门宽／小球半径','obstacle_position':'固定摆锤／柜体位置'}
for g,f in mapping.items():
 cfg['scene_by_family'][f].pop('camera_adjustment',None)
 if g in labels:cfg['scene_by_family'][f]['task_label']=labels[g]
rope=dict(color_srgb='#123B78',roughness=.75,bump_strength=.18,bump_distance=.001)
cfg['scene_by_family'][mapping['obstacle_position']].setdefault('actor_material_overrides',{})['pendulum_rope']=rope
put(R/'config.json',cfg)
for name in ['fit_room.py','egl_guard.py','render.py','run_test70_v2.py']:shutil.copy2(D/name,R/name)
s=(R/'render.py').read_text()
# Partition surfaces use Test70's existing PBR plaster, while doors and frames retain group wood.
s=s.replace("if name in placement.get('actor_material_overrides',{}):", "if name.startswith('wall_'):mat=wood_material(name+'_plaster','beige_wall_001')\n if name in placement.get('actor_material_overrides',{}):")
(R/'render.py').write_text(s)
s=(R/'run_test70_v2.py').read_text();a=s.index(" old=json.loads((PROMPTS/");b=s.index(" warnings=",a)
s=s[:a]+" caption=m['input_caption']\n captions=dict(input_caption=caption,caption_variant='initial_state_only',source=str(s/'metadata.json'));put(p/'captions/captions.json',captions);(p/'captions/initial_state.txt').write_text(caption+'\\n')\n"+s[b:]
(R/'run_test70_v2.py').write_text(s)
manifest=json.loads((P/'manifest.json').read_text());out=[];groups=json.loads((D/'group_assignments.json').read_text());family_cases={f:[] for f in mapping.values()}
for c in manifest['cases']:
 g=c['group'];f=mapping[g];value=c['control']['value'];source=P/'cases'/c['case_id'];sc=json.loads((source/'scene.json').read_text());t=np.load(source/'trajectory.npz');tr={};actors={}
 if g=='door_ball':ident='scene_door_frame_ball_w'+f'{round(value*100):03d}'
 elif g=='door_block':ident='scene_door_frame_w'+f'{round(value*100):03d}'
 elif g=='ball_radius':ident='v2v_door_ball_radius_r'+f'{round(value*1000):03d}'
 else:ident='v2v_pendulum_cabinet_xm'+f'{round(-value*1000):04d}'
 def actor(name,shape,size,pos,rot,dynamic=False,visual=False):
  actors[name]=dict(shape=shape,size_m=size,initial_position_m=pos[0].tolist(),initial_orientation_wxyz=rot[0].tolist(),dynamic=dynamic,mass_kg=.49911 if dynamic else 0,friction=.18 if dynamic else 1,restitution=.06 if dynamic else 1,collision_enabled=not visual)
  tr[name+'_positions']=pos.tolist();tr[name+'_rotations']=rot.tolist()
 for a in sc['objects']:
  pos=t['positions'] if a['dynamic'] else np.tile(a['pos'],(90,1));q=t['rotations_xyzw'] if a['dynamic'] else np.tile(a.get('quat',[0,0,0,1]),(90,1));rot=q[:,[3,0,1,2]]
  size=dict(zip(['hx','hy','hz'],a['size'])) if a['shape']=='box' else dict(radius=a['size'])
  actor(a['name'],a['shape'],size,pos,rot,a['dynamic'])
 if g=='obstacle_position':
  anchor=np.array(c['control']['anchor']);xyz=t['positions'];v=xyz-anchor;end=xyz-v/np.linalg.norm(v,axis=1)[:,None]*.18
  q=np.array([Rotation.align_vectors([d/np.linalg.norm(d)],[[0,0,1]])[0].as_quat() for d in v]);actor('pendulum_rope','cylinder',dict(radius=.018,height=1.02),(anchor+end)/2,q[:,[3,0,1,2]],visual=True)
  cam=dict(position=[.15,-3.9,1.9],target=[-.05,0,1.4],yfov=48)
  caption='A textured pendulum ball hangs from a dark blue rope at an angle beside a tall cabinet. The wooden support post stands directly on the floor.'
 else:
  cam=dict(position=[-3,-.65,1.95],target=[.72,0,.85],yfov=48)
  caption='A textured '+('wooden block' if g=='door_block' else 'ball')+' is moving on the floor toward an open doorway integrated into a wall.'
 meta=dict(sample_id=ident,dataset='test70_v2',family_key=f,task_type=g,source_group=g,split='controlled_physics',seed=1701,title=caption,gravity_mps2=[0,0,-9.81],simulation=dict(sim_hz=5688,fps=30,frame_count=90,duration_seconds=89/30),control=dict(c['control'],units='m',value_label=f'{value:g} m'),actors=actors,resimulation=dict(dynamic_names=[a['name'] for a in sc['objects'] if a['dynamic']],physics=sc['physics'],source=str(source)),input_caption=caption)
 dest=R/'source_cases'/ident
 for name,data in [('metadata.json',meta),('trajectory.json',tr),('camera.json',cam),('simulation_report.json',c)]:put(dest/name,data)
 for name in ['contacts.json','contact_events.json']:shutil.copy2(source/name,dest/name)
 (R/'samples'/ident).mkdir(parents=True,exist_ok=True);family_cases[f].append(ident);out.append(dict(case=ident,preview_case=c['case_id'],family=f,source_sha256=hashlib.sha256((source/'trajectory.npz').read_bytes()).hexdigest()))
for f,ids in family_cases.items():groups[f]['cases']=ids;groups[f]['scene']=cfg['scene_by_family'][f]
put(R/'case_order.json',[a['case'] for a in out]);put(R/'case_mapping.json',out);put(R/'group_assignments.json',groups)
for sub in ['logs','reports','testjsons/v2v_jsons/test70_v2_all_cycles']:(R/sub).mkdir(parents=True,exist_ok=True)
for name in ['render.py','run_test70_v2.py']:compile((R/name).read_text(),name,'exec')
print('Prepared',len(out),'cases at',R)
