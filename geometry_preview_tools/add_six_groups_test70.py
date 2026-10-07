from pathlib import Path
import sys,json,math,copy,hashlib,shutil
import numpy as np
import pybullet as p
SOURCE=Path('/home/gaoya/Code_Video/geometry_preview_tools/geometry_sim_6cases_import_20261007/geometry_sim_6cases');sys.path.insert(0,str(SOURCE/'src'))
from scenes import FAMILIES
from physics import SphereWorld3
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261007_test70_v2_six_groups');D=Path('/data/gaoya/AAA_test_video/test70_v2');B=R.parent
if (R/'config.json').exists():
 raise SystemExit('This experiment already exists. Resume run_queue.py; do not overwrite reviewed renderer/config/source cases.')
R.mkdir(exist_ok=True)
def put(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
BASE=B/'20261006_test70_v2_flat_ramp_render'
for f in ['render.py','egl_guard.py','fit_room.py','run_test70_v2.py','parallel_config.json']:shutil.copy2(BASE/f,R/f)
config=json.loads((BASE/'config.json').read_text());existing=json.loads((D/'config.json').read_text());physics=json.loads((B/'20261002_full_test70_test89/config.json').read_text())['physics']
family_profiles=['SCENE_PUCK_BARRIER','V2V_DOMINO','F11','F12','V2V_GAP','SCENE_DOOR_FRAME']
family_keys=['GEO_CYLINDER_OFFSET','GEO_BARRIER_LENGTH','GEO_RAIL_GAP','GEO_HUMP_HEIGHT','GEO_HOLE_OFFSET','GEO_TABLE_EDGE']
rows=[]
def prism(points,depth):
 # CCW x/z polygon, extruded along Y; same surface on both sides.
 n=len(points);verts=[[float(x),y,float(z)] for y in [-depth/2,depth/2] for x,z in points];faces=[]
 for i in range(1,n-1):faces.extend([0,i+1,i,n,n+i,n+i+1])
 for i in range(n):j=(i+1)%n;faces.extend([i,j,n+j,i,n+j,n+i])
 return verts,faces
for group,(build,values) in enumerate(FAMILIES):
 family=family_keys[group];profile=family_profiles[group];config['scene_by_family'][family]=copy.deepcopy(existing['scene_by_family'][profile]);config['scene_by_family'][family].pop('camera_adjustment',None);config['scene_by_family'][family].pop('tabletop',None);config['scene_by_family'][family].pop('task_origin_offset',None)
 config['actor_materials'][family]=copy.deepcopy(existing['actor_materials'][profile]);group_rows=[]
 for idx,value in enumerate(values):
  scene=build(value);case='geo6_'+scene.key+'_'+f'{idx+1:02d}';actors={}
  def actor(n,shape,size,pos,dynamic=False,vel=None,ang=None,q=None,verts=None,indices=None):
   a=dict(shape=shape,size_m=size,initial_position_m=list(map(float,pos)),initial_orientation_wxyz=[1,0,0,0] if q is None else list(map(float,q)),dynamic=dynamic,role='dynamic' if dynamic else 'anchored_fixture',mass_kg=.49911 if dynamic else 0,friction=.18 if dynamic else 1,restitution=.06 if dynamic else 1,initial_linear_velocity_mps=[0,0,0] if vel is None else list(map(float,vel)),initial_angular_velocity_radps=[0,0,0] if ang is None else list(map(float,ang)),instance_id=len(actors)+1,collision_enabled=True)
   if verts is not None:a.update(vertices=verts,indices=indices)
   actors[n]=a
  def box(n,pos,half,**kwargs):actor(n,'box',dict(zip(['hx','hy','hz'],map(float,half))),pos,**kwargs)
  if isinstance(scene.world,SphereWorld3):
   w=scene.world;actor('ball','sphere',dict(radius=w.r),w.p,True,w.v,w.omega)
   for a in w.boxes:
    if a.name!='floor':box(a.name,a.center,a.half)
   if group==2:
    for sign in [-1,1]:
     for xx in [.1,2.25,4.4]:box(f'rail_support_{sign}_{xx}',[xx,sign*(value/2+.09),.395],[.055,.055,.395])
  else:
   for a in scene.world.bodies:
    if a.name=='floor' or a.name.startswith('track_'):continue
    if scene.plane=='planar_xy':
     if a.shape=='circle':
      if a.dynamic:actor(a.name,'sphere',dict(radius=a.radius),[*a.p,.8+a.radius],True,[*a.v,0],[0,0,a.omega])
      else:actor(a.name,'cylinder',dict(radius=a.radius,height=a.render.get('height',.5)),[*a.p,.8+a.render.get('height',.5)/2])
     else:
      half=np.max(np.abs(a.vertices),axis=0);h=a.render.get('height',.43);box(a.name,[*a.p,.8+h/2],[*half,h/2],q=[math.cos(a.angle/2),0,0,math.sin(a.angle/2)])
    else:
     pos=[a.p[0],0,a.p[1]];vel=[a.v[0],0,a.v[1]];q=[math.cos(a.angle/2),0,-math.sin(a.angle/2),0]
     if a.shape=='circle':actor(a.name,'sphere',dict(radius=a.radius),pos,a.dynamic,vel,[0,-a.omega,0],q)
     else:
      half=np.max(np.abs(a.vertices),axis=0);box(a.name,pos,[half[0],a.render.get('depth',1.1)/2,half[1]],dynamic=a.dynamic,vel=vel,ang=[0,-a.omega,0],q=q)
   if scene.plane=='planar_xy':
    box('table_top',[1.4,0,.75],[4.1,2.7,.05])
    for xx in [-2.5,1.4,5.3]:
     for yy in [-2.4,2.4]:box(f'table_leg_{xx}_{yy}',[xx,yy,.35],[.065,.065,.35])
   if group==3:
    x=np.linspace(0,1.8,49);z=.70+value*np.sin(np.pi*x/1.8)**2;points=[[0,0],[1.8,0]]+list(map(list,zip(x[::-1],z[::-1])));verts,indices=prism(points,1.1);actor('hump','ramp_mesh',{},[0,0,0],verts=verts,indices=indices)
  dynamic=[n for n,a in actors.items() if a['dynamic']];p.connect(p.DIRECT)
  p.setGravity(0,0,-9.81);p.setTimeStep(1/5688);p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,restitutionVelocityThreshold=.05,contactBreakingThreshold=.001)
  floor=p.createMultiBody(0,p.createCollisionShape(p.GEOM_PLANE));p.changeDynamics(floor,-1,lateralFriction=1,restitution=1);bodies={}
  for n,a in actors.items():
   s=a['size_m']
   if a['shape']=='sphere':shape=p.createCollisionShape(p.GEOM_SPHERE,radius=s['radius'])
   elif a['shape']=='box':shape=p.createCollisionShape(p.GEOM_BOX,halfExtents=[s[k] for k in ['hx','hy','hz']])
   elif a['shape']=='cylinder':shape=p.createCollisionShape(p.GEOM_CYLINDER,radius=s['radius'],height=s['height'])
   else:shape=p.createCollisionShape(p.GEOM_MESH,vertices=a['vertices'],indices=a['indices'],flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
   q=a['initial_orientation_wxyz'];body=p.createMultiBody(a['mass_kg'],shape,basePosition=a['initial_position_m'],baseOrientation=q[1:]+q[:1]);bodies[n]=body
   p.changeDynamics(body,-1,lateralFriction=a['friction'],restitution=a['restitution'],linearDamping=.015 if a['dynamic'] else 0,angularDamping=.025 if a['dynamic'] else 0,rollingFriction=0,spinningFriction=0)
   if a['dynamic']:p.resetBaseVelocity(body,a['initial_linear_velocity_mps'],a['initial_angular_velocity_radps'])
  p.performCollisionDetection();name_by_id={v:k for k,v in bodies.items()};name_by_id[0]='floor';initial=[c[8] for c in p.getContactPoints() if (c[1] in [bodies[n] for n in dynamic] or c[2] in [bodies[n] for n in dynamic]) and c[8]<-.003];assert not initial,(case,initial)
  tr=dict(object_names=list(actors),object_roles=[a['role'] for a in actors.values()],frame_times_s=(np.arange(90)/30).tolist());contacts=[];step=0
  for n in actors:
   for field in ['positions','rotations','linear_velocity','angular_velocity']:tr[n+'_'+field]=[]
  for frame in range(90):
   while step<round(frame*5688/30):p.stepSimulation();step+=1
   for n,body in bodies.items():
    pos,q=p.getBasePositionAndOrientation(body);v,w=p.getBaseVelocity(body)
    for field,data in [('positions',pos),('rotations',[q[3],*q[:3]]),('linear_velocity',v),('angular_velocity',w)]:tr[n+'_'+field].append(list(data))
   cc=[dict(obj_a=name_by_id[c[1]],obj_b=name_by_id[c[2]],point=list(c[6]),point_on_a=list(c[5]),normal=list(c[7]),distance_m=c[8],normal_force_n=c[9]) for c in p.getContactPoints() if name_by_id[c[1]] in dynamic or name_by_id[c[2]] in dynamic];contacts.append(dict(frame=frame,time_s=frame/30,simulation_time_s=step/5688,contacts=cc))
  p.disconnect()
  assert all(np.isfinite(tr[n+'_positions']).all() for n in actors)
  captions=['A textured ball moves across a supported table toward a fixed cylindrical obstacle.','A textured ball moves across a supported table toward an angled wooden barrier.','A textured ball rolls from a raised platform toward two parallel supporting rails.','A textured ball rolls along a raised track toward a smooth hump.','A textured ball moves across a solid platform with a rectangular opening.','A textured ball moves toward a wooden block resting near the edge of a raised platform.']
  control=dict(variable=scene.variable,value=value,units='m',value_label=f'{value:g} m')
  if group==5:control.update(edge_world_x_m=value,block_initial_center_x_m=.45,block_half_length_x_m=.16,initial_edge_clearance_m=value-.61)
  meta=dict(sample_id=case,dataset='test70_v2',family_key=family,task_type=scene.key,source_group=scene.key,split='controlled_physics',seed=1701,title=scene.title,input_caption=captions[group],gravity_mps2=[0,0,-9.81],simulation=dict(sim_hz=5688,fps=30,frame_count=90,duration_seconds=89/30),control=control,actors=actors,resimulation=dict(dynamic_names=dynamic,physics=physics,source_archive='/data/gaoya/tmp/geometry_sim_6cases_source.zip',geometry_preserved=True,planar_constraint_removed=True,added_supports=group in [0,1,2]),scenario_spec=dict(source_model=scene.plane,source_camera=scene.camera))
  dest=R/'source_cases'/case;put(dest/'metadata.json',meta);put(dest/'trajectory.json',tr);put(dest/'contacts.json',contacts);put(dest/'simulation_report.json',dict(case=case,frames=90,finite=True,initial_penetrations=initial,physics_engine='PyBullet',contact_records=sum(len(x['contacts']) for x in contacts),source_archive_sha256=hashlib.sha256(Path('/data/gaoya/tmp/geometry_sim_6cases_source.zip').read_bytes()).hexdigest()))
  (R/'samples'/case).mkdir(parents=True,exist_ok=True);row=dict(case=case,family=family,title=scene.title,value=value,variable=scene.variable,profile=profile);rows.append(row);group_rows.append((case,meta,tr))
 # Fixed camera fitted to all three rollouts, with key geometry also in view.
 points=[]
 for c,m,t in group_rows:
  for n in m['resimulation']['dynamic_names']:
   pos=np.asarray(t[n+'_positions']);radius=max(m['actors'][n]['size_m'].values());points.extend(pos+np.array([radius,radius,radius]));points.extend(pos-np.array([radius,radius,radius]))
  for n,a in m['actors'].items():
   if n in ['cylinder','barrier','hump','rail_front','rail_back']:
    if a['shape']=='ramp_mesh':points.extend(a['vertices'])
    else:
     pos=np.array(a['initial_position_m']);size=a['size_m'];half=np.array([size[k] for k in ['hx','hy','hz']]) if a['shape']=='box' else np.array([size.get('radius',.2),size.get('radius',.2),size.get('height',.4)/2]);points.extend([pos+half,pos-half])
 pts=np.asarray(points);lo=pts.min(0);hi=pts.max(0);target=(lo+hi)/2;target[2]=max(.5,target[2]);elev=math.radians(40 if group in [0,1,4] else 28);direction=np.array([.17*math.cos(elev),-math.sqrt(1-.17**2)*math.cos(elev),math.sin(elev)]);forward=-direction;right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);delta=pts-target;fov=math.radians(48);distance=max(4.2,float(np.max(np.maximum(abs(delta@right)/(math.tan(fov/2)*896/512),abs(delta@up)/math.tan(fov/2))-delta@forward))*1.18)
 cam=dict(position=(target+direction*distance).tolist(),target=target.tolist(),yfov=48)
 for case,m,tr in group_rows:put(R/'source_cases'/case/'camera.json',cam)
 # Same appearance and camera for all variants in a family.
 config['scene_by_family'][family]['task_label']=scene.title
put(R/'config.json',config);put(R/'case_order.json',[x['case'] for x in rows]);put(R/'groups.json',rows);put(R/'integration_plan.json',dict(source_cases=18,target_defined_cases=88,physics=physics,render='Existing Test70 v2 Cycles renderer, PBR material profiles and indoor scenes',gpu=1))
for folder in ['logs','reports','testjsons/v2v_jsons/test70_v2_all_cycles']:(R/folder).mkdir(parents=True,exist_ok=True)
print('SIMULATED',len(rows),flush=True)
