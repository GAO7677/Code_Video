from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,math,subprocess,concurrent.futures
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261005_door_pendulum_motion_preview')
BASE=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261002_full_test70_test89')
physics=json.loads((BASE/'config.json').read_text())['physics']
FF='/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg'
def box(name,pos,half,color,dynamic=False):return dict(name=name,shape='box',pos=pos,size=half,color=color,dynamic=dynamic)
def sphere(name,pos,r,color):return dict(name=name,shape='sphere',pos=pos,size=r,color=color,dynamic=True)
def setup(group,value):
 objs=[];anchor=None
 if group!='obstacle_position':
  width=value if group in ['door_ball','door_block'] else .30;r=value if group=='ball_radius' else .18
  # A full room partition with a 2.1 m clear opening; frame is tangent to wall.
  x=.72;h=2.1
  for side in [-1,1]:
   objs.append(box('wall_'+str(side),[x,side*(width/2+(3-width/2)/2),1.4],[.14,(3-width/2)/2,1.4],[.77,.75,.7,1]))
   objs.append(box('jamb_'+str(side),[x-.015,side*(width/2+.025),h/2],[.17,.025,h/2],[.38,.23,.12,1]))
  objs.append(box('wall_header',[x,0,(h+2.8)/2],[.14,width/2,(2.8-h)/2],[.77,.75,.7,1]))
  objs.append(box('lintel',[x-.015,0,h+.035],[.17,width/2,.035],[.38,.23,.12,1]))
  # Leaf opens 100 degrees toward the rear room, outside the moving path.
  theta=math.radians(-100);hinge=np.array([x+.16,-width/2-.025,h/2]);direction=np.array([-math.sin(theta),math.cos(theta),0]);leaf=box('fixed_open_door',(hinge+direction*(width/2)).tolist(),[.022,width/2,h/2],[.46,.29,.16,1]);leaf['quat']=p.getQuaternionFromEuler([0,0,theta]);objs.append(leaf)
  if group=='door_block':moving=box('moving_block',[-.2,0,.28],[.34,.24,.28],[.67,.26,.13,1],True)
  else:moving=sphere('moving_ball',[-.2,0,r],r,[.08,.32,.72,1])
  speed=2.6 if group=='door_block' else 1.8
  if group=='door_ball':moving['pos'][1]=.10
  moving['velocity']=[speed,0,0];objs.append(moving)
  cfg=dict(variable='ball_radius_m' if group=='ball_radius' else 'door_opening_width_m',value=value,fixed_door_width_m=width,door_clear_height_m=2.1,door_leaf_open_angle_deg=100,start_x_m=-.2,lateral_offset_m=.10 if group=='door_ball' else 0,initial_speed_mps=speed,ball_radius_m=r if group!='door_block' else None,block_dimensions_m=[.68,.48,.56] if group=='door_block' else None)
 else:
  anchor=np.array([.2,0,2.5]);length=1.1;angle=math.radians(18);bob=anchor+np.array([length*math.sin(angle),0,-length*math.cos(angle)])
  objs=[box('base',[.2,.26,.09],[.45,.4,.09],[.5,.32,.19,1]),box('post',[.2,.26,1.34],[.065,.05,1.16],[.5,.32,.19,1]),box('crossbar',[.2,.11,2.53],[.14,.2,.03],[.5,.32,.19,1]),box('cabinet',[value,0,1.275],[.24,.36,1.275],[.3,.43,.46,1]),sphere('pendulum_bob',bob.tolist(),.18,[.72,.12,.25,1])]
  cfg=dict(variable='cabinet_center_x_m',value=value,cabinet_right_face_x_m=value+.24,anchor=anchor.tolist(),length_m=length,release_angle_deg=18,bob_radius_m=.18,cabinet_dimensions_m=[.48,.72,2.55],initial_speed_mps=0)
 return objs,cfg,anchor

def run(job):
 group,value=job;ident=group+'_'+str(value).replace('.','p').replace('-','m');out=R/'cases'/ident;out.mkdir(parents=True,exist_ok=True);objs,cfg,anchor=setup(group,value);p.connect(p.DIRECT)
 try:
  hz=5688;p.setGravity(0,0,-9.81);p.setTimeStep(1/hz);p.setPhysicsEngineParameter(numSolverIterations=100,deterministicOverlappingPairs=1,restitutionVelocityThreshold=.05,contactBreakingThreshold=.001)
  floor=p.createMultiBody(0,p.createCollisionShape(p.GEOM_BOX,halfExtents=[10,10,.05]),p.createVisualShape(p.GEOM_BOX,halfExtents=[10,10,.05],rgbaColor=[.3,.32,.34,1]),basePosition=[0,0,-.05]);p.changeDynamics(floor,-1,lateralFriction=1,restitution=1)
  bodies={};moving=None
  for a in objs:
   args=dict(halfExtents=a['size']) if a['shape']=='box' else dict(radius=a['size']);typ=p.GEOM_BOX if a['shape']=='box' else p.GEOM_SPHERE
   b=p.createMultiBody(.49911 if a['dynamic'] else 0,p.createCollisionShape(typ,**args),p.createVisualShape(typ,rgbaColor=a['color'],**args),basePosition=a['pos'],baseOrientation=a.get('quat',[0,0,0,1]));bodies[a['name']]=b
   if a['dynamic']:
    moving=b;p.changeDynamics(b,-1,lateralFriction=.18,restitution=.06,linearDamping=.015,angularDamping=.025,rollingFriction=0,spinningFriction=0);p.resetBaseVelocity(b,a.get('velocity',[0,0,0]),[0,0,0])
   else:p.changeDynamics(b,-1,lateralFriction=1,restitution=1)
  rope=None
  if anchor is not None:
   pos=np.array(p.getBasePositionAndOrientation(moving)[0]);joint=p.createConstraint(bodies['post'],-1,moving,-1,p.JOINT_POINT2POINT,[0,0,0],(anchor-np.array([.2,.26,1.34])).tolist(),(anchor-pos).tolist());p.changeConstraint(joint,maxForce=500)
   rope=p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_CYLINDER,radius=.018,length=.92,rgbaColor=[.035,.12,.4,1]),basePosition=[0,0,0])
  p.performCollisionDetection();penetrations=[dict(a=c[1],b=c[2],depth=c[8]) for c in p.getContactPoints(bodyA=moving) if c[8]<-.003];assert not penetrations,penetrations
  positions=[];rotations=[];contacts=[];support=[];step=0;events=[]
  for frame in range(90):
   target=round(frame*hz/30)
   while step<target:
    p.stepSimulation();step+=1
    for contact in p.getContactPoints(bodyA=moving):
     if contact[2]!=floor and contact[9]>1e-5:events.append(dict(step=step,time_s=step/hz,other=next((n for n,b in bodies.items() if b==contact[2]),str(contact[2])),force=contact[9]))
   xyz,q=p.getBasePositionAndOrientation(moving);positions.append(xyz);rotations.append(q)
   contacts.append([dict(other=next((n for n,b in bodies.items() if b==c[2]),'floor'),force=c[9],distance=c[8]) for c in p.getContactPoints(bodyA=moving) if c[2]!=floor and c[9]>1e-5])
  positions=np.array(positions);np.savez_compressed(out/'trajectory.npz',positions=positions,rotations_xyzw=rotations,times=np.arange(90)/30)
  hit=sorted({min(89,round(e['time_s']*30)) for e in events});isdoor=group!='obstacle_position'
  report=dict(case_id=ident,group=group,control=cfg,frames=90,fps=30,initial_penetrations=penetrations,collision_frames=hit,first_contact_frame=hit[0] if hit else None,contact_objects=sorted({e['other'] for e in events}),first_contact_time_s=events[0]['time_s'] if events else None,contact_event_count=len(events),final_position=positions[-1].tolist(),passed_door=bool(positions[-1,0]>(.86+(.34 if group=='door_block' else cfg['ball_radius_m']))) if isdoor else None,pendulum_length_error_m=float(np.max(np.abs(np.linalg.norm(positions-anchor,axis=1)-1.1))) if anchor is not None else None)
  (out/'report.json').write_text(json.dumps(report,indent=2));(out/'scene.json').write_text(json.dumps(dict(objects=objs,control=cfg,physics=physics),indent=2));(out/'contacts.json').write_text(json.dumps(contacts));(out/'contact_events.json').write_text(json.dumps(events))
  if (out/'motion.mp4').exists():print(ident,'pass',report['passed_door'],'contact',report['first_contact_time_s'],flush=True);return report
  # Physics replay in TinyRenderer: CPU only, no GPU or photorealistic assets.
  view=p.computeViewMatrix([-3.5,-3.5,2.7],[.5,0,1.0],[0,0,1]) if isdoor else p.computeViewMatrix([3,-7,3.1],[-.25,0,1.35],[0,0,1]);proj=p.computeProjectionMatrixFOV(48,640/360,.03,30)
  cmd=[FF,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','-','-an','-c:v','libx264','-threads','1','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'motion.mp4')];enc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
  from PIL import Image,ImageDraw
  for i,(xyz,q) in enumerate(zip(positions,rotations)):
   p.resetBasePositionAndOrientation(moving,xyz,q)
   if rope is not None:
    direction=xyz-anchor;end=xyz-direction/np.linalg.norm(direction)*.18;rot,_=Rotation.align_vectors([direction/np.linalg.norm(direction)],[[0,0,1]]);p.resetBasePositionAndOrientation(rope,((anchor+end)/2).tolist(),rot.as_quat())
   rgba=np.asarray(p.getCameraImage(640,360,view,proj,renderer=p.ER_TINY_RENDERER,shadow=1,lightDirection=[-3,-4,8])[2],dtype=np.uint8).reshape(360,640,4);im=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(im);draw.rectangle([0,0,640,24],fill=(245,245,245));draw.text((8,6),f'{ident} | t={i/30:.2f}s | contacts={len(contacts[i])}',fill=(20,20,20))
   if i in [0,45,89]:im.save(out/f'frame_{i:03d}.png')
   enc.stdin.write(im.tobytes())
  enc.stdin.close();assert enc.wait()==0;print(ident,'pass',report['passed_door'],'contact',report['first_contact_frame'],flush=True);return report
 finally:p.disconnect()
if __name__=='__main__':
 jobs=[(g,v) for g,vs in [('door_ball',[.38,.46,.54,.62,.74]),('door_block',[.38,.46,.54,.62,.74]),('ball_radius',[.08,.11,.14,.17,.20]),('obstacle_position',[-.85,-.75,-.65,-.55,-.50])] for v in vs]
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
 (R/'manifest.json').write_text(json.dumps(dict(status='MOTION_PREVIEW_AWAITING_USER_APPROVAL',renderer='PyBullet TinyRenderer CPU',cases=results),ensure_ascii=False,indent=2));print('ALL_DONE',flush=True)
