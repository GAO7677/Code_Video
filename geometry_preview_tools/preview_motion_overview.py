from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,subprocess,concurrent.futures
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261005_door_pendulum_motion_preview');FF='/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg'
def run(c):
 out=R/'cases'/c['case_id'];scene=json.loads((out/'scene.json').read_text());tr=np.load(out/'trajectory.npz');p.connect(p.DIRECT);moving=None;rope=None
 try:
  p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_BOX,halfExtents=[10,10,.05],rgbaColor=[.3,.32,.34,1]),basePosition=[0,0,-.05])
  for a in scene['objects']:
   args=dict(halfExtents=a['size']) if a['shape']=='box' else dict(radius=a['size']);b=p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_BOX if a['shape']=='box' else p.GEOM_SPHERE,rgbaColor=a['color'],**args),basePosition=a['pos'],baseOrientation=a.get('quat',[0,0,0,1]))
   if a['dynamic']:moving=b
  door=c['group']!='obstacle_position'
  if not door:rope=p.createMultiBody(0,-1,p.createVisualShape(p.GEOM_CYLINDER,radius=.018,length=.92,rgbaColor=[.03,.12,.4,1]))
  view=p.computeViewMatrix([.8,0,5],[.8,0,0],[0,1,0]) if door else p.computeViewMatrix([.2,-4.8,2.0],[-.15,0,1.35],[0,0,1]);proj=p.computeProjectionMatrixFOV(42,640/360,.03,30)
  enc=subprocess.Popen([FF,'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x360','-r','30','-i','-','-c:v','libx264','-threads','1','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'overview.mp4')],stdin=subprocess.PIPE)
  for i,(xyz,q) in enumerate(zip(tr['positions'],tr['rotations_xyzw'])):
   p.resetBasePositionAndOrientation(moving,xyz,q)
   if rope is not None:
    a=np.array([.2,0,2.5]);v=xyz-a;end=xyz-v/np.linalg.norm(v)*.18;rot,_=Rotation.align_vectors([v/np.linalg.norm(v)],[[0,0,1]]);p.resetBasePositionAndOrientation(rope,(a+end)/2,rot.as_quat())
   rgba=np.array(p.getCameraImage(640,360,view,proj,renderer=p.ER_TINY_RENDERER,shadow=1)[2],dtype=np.uint8).reshape(360,640,4);im=Image.fromarray(rgba[:,:,:3]);draw=ImageDraw.Draw(im);draw.rectangle((0,0,640,22),fill='white');draw.text((6,5),f'{c["case_id"]} | {i/30:.2f}s | '+('TOP VIEW' if door else 'FRONT VIEW'),fill='black');enc.stdin.write(im.tobytes())
   if i==45:im.save(out/'overview_045.png')
  enc.stdin.close();assert enc.wait()==0
 finally:p.disconnect()
if __name__=='__main__':
 cases=json.loads((R/'manifest.json').read_text())['cases']
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:list(pool.map(run,cases))
