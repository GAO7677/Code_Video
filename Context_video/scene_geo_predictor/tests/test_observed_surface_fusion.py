"""Physical/visibility regressions, independent of any pilot case or family."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
import pybullet as p
from generic_context_geometry import finite_mesh_observed


def scene():
    h,w=84,126
    k=np.tile(np.array([[100.,0,63],[0,100.,42],[0,0,1.]]),(8,1,1))
    e=np.tile(np.c_[np.eye(3),np.zeros(3)],(8,1,1))
    depth=np.full((8,h,w),2.)
    mask=np.zeros_like(depth,dtype=bool)
    for t in range(8):
        mask[t,34:48,15+t*10:29+t*10]=True
        depth[t][mask[t]]=1.5
    return depth,k,e,mask


def hit(mesh,x,y):
    c=p.connect(p.DIRECT)
    try:
        s=p.createCollisionShape(p.GEOM_MESH,vertices=mesh['vertices'],indices=np.array(mesh['faces']).ravel().tolist(),flags=p.GEOM_FORCE_CONCAVE_TRIMESH,physicsClientId=c)
        p.createMultiBody(baseMass=0,baseCollisionShapeIndex=s,physicsClientId=c)
        return p.rayTest([x,y,0],[x,y,3],physicsClientId=c)[0]
    finally:p.disconnect(physicsClientId=c)


def test_revealed_surface_and_real_gap():
    d,k,e,m=scene()
    d[:,10:25,50:70]=0  # No view observed this region: do not invent a face.
    mesh,_=finite_mesh_observed(d,k,e,m,1.)
    assert hit(mesh,-.2,0)[0]>=0, 'later object mask erased previously visible plane'
    assert abs(hit(mesh,-.2,0)[3][2]-2)<1e-6, 'dynamic surface leaked into static geometry'
    assert hit(mesh,0,-.5)[0]==-1, 'unobserved gap was filled'


def test_camera_transform_and_scale():
    d,k,e,m=scene()
    # Camera translations parallel to a plane must preserve its world depth.
    e[:,0,3]=np.arange(8)*.01
    mesh,_=finite_mesh_observed(d,k,e,m,1.7)
    z=np.array(mesh['vertices'])[np.array(mesh['faces']).ravel(),2]
    assert np.allclose(z,3.4), 'wrong reprojection or scale application'


if __name__=='__main__':
    test_revealed_surface_and_real_gap();test_camera_transform_and_scale()
    print('2 visibility/physics regression tests passed')
