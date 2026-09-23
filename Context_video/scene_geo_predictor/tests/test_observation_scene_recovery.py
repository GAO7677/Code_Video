import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from observation_scene_recovery import silhouette_depth_sphere,regularize_planes,fixed_camera
from generic_context_geometry import sphere_fit

def scene(flatten=1):
    k=np.array([[180.,0,100],[0,180.,100],[0,0,1]])
    y,x=np.mgrid[:200,:200];rays=np.stack([x,y,np.ones_like(x)],-1)@np.linalg.inv(k).T;d=[];m=[]
    for t in range(8):
        c=np.array([-.08+.02*t,0,2.]);aa=(rays*rays).sum(-1);dot=rays@c;disc=dot**2-aa*(c@c-.3**2)
        mask=disc>0;z=(dot-np.sqrt(np.maximum(disc,0)))/aa
        d.append(np.where(mask,1.7+flatten*(z-1.7),4));m.append(mask)
    return np.array(d),np.tile(k,(8,1,1)),np.tile(np.c_[np.eye(3),np.zeros(3)],(8,1,1)),np.array(m),np.arange(8)/30

def test_silhouette_removes_curvature_radius_ambiguity():
    d,k,e,m,t=scene(.4)
    old=sphere_fit(d,k,e,m,1,t)
    assert abs(old['radius']-.3)>.1 # actual old failure despite its admission
    for scale in [1.,2.]:
        r=silhouette_depth_sphere(d,np.ones_like(d),k,e,m,scale,t)
        assert r['status']=='ESTIMATED',r
        assert abs(r['radius']-.3*scale)<.01*scale
        assert np.linalg.norm(np.array(r['p7'])-[.06*scale,0,2*scale])<.04*scale
        assert np.linalg.norm(np.array(r['v7'])-[.6*scale,0,0])<.02*scale

def test_plane_preserves_hole_and_bounds_displacement():
    x,y=np.meshgrid(np.linspace(-1,1,35),np.linspace(-1,1,35));rng=np.random.default_rng(42)
    verts=np.stack([x,y,2+.2*x+rng.normal(0,.002,x.shape)],-1).reshape(-1,3)
    faces=[]
    for i in range(34):
        for j in range(34):
            if 12<=j<=20:continue # genuine gap must not be filled
            a=i*35+j;faces.extend([[a,a+1,a+35],[a+1,a+36,a+35]])
    mesh={'vertices':verts.tolist(),'faces':faces,'source':'synthetic finite plane with gap'}
    r=regularize_planes(mesh)
    assert r['faces']==faces
    assert np.max(np.linalg.norm(np.array(r['vertices'])-verts,axis=1))<=.010001
    assert r['plane_regularization']['patches']

def test_camera_rejects_motion():
    import cv2
    rng=np.random.default_rng(42);im=rng.integers(0,255,(100,140,3),dtype=np.uint8)
    d,k,e,m,t=scene();m=np.zeros((8,100,140),bool)
    ks,es,a=fixed_camera([im]*8,m,k,e)
    assert np.all(ks==ks[0]) and max(a['static_track_p90_px'])<.1
    moved=[cv2.warpAffine(im,np.float32([[1,0,i],[0,1,0]]),(140,100)) for i in range(8)]
    try:fixed_camera(moved,m,k,e)
    except ValueError:pass
    else:raise AssertionError('moving camera silently accepted')

if __name__=='__main__':
    test_silhouette_removes_curvature_radius_ambiguity();test_plane_preserves_hole_and_bounds_displacement();test_camera_rejects_motion()
    print('PASS: silhouette/depth scale, flattened cap, bounded planes/gaps, fixed/moving camera')
