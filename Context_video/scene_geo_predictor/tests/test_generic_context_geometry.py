import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from generic_context_geometry import sphere_fit,finite_mesh,motion_prompt
from generic_bullet_input import run

def test_unknown_radius_and_scale():
    k=np.array([[100.,0,50.],[0,100.,50.],[0,0,1.]])
    yy,xx=np.mgrid[:100,:100];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(k).T
    depths=[];masks=[];centers=[]
    for t in range(8):
        c=np.array([-.08+t*.02,0,2.]);centers.append(c*2)
        dot=rays@c;aa=(rays*rays).sum(-1);disc=dot**2-aa*(c@c-.3**2)
        mask=disc>0;z=np.where(mask,(dot-np.sqrt(np.maximum(disc,0)))/aa,4.)
        depths.append(z);masks.append(mask)
    e=np.tile(np.c_[np.eye(3),np.zeros(3)],(8,1,1));times=np.arange(8)/30
    result=sphere_fit(np.array(depths),np.tile(k,(8,1,1)),e,np.array(masks),2.,times)
    assert result['status']=='ESTIMATED',result
    assert abs(result['radius']-.6)<1e-5
    assert np.linalg.norm(np.array(result['p7'])-centers[-1])<1e-5
    assert np.linalg.norm(np.array(result['v7'])-[1.2,0,0])<1e-5
    mesh,gravity=finite_mesh(np.array(depths),np.tile(k,(8,1,1)),e,np.array(masks),2.)
    assert gravity['status']=='UNKNOWN' and mesh['thickness']['value'] is None
    assert run(result,[mesh],{},gravity)['step_calls']==0

def test_motion_does_not_require_circle():
    background=np.random.default_rng(42).integers(20,120,(120,160,3),dtype=np.uint8)
    rgb=np.repeat(background[None],8,axis=0)
    for t in range(8):rgb[t,45:65,35+t*4:55+t*4]=220
    # Background subtraction may expose both current and vacated regions.
    # Ambiguity must reject, never fall back to circle or a default box.
    try:
        _,audit=motion_prompt(rgb)
        assert audit['discovery_uses_circularity'] is False
    except ValueError as exc:
        assert str(exc)=='UNKNOWN_multiple_motion_candidates'

def test_bullet_timing_no_hidden_alignment():
    state={'status':'ESTIMATED','shape':'sphere','radius':.2,'p7':[0,0,0], 'v7':[1,0,0],'omega':[0,0,0]}
    config={'fixedTimeStep':1/240,'numSubSteps':8,'calls_per_output':8,'outputs':41,
            'mass':1.,'friction':.35,'restitution':.25,'linear_damping':0.,'angular_damping':0.}
    result=run(state,[],config,{'status':'ESTIMATED','vector':[0,0,0]})
    assert result['status']=='EXECUTED' and result['step_calls']==328
    assert np.allclose(result['positions'][0],[1/30,0,0],atol=1e-6)
    assert np.allclose(result['positions'][-1],[41/30,0,0],atol=1e-6)
    assert result['position_modified'] is False

if __name__=='__main__':
    test_unknown_radius_and_scale();test_motion_does_not_require_circle();test_bullet_timing_no_hidden_alignment();print('3 tests passed')
