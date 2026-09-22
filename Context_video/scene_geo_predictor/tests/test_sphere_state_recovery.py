import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from sphere_state_recovery import fit_rays, motion_fits

def test_exact_tangent_rays():
    k = np.array([[430.,0,319.5],[0,430.,179.5],[0,0,1.]])
    for center in [np.array([0.,0.,4.]), np.array([-1.,.4,4.5])]:
        axis = center/np.linalg.norm(center)
        u = np.cross(axis,[0,1,0]); u /= np.linalg.norm(u)
        v = np.cross(axis,u)
        a = np.arcsin(.11/np.linalg.norm(center))
        theta = np.linspace(0,2*np.pi,180,endpoint=False)
        rays = np.cos(a)*axis + np.sin(a)*(np.cos(theta)[:,None]*u+np.sin(theta)[:,None]*v)
        pixels = rays@k.T; pixels = pixels[:,:2]/pixels[:,2,None]
        estimated, _ = fit_rays(pixels,k,.11)
        assert np.linalg.norm(estimated-center)<1e-6

def test_terminal_velocity():
    t = np.arange(8)/30.; s=t-t[-1]
    p=np.array([1.,2.,3.]); v=np.array([2.,-.1,.3]); a=np.array([.4,0.,-2.])
    fits=motion_fits(p+s[:,None]*v+.5*s[:,None]**2*a,t)
    assert np.allclose(fits['robust_quadratic']['p7'],p,atol=1e-8)
    assert np.allclose(fits['robust_quadratic']['v7'],v,atol=1e-8)

if __name__=='__main__':
    test_exact_tangent_rays(); test_terminal_velocity(); print('2 tests passed')
