import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from local_plane_completion import complete_occluded_depth


def setup(slope=0.):
    yy,xx=np.mgrid[:90,:120];k=np.array([[100.,0,60],[0,100.,45],[0,0,1.]])
    z=2/(1+slope*(xx-60)/100);target=np.zeros_like(z,dtype=bool);target[35:55,50:70]=True
    expected=z.copy();z[target]=0
    return z,target,k,expected


def test_flat_and_sloped_planes():
    for slope in [0.,.3,-.5]:
        z,m,k,gt=setup(slope);out,filled,audit=complete_occluded_depth(z,m,k)
        assert filled.sum()==400,audit
        assert np.allclose(out,gt,atol=1e-8)
        assert np.array_equal(out[~m],z[~m]),'observed data changed'


def test_visible_gap_and_platform_edge():
    z,m,k,_=setup();z[10:25,10:25]=0
    out,filled,_=complete_occluded_depth(z,m,k)
    assert np.all(out[10:25,10:25]==0),'unrelated true gap filled'
    z,m,k,_=setup();z[:,60:]=4;z[m]=0
    _,filled,audit=complete_occluded_depth(z,m,k)
    assert not filled.any(),audit  # Step in surrounding geometry must reject.
    z,m,k,_=setup();z[:,60:]=0
    _,filled,audit=complete_occluded_depth(z,m,k)
    assert not filled.any(),audit  # Edge into unobserved region must reject.


if __name__=='__main__':
    test_flat_and_sloped_planes();test_visible_gap_and_platform_edge()
    print('2 local plane completion tests passed (flat, tilted, visible gap, step, open edge)')
