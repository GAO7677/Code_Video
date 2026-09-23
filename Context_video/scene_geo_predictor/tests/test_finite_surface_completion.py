import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
import numpy as np
from finite_surface_completion import complete_surface_depth


def test_multiple_surfaces_in_one_occlusion():
    z=np.full((90,120),2.); z[45:]=3.
    truth=z.copy(); target=np.zeros(z.shape,bool);target[30:60,50:70]=True;z[target]=0
    k=np.array([[100.,0,60],[0,100.,45],[0,0,1.]])
    out,filled,audit=complete_surface_depth(z,target,k)
    assert filled[:45].sum()>100 and filled[45:].sum()>100,audit
    assert np.allclose(out[filled],truth[filled]),'competing surfaces joined as one slanted plane'
    assert np.array_equal(out[~target],z[~target])


def test_visible_gaps_and_open_edge():
    z=np.full((90,120),2.);z[:,60:]=0
    target=np.zeros(z.shape,bool);target[30:60,50:70]=True;z[target]=0
    k=np.array([[100.,0,60],[0,100.,45],[0,0,1.]])
    out,filled,_=complete_surface_depth(z,target,k)
    assert not filled[:,60:].any(),'extrapolated past visible platform edge'
    assert np.array_equal(out[~target],z[~target])


if __name__=='__main__':
    test_multiple_surfaces_in_one_occlusion();test_visible_gaps_and_open_edge()
    print('PASS multiple surfaces per hole; visible gap/open edge; observed invariance')
