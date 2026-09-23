import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from generic_bullet_input import run

def test_mesh_alignment():
    mesh={'type':'triangle_mesh','position':[0,0,0],'orientation':[0,0,0,1],
          'vertices':[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]],'faces':[[0,1,2],[0,2,3]]}
    physics={'fixedTimeStep':1/240,'numSubSteps':8,'calls_per_output':8,'outputs':1,
             'mass':1.,'friction':.35,'restitution':.25,'linear_damping':0.,'angular_damping':0.}
    gravity={'status':'ESTIMATED','vector':[0,0,-9.81]}
    state={'status':'ESTIMATED','shape':'sphere','radius':.1,'p7':[0,0,.08],'v7':[0,0,0],'omega':[0,0,0]}
    options={'up':[0,0,1],'confirmed_support':True,'aligned':True}
    strict=run(state,[mesh],physics,gravity,diagnostic_alignment={**options,'aligned':False})
    assert strict['status']=='FAIL' and strict['step_calls']==0
    aligned=run(state,[mesh],physics,gravity,diagnostic_alignment=options)
    assert aligned['status']=='EXECUTED' and aligned['initialization']['status']=='CONTACT_ALIGNED'
    assert abs(aligned['initialization']['displacement_m'][2]-.02001)<1e-5
    for z,reason in [(.03,'correction_exceeds_55mm'),(.2,'missing_confirmed_initial_support_no_suction')]:
        result=run({**state,'p7':[0,0,z]},[mesh],physics,gravity,diagnostic_alignment=options)
        assert result['status']=='FAIL' and result['reason']==reason and result['step_calls']==0,result
    result=run(state,[mesh],physics,gravity,diagnostic_alignment={**options,'confirmed_support':False})
    assert result['reason']=='support_not_confirmed_by_GT_A' and result['step_calls']==0

if __name__=='__main__':test_mesh_alignment();print('mesh alignment bounds / strict / no suction / confirmation PASS')
