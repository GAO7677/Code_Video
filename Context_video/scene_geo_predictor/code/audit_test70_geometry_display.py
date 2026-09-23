"""Read-only estimator diagnosis, GT used only in post-freeze measurements."""
import json
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import crop_transform, dump_json
from run_grounded_generic_pilot36 import verify

def main():
    root=Path('/data/gaoya/agent-data/outputs/test70_context_pipeline_20260923_v1')
    verify(root,'estimate_freeze.json');verify(root,'rollout_freeze.json')
    mapping={r['id']:r for r in json.loads((root/'evaluation_mapping.json').read_text())}
    viewer=json.loads((root/'viewer_data.json').read_text())
    records=[]
    truth=Path('/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v1/cases')
    for row in viewer['records']:
        cid=row['id'];folder=root/'estimates'/cid
        mesh=json.loads((folder/'collision_primitive.json').read_text())
        with np.load(folder/'vggt_raw.npz') as a:
            depth=a['depth'][...,0];k=a['intrinsic'];e=a['extrinsic']
        with np.load(folder/'tracking.npz') as a:masks=a['masks']
        est=json.loads((folder/'result.json').read_text());scale=est['scale']
        verts=np.array(mesh['vertices']);faces=np.array(mesh['faces']);used=np.unique(faces)
        cam=(verts@e[0,:,:3].T+e[0,:,3]*scale)
        px=cam[used]@k[0].T;px=px[:,:2]/px[:,2,None]
        yy,xx=np.mgrid[0:depth.shape[1]:3,0:depth.shape[2]:3]
        expected=np.stack([xx,yy],-1).reshape(-1,2)[used]
        flags=np.array(mesh['face_inferred'],bool)
        rec={'id':cid,'case':row['case_id'],
             'mesh_roundtrip_max_px':float(np.linalg.norm(px-expected,axis=1).max()),
             'triangles':len(faces),'inferred_triangles':int(flags.sum()),
             'relative_jump_3pct_in_m_median':float(np.median(cam[used,2])*.03),
             'fusion_unknown_fraction':mesh['fusion_audit']['unknown_pixels']/depth[0].size,
             'state_metrics':row['evaluation'].get('state_metrics')}
        # Image7 overlay is projected correctly but has no foreground depth test.
        xyz7=verts@e[7,:,:3].T+e[7,:,3]*scale
        uv=xyz7@k[7].T;valid=xyz7[:,2]>0
        uv=uv[:,:2]/np.maximum(uv[:,2,None],1e-9)
        aff=crop_transform(masks.shape[1:])['affine']
        original=(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(aff).T)[:,:2]
        bitmap=np.zeros(masks.shape[1:],np.uint8)
        polys=[np.rint(original[f]).astype(np.int32) for f in faces if valid[f].all() and np.abs(original[f]).max()<1e5]
        cv2.fillPoly(bitmap,polys,1)
        rec['mesh_overlay_covers_RGB7_target_fraction']=float(bitmap[masks[7]].mean())
        case=Path(mapping[cid]['source_sample'])
        rec['matching_static_GT_depth_available']=(truth.parent.parent/'physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/cases'/case.name/'cycles_depth.npz').exists()
        if not row['support_reason']:
            meta=json.loads((truth/case.name/'truth_metadata.json').read_text())
            actors=json.loads((case/'metadata.json').read_text())['actors']
            with np.load(truth/case.name/'dynamic_masks.npz') as a:names=list(a['object_names'].astype(str))
            assert len(names)==1
            actor=names[0];radius=actors[actor]['size_m']['radius']
            with np.load(case/'raw/states_xyzw.npz') as a:
                ns=list(a['object_names'].astype(str));pos=a['positions'][:8,ns.index(actor)]
            eye=np.array(meta['camera']['location']);f=np.array(meta['camera']['target'])-eye;f/=np.linalg.norm(f)
            right=np.cross(f,[0,0,1]);right/=np.linalg.norm(right);R=np.stack([right,-np.cross(right,f),f]);gtcam=(pos-eye)@R.T
            width,height=meta['resolution'];focal=height/(2*np.tan(np.radians(meta['camera']['effective_yfov_deg'])/2))
            K=aff@np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]])
            rec['estimated_to_GT_focal_ratio']=float(np.median(k[:,0,0])/K[0,0])
            yy,xx=np.mgrid[:depth.shape[1],:depth.shape[2]];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(K).T
            aa=(rays*rays).sum(-1);ratios=[]
            for t,c in enumerate(gtcam):
                dot=rays@c;disc=dot**2-aa*(c@c-radius**2)
                valid=cv2.erode((disc>0).astype(np.uint8),np.ones((5,5),np.uint8))>0
                z=(dot-np.sqrt(np.maximum(disc,0)))/aa
                valid &= np.isfinite(depth[t])&(depth[t]>0)
                ratios.extend((z[valid]/depth[t][valid]).tolist())
            rec['GT_sphere_front_required_scale']=float(np.median(ratios)) if ratios else None
            rec['fixed_to_sphere_required_scale']=float(scale/np.median(ratios)) if ratios else None
            rec['scale_check_scope']='GT sphere visible front only, not static geometry or a correction'
        records.append(rec)
    out=root/'geometry_diagnosis_20260923_v1';out.mkdir(exist_ok=False)
    dump_json(out/'audit.json',{'records':records,'estimate_modified':False})
    for key in ['mesh_roundtrip_max_px','inferred_triangles','relative_jump_3pct_in_m_median','fusion_unknown_fraction','mesh_overlay_covers_RGB7_target_fraction','fixed_to_sphere_required_scale','estimated_to_GT_focal_ratio']:
        vals=[r[key] for r in records if r.get(key) is not None]
        print(key,'n',len(vals),'min/median/max',np.round([min(vals),np.median(vals),max(vals)],6).tolist())
    print('static_GT_depth_available',sum(r['matching_static_GT_depth_available'] for r in records))
    print('zero_completion_cases',sum(r['inferred_triangles']==0 for r in records))
    for cid in ['clip_000','clip_015','clip_030','clip_040','clip_054']:
        print(json.dumps(next(r for r in records if r['id']==cid)))

if __name__=='__main__':cv2.setNumThreads(2);main()
