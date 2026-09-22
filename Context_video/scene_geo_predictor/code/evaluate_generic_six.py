"""Post-freeze GT diagnostics. Never writes estimates or applies scale corrections."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from context_rgb_pybullet_common import dump_json,sha256_file,crop_transform
from generic_bullet_input import run

TRUTH=Path('/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/cases')
def evaluate(root):
    freeze=json.loads((root/'estimate_freeze.json').read_text())
    for path,digest in freeze.items():
        if sha256_file(root/path)!=digest:raise ValueError('estimate hash changed')
    protocol=json.loads((root/'protocol.json').read_text());scale=protocol['scale_config']['meters_per_reconstruction_unit']
    output=root/'evaluation_v3';output.mkdir(exist_ok=False);records=[]
    mapping=json.loads((root/'evaluation_mapping.json').read_text())
    for entry in mapping:
        key=entry['id'];sample=Path(entry['source_sample']);estdir=root/'estimates'/key
        result=json.loads((estdir/'result.json').read_text());row={'id':key,'source_case_eval_only':sample.name,'estimate_status':result['status'],
            'state_status':result['state']['status'],'state_reason':result['state'].get('reason',result['state'].get('reasons')),
            'gravity_status':result['gravity']['status'],'rollout':result['rollout'],'state_metrics':None}
        depth_truth=TRUTH/sample.name
        truth=TRUTH.parent.parent/'physv_v2v_0819_cycles_aligned_truth_v1/cases'/sample.name
        meta=json.loads((truth/'truth_metadata.json').read_text());source_meta=json.loads((sample/'metadata.json').read_text())
        row['GT_sources']={'masks':str(truth),'depth':str(depth_truth) if (depth_truth/'cycles_depth.npz').exists() else None}
        with np.load(truth/'dynamic_masks.npz',allow_pickle=False) as a:
            gm=a['union_thw'][:8];names=list(a['object_names'].astype(str))
        if len(names)!=1:raise ValueError('single GT actor required for eval')
        actor=names[0];radius=source_meta['actors'][actor]['size_m']['radius']
        with np.load(sample/'raw/states_xyzw.npz',allow_pickle=False) as a:
            ns=list(a['object_names'].astype(str));assert ns.count(actor)==1;i=ns.index(actor)
            pos=a['positions'][:8,i];vel=a['linear_velocities'][7,i]
        # Evaluation-only renderer camera, independently checked by saved projected centers below.
        eye=np.array(meta['camera']['location']);target=np.array(meta['camera']['target'])
        forward=(target-eye)/np.linalg.norm(target-eye);right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,forward)
        rotation=np.stack([right,-up,forward]);extrinsic=np.eye(4);extrinsic[:3,:3]=rotation;extrinsic[:3,3]=-rotation@eye
        gtcam=pos@extrinsic[:3,:3].T+extrinsic[:3,3]
        width,height=meta['resolution'];focal=height/(2*np.tan(np.radians(meta['camera']['effective_yfov_deg'])/2))
        kg=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1.]])
        with np.load(truth/'trajectory_pixels.npz') as a:reference_uv=a['centers_tnc'][:8,0,:2]
        projected=gtcam@kg.T;projected=projected[:,:2]/projected[:,2,None]
        # Saved values may be normalized; explicitly record conversion.
        if np.max(abs(reference_uv))<=1.5:reference_uv=reference_uv*np.array([width,height])
        row['GT_projection_check_pixels']=float(np.max(np.linalg.norm(projected-reference_uv,axis=1)))
        if row['GT_projection_check_pixels']>2:raise ValueError('GT camera projection mismatch')
        centroids=np.array([[np.where(m)[1].mean(),np.where(m)[0].mean()] for m in gm])
        flipped=centroids.copy();flipped[:,1]=height-1-flipped[:,1]
        before=float(np.median(np.linalg.norm(centroids-reference_uv,axis=1)))
        after=float(np.median(np.linalg.norm(flipped-reference_uv,axis=1)))
        row['GT_mask_axis_audit']={'raw_centroid_projection_error_px':before,'flipped_error_px':after,
                                  'transform':'vertical_flip' if after<before else 'identity','scope':'evaluation_only'}
        if after<before and after<2:gm=gm[:,::-1,:].copy()
        elif before>2:raise ValueError('GT mask coordinate convention unresolved')
        with np.load(estdir/'vggt_raw.npz') as a:depth=a['depth'][...,0];ke=a['intrinsic'];ee=a['extrinsic']
        with np.load(estdir/'tracking.npz') as a:mask=a['masks'];prompt=a['prompt']
        iou=[float(np.sum(a&b)/max(np.sum(a|b),1)) for a,b in zip(mask,gm)]
        row.update(mask_iou_mean=float(np.mean(iou)),mask_iou_frames=iou,gt_radius_m=radius)
        state=result['state']
        if 'radius' in state:
            p=ee[0,:,:3]@np.array(state['p7'])+ee[0,:,3]*scale
            v=ee[0,:,:3]@np.array(state['v7']);truev=extrinsic[:3,:3]@vel
            centers=np.array(state['center_sequence'])@ee[0,:,:3].T+ee[0,:,3]*scale
            row['state_metrics']={'admitted':state['status']=='ESTIMATED','frame':'camera0 optical coordinates, rigid GT camera evaluation only; no scale fit',
                'p7_error_m':float(np.linalg.norm(p-gtcam[7])),'center_sequence_mean_error_m':float(np.linalg.norm(centers-gtcam,axis=1).mean()),
                'v7_vector_error_mps':float(np.linalg.norm(v-truev)),'radius_est_m':state['radius'],'radius_error_m':abs(state['radius']-radius)}
        if not (depth_truth/'cycles_depth.npz').exists():
            # No substitute raw renderer depth: it is a different camera.
            row['static_geometry_error']={'status':'BLOCKED','reason':'matching Cycles GT static depth unavailable'}
            h,w=gm.shape[1:];transform=crop_transform((h,w));kp=transform['affine']@kg
            ph,pw=depth.shape[1:];yy,xx=np.mgrid[:ph,:pw];rays=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(kp).T
            ratios=[]
            for t in range(8):
                c=gtcam[t];aa=(rays*rays).sum(-1);dot=rays@c;disc=dot**2-aa*(c@c-radius**2)
                nh,nw=transform['resized_hw'];top=transform['crop_top']
                maskgt=cv2.resize(gm[t].astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)[top:top+ph,:pw]
                valid=(disc>0)&(cv2.erode(maskgt,np.ones((3,3),np.uint8))>0)
                z=(dot-np.sqrt(np.maximum(disc,0)))/aa
                ratios.extend((z[valid]/depth[t][valid]).tolist())
            row['scale']={'status':'EVALUATED_SPHERE_ONLY','fixed':scale,'GT_diagnostic_required_median':float(np.median(ratios)),
                          'fixed_to_required_ratio':float(scale/np.median(ratios)),'support_pixels':len(ratios),
                          'note':'analytic GT sphere front at GT camera pixels; EVALUATION ONLY, no scale correction; no static-scene test'} if ratios else {'status':'BLOCKED','reason':'no valid GT sphere front pixels'}
            gd=None
        else:
            with np.load(depth_truth/'cycles_depth.npz') as a:gd=a['depth'][:8]
        h,w=gm.shape[1:];yy,xx=np.mgrid[:h,:w];ray=np.stack([xx,yy,np.ones_like(xx)],-1)@np.linalg.inv(kg).T
        raynorm=np.linalg.norm(ray,axis=-1)
        # Independently verify Depth pass semantics against known sphere front intersections.
        checks=[]
        for t in range(8) if gd is not None else []:
            c=gtcam[t];aa=(ray*ray).sum(-1);dot=ray@c;disc=dot**2-aa*(c@c-radius**2)
            central=(disc>0)&(cv2.erode(gm[t].astype(np.uint8),np.ones((5,5),np.uint8))>0)
            expected=(dot-np.sqrt(np.maximum(disc,0)))/aa
            valid=central&(gd[t]>0)&(gd[t]<100)
            checks.append([float(np.median(abs(gd[t][valid]-expected[valid]))),float(np.median(abs(gd[t][valid]-expected[valid]*raynorm[valid])))])
        if checks:
            zerr,rerr=np.median(checks,axis=0);semantics='ray_distance' if rerr<zerr else 'camera_z'
            row['gt_depth_semantics_check']={'selected':semantics,'camera_z_sphere_error_m':float(zerr),'ray_distance_sphere_error_m':float(rerr)}
        if gd is None:pass
        elif min(zerr,rerr)>.03:
            row['scale']={'status':'BLOCKED','reason':'GT depth semantics/projection not validated'}
        else:
            gt_z=gd/raynorm[None] if semantics=='ray_distance' else gd
            ph,pw=depth.shape[1:];transform=crop_transform((h,w));gtprocessed=[];validm=[]
            for t in range(8):
                nh,nw=transform['resized_hw'];top=transform['crop_top']
                gz=cv2.resize(gt_z[t],(nw,nh),interpolation=cv2.INTER_NEAREST)[top:top+ph,:pw]
                sm=cv2.resize(gm[t].astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)[top:top+ph,:pw]
                gtprocessed.append(gz);validm.append((gz>.05)&(gz<30)&(sm==0)&np.isfinite(depth[t])&(depth[t]>0))
            gtprocessed=np.array(gtprocessed);validm=np.array(validm)
            ratios=gtprocessed[validm]/depth[validm];err=(depth*scale-gtprocessed)[validm]
            row['scale']={'status':'EVALUATED','fixed':scale,'GT_diagnostic_required_median':float(np.median(ratios)),
                'fixed_to_required_ratio':float(scale/np.median(ratios)),
                'required_ratio_p10_p90':np.percentile(ratios,[10,90]).tolist(),
                'static_depth_absrel':float(np.mean(abs(err)/gtprocessed[validm])),
                'static_depth_abs_error_median_m':float(np.median(abs(err))),
                'note':'GT diagnostic only; no correction or rerun'}
        if (estdir/'collision_primitive.json').exists() and state['status']=='ESTIMATED':
            mesh=json.loads((estdir/'collision_primitive.json').read_text())
            row['initialization']=run(state,[mesh],protocol['physics'],result['gravity'],advance=False)
        else:row['initialization']={'status':'NOT_RUN','reason':'state failed; do not promote candidate','initial_overlap':None}
        sheet=Image.new('RGB',(4*448,2*280),'#172030');draw=ImageDraw.Draw(sheet)
        for t in range(8):
            rgb=np.array(Image.open(root/'inputs'/key/f'rgb_{t:02d}.png'))
            for m,color in [(gm[t],(30,240,100)),(mask[t],(255,50,80))]:
                contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE);cv2.drawContours(rgb,contours,-1,color,1)
            image=Image.fromarray(rgb);image.thumbnail((448,256));x=t%4*448;y=t//4*280;sheet.paste(image,(x,y+24));draw.text((x+4,y+4),f'{key} RGB{t} red SAM / green GT eval',fill='white')
        sheet.save(output/f'{key}_tracking.jpg');records.append(row);print(key,row.get('scale'),row['state_metrics'],flush=True)
    report={'status':'EXECUTED','complete_physics_inputs':0,'rollouts':0,'attempts':6,'records':records,
            'freeze_sha256':sha256_file(root/'estimate_freeze.json'),'fixed_scale_applied':scale,
            'geometry_metric_scope':'static pixel depth error, not complete collision topology accuracy',
            'gravity_implementation_limit':'normal-axis extraction only; no validated observation-only axis/sign disambiguator'}
    dump_json(output/'report.json',report)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2);evaluate(a.root)
