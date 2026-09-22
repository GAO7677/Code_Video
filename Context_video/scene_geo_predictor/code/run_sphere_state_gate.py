"""Separate estimate/freeze and evaluation commands for the fixed 36-case gate."""
import argparse
import json
from pathlib import Path
import numpy as np
from context_rgb_pybullet_common import dump_json, sha256_file
from sphere_state_recovery import fit_mask, motion_fits

ROOT=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')

def estimate(output):
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    protocol={'radius_m':.11, 'prompt_source':'rgb_auto_prompt', 'prompt_frame':7,
              'mask_source':'frozen SAM2.1 bidirectional tracking; RGB-only circle prompt',
              'selected_motion_method':'robust_linear', 'contour':'marching squares at mask level 0.5; all components',
              'sphere_loss':'soft_l1, f_scale=0.002 m', 'motion_loss':'soft_l1, f_scale=0.01 m',
              'thresholds':{'p7_median_m':.03,'p7_p90_m':.05,'v7_vector_median_mps':.08,'direction_median_deg':5},
              'geometry_and_rollout':'NOT_RUN until state gate passes',
              'auto_prompt_priors':'single moving sphere; Hough radius 6..22 pixels; RGB0..7 only'}
    dump_json(output/'protocol.json',protocol)
    manifest=json.loads((ROOT/'vision_inputs/manifest.json').read_text())
    source=json.loads((ROOT/'vision_v1/inference_report.json').read_text())
    # Check frozen inference provenance, rather than relabelling arbitrary masks.
    sam=source['sam2'] if 'sam2' in source else source['sam2_tracking']
    provenance={r['case_id']:r for r in sam['records']}
    freeze={}
    for record in manifest['records']:
        key=record['case_id']  # opaque lookup identifier; never parsed for parameters
        folder=ROOT/'vision_inputs'/record['input_dir']
        payload=json.loads((folder/'input.json').read_text())
        calibration=json.loads((folder/'calibration.json').read_text())
        mask_path=ROOT/'vision_v1/sam2_masks'/f'{key}.npz'
        row={'case_id':key, 'prompt_source':'rgb_auto_prompt', 'status':'FAIL',
             'flags':{name:False for name in ['mask_fallback','bbox_fallback','default_radius_fallback',
             'family_template_geometry_fallback','default_surface_height_fallback','blueprint_read_during_estimation',
             'gt_scale_fallback','sample_key_parser_used','geometry_value_parser_used']},
             'primitive_fit_failed':None,'primitive_status':'NOT_RUN',
             'inputs':{str(folder/'input.json'):sha256_file(folder/'input.json'),
                       str(folder/'calibration.json'):sha256_file(folder/'calibration.json'),
                       str(mask_path):sha256_file(mask_path)}}
        try:
            prov=provenance[key]
            if prov['prompt_frame']!=7 or prov['prompt_mode']!='RGB_motion_circle_then_SAM2_bidirectional' or prov['sha256']!=sha256_file(mask_path):
                raise ValueError('frozen mask provenance mismatch')
            with np.load(mask_path,allow_pickle=False) as archive:
                masks=archive['masks']
            if masks.shape!=(8,360,640):
                raise ValueError('mask shape mismatch')
            k=np.array(calibration['intrinsic_K']); e=np.array(calibration['world_to_camera_3x4'])
            centers=[]; legacy=[]; fits=[]
            for mask in masks:
                c,b,f=fit_mask(mask,k,e,.11); centers.append(c); legacy.append(b); fits.append(f)
            times=np.array(payload['time_s'])
            row.update(status='EXECUTED', centers=np.array(centers).tolist(), frame_fit=fits,
                       methods=motion_fits(np.array(centers),times),
                       legacy_sphere_depth=motion_fits(np.array(legacy),times),
                       selected_method='robust_linear',prompt=prov)
        except (ValueError,KeyError) as exc:
            row['reason']=str(exc)
        path=output/'estimates'/f'{key}.json'; dump_json(path,row)
        freeze[str(path.relative_to(output))]=sha256_file(path)
    freeze['protocol.json']=sha256_file(output/'protocol.json')
    dump_json(output/'freeze.json',freeze)
    print('FROZEN',len(manifest['records']),sha256_file(output/'freeze.json'))

def evaluate(output, gt_root):
    freeze=json.loads((output/'freeze.json').read_text())
    for name,digest in freeze.items():
        if sha256_file(output/name)!=digest:
            raise ValueError('estimate freeze mismatch')
    manifest=json.loads((gt_root/'pilot_manifest.json').read_text())
    records=[]
    for item in manifest['records']:
        row=json.loads((output/'estimates'/f"{item['key']}.json").read_text())
        result={'case_id':item['key'],'family':item['family'],'group_id':item['group_id'],'status':row['status'],'methods':{}}
        if row['status']=='EXECUTED':
            with np.load(Path(item['sample_dir'])/'raw/states_xyzw.npz',allow_pickle=False) as data:
                names=list(data['object_names'].astype(str)); assert names.count('pilot_ball')==1
                i=names.index('pilot_ball'); p=data['positions'][7,i]; v=data['linear_velocities'][7,i]
            for source,prefix in [('methods','sphere/'),('legacy_sphere_depth','legacy/')]:
                for method,values in row[source].items():
                    pe=np.array(values['p7']); ve=np.array(values['v7'])
                    denominator=np.linalg.norm(v)*np.linalg.norm(ve)
                    direction=float(np.degrees(np.arccos(np.clip(np.dot(v,ve)/denominator,-1,1)))) if denominator>1e-10 else None
                    result['methods'][prefix+method]={'p7_error_m':float(np.linalg.norm(pe-p)),
                        'v7_vector_error_mps':float(np.linalg.norm(ve-v)),
                        'v7_magnitude_error_mps':float(abs(np.linalg.norm(ve)-np.linalg.norm(v))),
                        'direction_error_deg':direction,'position_delta_m':(pe-p).tolist()}
        records.append(result)
    def summarize(rows,method):
        vals=[r['methods'][method] for r in rows if method in r['methods']]
        if not vals: return {'status':'FAIL','count':0}
        metrics={}
        for field in ['p7_error_m','v7_vector_error_mps','v7_magnitude_error_mps','direction_error_deg']:
            x=[v[field] for v in vals if v[field] is not None]
            metrics[field]={'median':float(np.median(x)),'p90':float(np.percentile(x,90)),'mean':float(np.mean(x))} if x else None
        passed=(len(vals)==len(rows) and metrics['p7_error_m']['median']<=.03 and metrics['p7_error_m']['p90']<=.05
                and metrics['v7_vector_error_mps']['median']<=.08 and metrics['direction_error_deg'] is not None
                and metrics['direction_error_deg']['median']<=5)
        return {'status':'PASS' if passed else 'FAIL','count':len(vals),'metrics':metrics}
    methods=['sphere/'+m for m in ['last2','robust_linear','robust_quadratic']]+['legacy/'+m for m in ['last2','robust_linear','robust_quadratic']]
    summary={m:summarize(records,m) for m in methods}
    families={f:summarize([r for r in records if r['family']==f],'sphere/robust_linear') for f in sorted({r['family'] for r in records})}
    groups={g:summarize([r for r in records if r['group_id']==g],'sphere/robust_linear') for g in sorted({r['group_id'] for r in records})}
    report={'status':'EXECUTED','gate':summary['sphere/robust_linear']['status'], 'freeze_sha256':sha256_file(output/'freeze.json'),
            'unique_groups':len(groups),'case_count':len(records),'summary':summary,'families':families,'groups':groups,'records':records,
            'downstream':'NOT_RUN','note':'method fixed before GT; correlated variants are not independent histories'}
    dump_json(output/'evaluation.json',report)
    print(json.dumps({k:report[k] for k in ['gate','case_count','unique_groups','summary','families']},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('phase',choices=['estimate','evaluate'])
    parser.add_argument('--output',type=Path,required=True); parser.add_argument('--gt-root',type=Path)
    args=parser.parse_args()
    if args.phase=='estimate':
        if args.gt_root: parser.error('GT is prohibited for estimation')
        estimate(args.output)
    else:
        if not args.gt_root: parser.error('evaluation requires --gt-root')
        evaluate(args.output,args.gt_root)
