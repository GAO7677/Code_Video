"""Read-only replay of completion inputs; diagnostic plots, no estimator edits."""
import sys
import inspect
import json
from pathlib import Path
import cv2
import numpy as np
from context_rgb_pybullet_common import crop_transform, resize_crop_mask, dump_json
from generic_context_geometry import finite_mesh_observed
from segmented_plane_completion import complete_segmented_depth

root = Path(sys.argv[1]); cid = sys.argv[2]; out = Path(sys.argv[3])
out.mkdir(exist_ok=False)
cv2.setNumThreads(2)
folder = root/'estimates'/cid
with np.load(folder/'vggt_raw.npz') as a:
    depth, k, e = a['depth'][...,0], a['intrinsic'], a['extrinsic']
with np.load(folder/'tracking.npz') as a:
    original = a['masks']
tr = crop_transform(original.shape[1:])
masks = np.stack([resize_crop_mask(m, tr) for m in original])
scale = json.loads((folder/'result.json').read_text())['scale']
source, start = inspect.getsourcelines(complete_segmented_depth)
line = start + next(i for i, s in enumerate(source) if 'if inliers.mean() < .65' in s)
captures = []
def trace(frame, event, arg):
    if frame.f_code is complete_segmented_depth.__code__ and event == 'line' and frame.f_lineno == line:
        f = frame.f_locals
        captures.append({key: f[key].copy() if isinstance(f[key], np.ndarray) else f[key]
                         for key in ['region','support','inliers','points','normal','offset','depth_scale','label','z']})
    return trace
sys.settrace(trace)
try:
    mesh, _ = finite_mesh_observed(depth,k,e,masks,scale,complete_local_planes=True,segmented_completion=True)
finally:
    sys.settrace(None)
rgb = cv2.imread(str(root/'inputs'/cid/'rgb_00.png'))
rgb = cv2.warpAffine(rgb,np.asarray(tr['affine'])[:2],(depth.shape[2],depth.shape[1]))
records = []
for f in captures:
    region, support, inliers = f['region'], f['support'], f['inliers']
    overlay = rgb.copy()
    selected = np.zeros_like(support); selected[support] = inliers
    overlay[region] = (0,220,255)
    overlay[selected] = (40,230,40)
    overlay[support & ~selected] = (40,40,240)
    overlay = cv2.addWeighted(rgb,.35,overlay,.65,0)
    y,x = np.where(region|support)
    x0,x1=max(0,x.min()-20),min(rgb.shape[1],x.max()+21)
    y0,y1=max(0,y.min()-20),min(rgb.shape[0],y.max()+21)
    panel = np.concatenate([rgb[y0:y1,x0:x1],overlay[y0:y1,x0:x1]],axis=1)
    panel = cv2.resize(panel,None,fx=4,fy=4,interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out/f'component_{f["label"]}.png'),panel)
    cv2.imwrite(str(out/f'full_{f["label"]}.png'),overlay)
    residual = abs(f['points']@f['normal']+f['offset'])*scale
    records.append({'component':int(f['label']),'hole_pixels':int(region.sum()),
                    'support_pixels':int(support.sum()),'inlier_fraction':float(inliers.mean()),
                    'plane_tolerance_m':float(.005*f['depth_scale']*scale),
                    'ring_residual_m_p50_p90_p95':np.percentile(residual,[50,90,95]).tolist(),
                    'ring_depth_m_p10_p50_p90':np.percentile(f['z'][support]*scale,[10,50,90]).tolist(),
                    'pixel_bbox_xyxy':[int(x0),int(y0),int(x1),int(y1)]})
dump_json(out/'diagnosis.json',{'records':records,'audit':mesh['completion_audit'],
                             'legend':'yellow unknown hole; green plane inliers; red rejected ring samples',
                             'GT_used':False,'modified_estimation':False})
print(json.dumps(records,indent=2))
