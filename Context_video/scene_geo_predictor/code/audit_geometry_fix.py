"""Frozen-output regression and observation coverage audit, no state changes."""
import argparse,json
from pathlib import Path
import numpy as np
from context_rgb_pybullet_common import dump_json,crop_transform,resize_crop_mask
from run_grounded_generic_pilot36 import verify

def audit(root,baseline):
    for r in [root,baseline]:
        verify(r,'estimate_freeze.json');verify(r,'rollout_freeze.json')
    new=json.loads((root/'viewer_data.json').read_text())['records']
    old={x['id']:x for x in json.loads((baseline/'viewer_data.json').read_text())['records']}
    gravity={x['id']:x for x in json.loads((root/'lower_plane_gravity_v1/candidates.json').read_text())}
    scale=json.loads((root/'protocol.json').read_text())['scale_config']['meters_per_reconstruction_unit']
    rows=[]
    for row in new:
        key=row['id'];folder=root/'estimates'/key
        s=json.loads((folder/'result.json').read_text())['state']
        previous=json.loads((baseline/'estimates'/key/'result.json').read_text())['state']
        assert s==previous,'state changed in geometry-only ablation'
        a=row['trajectory_metrics'];b=old[key]['trajectory_metrics']
        entry={'id':key,'case_id':row['case_id'],'family':row['family'],'state_unchanged':True,
               'status':row['rollout_status'],'old_ADE_m':b['ADE_m'] if b else None,
               'new_ADE_m':a['ADE_m'] if a else None,'delta_ADE_m':a['ADE_m']-b['ADE_m'] if a and b else None,
               'contact_frames':row.get('rollout_contact_frames'),'support_coverage':None}
        selected=gravity[key]['frames'][0]['selected']
        if selected and 'p7' in s:
            with np.load(folder/'vggt_raw.npz') as z:k=z['intrinsic'];e=z['extrinsic']
            with np.load(folder/'tracking.npz') as z:m=z['masks']
            m=np.stack([resize_crop_mask(x,crop_transform(m.shape[1:])) for x in m])
            n=np.array(selected['normal_camera_facing']);c=e[0,:,:3]@s['p7']+e[0,:,3]*scale
            height=float(n@c+selected['offset']);candidate=(c-height*n-e[0,:,3]*scale)@e[0,:,:3]
            views=[]
            for t in range(8):
                cam=e[t,:,:3]@candidate+e[t,:,3]*scale;p=k[t]@cam
                if p[2]<=0:views.append('OUT_OF_VIEW');continue
                x,y=np.rint(p[:2]/p[2]).astype(int)
                views.append('OUT_OF_VIEW' if x<0 or y<0 or x>=m.shape[2] or y>=m.shape[1] else 'TARGET_OCCLUDED' if m[t,y,x] else 'UNMASKED')
            entry['support_coverage']={'status':'DIAGNOSTIC_NOT_CONTACT_CLASSIFICATION',
                'source':'projection onto observed lower-image candidate plane, not asserted to be support',
                'sphere_clearance_from_candidate_plane_m':height-s['radius'],'views':views,
                'target_occluded_all_frames':all(x=='TARGET_OCCLUDED' for x in views)}
        rows.append(entry)
    compared=[x for x in rows if x['delta_ADE_m'] is not None]
    summary={'cases':len(rows),'paired_rollouts':len(compared),
        'improved_gt_1cm':sum(x['delta_ADE_m']<-.01 for x in compared),
        'regressed_gt_1cm':sum(x['delta_ADE_m']>.01 for x in compared),
        'within_1cm':sum(abs(x['delta_ADE_m'])<=.01 for x in compared),
        'old_mean_ADE_m':float(np.mean([x['old_ADE_m'] for x in compared])),
        'new_mean_ADE_m':float(np.mean([x['new_ADE_m'] for x in compared])),
        'cases_with_recorded_contact':sum(bool(x['contact_frames']) for x in compared),
        'occluded_candidate_support_all_frames':sum(bool(x['support_coverage'] and x['support_coverage']['target_occluded_all_frames']) for x in rows)}
    dump_json(root/'geometry_fix_audit.json',{'summary':summary,'records':rows})
    report='''# 2026-09-23 通用几何修复：观测融合与剩余限制

已确认并修复：单帧深度不应删除其他帧动态物体经过的全部区域。
新入口observed_multiframe逐帧排除本帧mask，按VGGT相机重投影，再融合至少两帧一致的静态表面。
不读取family/case参数、不补默认平台、不改状态/尺度/物性/solver。

合成测试：旧实现无法命中本应可见的平面（FAIL）；新实现通过移动遮挡恢复、真实缺口保持、动态表面排除、相机变换与尺度测试；原3个状态/Bullet测试通过。

这修复了几何删除错误，但没有证明完整轨迹输入准确。严格保持未观测区未知时，持续遮挡的球下区域无法靠多帧融合恢复。候选平面下投影仅用于审计，不当作球一定受支撑的证据。
局部平面补全是另外的观察侧先验，需要用户确认后才能启用，当前未实施。

## 完整36例配对回归

```json
'''+json.dumps(summary,ensure_ascii=False,indent=2)+'''
```

| case | 旧ADE | 新ADE | 差值 | 接触帧 | 状态 |
|---|---:|---:|---:|---:|---|
'''
    for x in rows:
        def fmt(v):return 'NOT_RUN' if v is None else f'{v:.4f}'
        report+=f"| {x['case_id']} | {fmt(x['old_ADE_m'])} | {fmt(x['new_ADE_m'])} | {fmt(x['delta_ADE_m'])} | {x['contact_frames']} | {x['status']} |\n"
    (root/'geometry_fix_report.md').write_text(report)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();audit(a.root,a.baseline)
