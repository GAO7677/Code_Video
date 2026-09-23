"""Post-freeze paired audit of the explicitly authorized completion prior."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
from context_rgb_pybullet_common import dump_json
from run_grounded_generic_pilot36 import verify

def report(root,baseline):
    for directory in [root,baseline]:
        verify(directory,'estimate_freeze.json');verify(directory,'rollout_freeze.json')
    rows=json.loads((root/'viewer_data.json').read_text())['records']
    old={r['id']:r for r in json.loads((baseline/'viewer_data.json').read_text())['records']}
    records=[];reasons=Counter()
    for r in rows:
        key=r['id'];new_mesh=json.loads((root/'estimates'/key/'collision_primitive.json').read_text())
        old_mesh=json.loads((baseline/'estimates'/key/'collision_primitive.json').read_text())
        used=np.unique(np.array(old_mesh['faces']).ravel())
        assert np.array_equal(np.array(old_mesh['vertices'])[used],np.array(new_mesh['vertices'])[used]),'observed vertex moved'
        a=json.loads((root/'estimates'/key/'result.json').read_text())['state']
        b=json.loads((baseline/'estimates'/key/'result.json').read_text())['state']
        assert a==b,'state changed'
        audit=new_mesh['completion_audit'];reasons.update(c['reason'] for c in audit['components'] if c['status']!='INFERRED')
        t=r['trajectory_metrics'];previous=old[key]['trajectory_metrics']
        records.append({'id':key,'case_id':r['case_id'],'family':r['family'],
            'inferred_pixels':audit['inferred_pixels'],'inferred_triangles':sum(new_mesh['face_inferred']),
            'state_unchanged':True,'observed_vertices_unchanged':True,
            'old_ADE_m':previous['ADE_m'] if previous else None,'new_ADE_m':t['ADE_m'] if t else None,
            'old_FDE_m':previous['FDE_m'] if previous else None,'new_FDE_m':t['FDE_m'] if t else None,
            'old_contact_frames':old[key]['rollout_contact_frames'],'new_contact_frames':r['rollout_contact_frames'],
            'initial_penetration_m':r['initial_penetration_m'],'status':r['rollout_status']})
    paired=[r for r in records if r['new_ADE_m'] is not None and r['old_ADE_m'] is not None]
    summary={'cases':len(records),'paired_rollouts':len(paired),'state_failures':len(records)-len(paired),
        'cases_with_inferred_pixels':sum(r['inferred_pixels']>0 for r in records),
        'cases_with_inferred_triangles':sum(r['inferred_triangles']>0 for r in records),
        'completed_rollouts_with_inferred_triangles':sum(r['inferred_triangles']>0 for r in paired),
        'old_ADE_m':float(np.mean([r['old_ADE_m'] for r in paired])),
        'new_ADE_m':float(np.mean([r['new_ADE_m'] for r in paired])),
        'improved_gt_1cm':sum(r['new_ADE_m']<r['old_ADE_m']-.01 for r in paired),
        'regressed_gt_1cm':sum(r['new_ADE_m']>r['old_ADE_m']+.01 for r in paired),
        'within_1cm':sum(abs(r['new_ADE_m']-r['old_ADE_m'])<=.01 for r in paired),
        'rejected_component_reasons':dict(reasons),'state_and_observed_mesh_unchanged':True}
    dump_json(root/'completion_comparison.json',{'summary':summary,'records':records})
    text='''# 2026-09-23 显式局部平面补全模式

用户授权后实现，默认仍是纯观测模式；必须显式传`--geometry-mode local_plane_completion`。
候选仅来自参考帧目标遮挡区中的未知深度，所有family共用同一逻辑；无GT/文件名/默认平台/固定球半径。
采用周围环带局部平面拟合，要求观测覆盖≥90%、至少40点、8方向各≥3点、RMS/深度≤0.005、P95/深度≤0.01。拒绝图像边界、单侧证据、台阶/非平面及超出邻域深度范围的外推。
只改未知像素，保留真实观测顶点；补全后的面以face_inferred单独标注。补全的是局部表面，不是给球增加约束或自动贴地。

局部平面先验不能证明遮挡背后一定没有真实缺口；完全被遮挡的缺口在当前输入下不可辨识。可见缺口/台阶有反证时拒绝，证据不足保持UNKNOWN。未放宽阈值来提高本批通过率。

## 验证

`tests/test_local_plane_completion.py`验证平面、两种倾斜平面、可见真实缺口、深度台阶、开放边缘和已观测像素不变；原几何融合2项及状态/时间步3项测试均通过。
36例均尝试；冻结后评测，逐例确认球状态和原观测mesh顶点完全未改。推理只用已有冻结视觉输出，CPU两线程，无GPU训练或动力学修改。

## 与纯观测多帧mesh对照

```json
'''+json.dumps(summary,ensure_ascii=False,indent=2)+'''
```

| case | 补全面数 | 旧ADE | 新ADE | 旧接触帧 | 新接触帧 | 状态 |
|---|---:|---:|---:|---:|---:|---|
'''
    def fmt(v):return 'NOT_RUN' if v is None else f'{v:.4f}'
    for r in records:text+=f"| {r['case_id']} | {r['inferred_triangles']} | {fmt(r['old_ADE_m'])} | {fmt(r['new_ADE_m'])} | {r['old_contact_frames']} | {r['new_contact_frames']} | {r['status']} |\n"
    text+='\n补全逻辑EXECUTED，完整链路仍PARTIAL：深度/尺度与球状态误差未修复；接触帧增加不等于contact outcome正确，GT contact accuracy本轮NOT_EVALUATED。\n'
    (root/'completion_report.md').write_text(text);print(json.dumps(summary,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();report(a.root,a.baseline)
