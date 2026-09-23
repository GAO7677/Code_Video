"""Post-freeze diagnostic artifacts; never feeds evaluation back into estimation."""
import argparse
import json
import shutil
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from context_rgb_pybullet_common import dump_json, crop_transform, resize_crop_mask
from run_grounded_generic_pilot36 import verify


def publish(root):
    cv2.setNumThreads(2)
    for name in ['estimate_freeze.json', 'rollout_freeze.json']:
        verify(root, name)
    data = json.loads((root / 'viewer_data.json').read_text())
    base = Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
    gtroot = Path('/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5/samples')
    for row in data['records']:
        clip = row['id']; folder = root / 'estimates' / clip
        est = json.loads((folder / 'result.json').read_text()); state = est['state']
        row['target_detection'] = est['target_detection']
        row['fit_diagnostics'] = {key: state.get(key) for key in ['reasons', 'radius', 'radius_cv', 'relative_rms', 'jacobian_condition', 'frames', 'p7', 'v7']}
        row['fallback'] = est['fallback']
        with np.load(folder / 'tracking.npz') as a: masks = a['masks']
        with np.load(folder / 'vggt_raw.npz') as a: depth, k, e = a['depth'][..., 0], a['intrinsic'], a['extrinsic']
        transform = crop_transform(masks.shape[1:]); affine = transform['affine']
        processed = np.stack([resize_crop_mask(m, transform) for m in masks])
        dest = root / 'diagnostics' / clip; dest.mkdir(parents=True, exist_ok=True)
        lo, hi = np.percentile(depth[np.isfinite(depth)], [2, 98])
        for t in range(8):
            color = cv2.applyColorMap(np.uint8(np.clip((depth[t]-lo)/(hi-lo), 0, 1)*255), cv2.COLORMAP_TURBO)
            contours = cv2.findContours(processed[t].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(color, contours, -1, (180, 60, 255), 1)
            cv2.imwrite(str(dest / f'depth_{t}.png'), color)
        mesh = json.loads((folder / 'collision_primitive.json').read_text())
        faces = np.array(mesh['faces'])
        edges = np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
        unique, counts = np.unique(edges,axis=0,return_counts=True)
        boundary = unique[counts==1]
        cam7 = np.array(mesh['vertices']) @ e[7,:,:3].T + e[7,:,3]*est['scale']
        pixel7 = cam7 @ k[7].T
        pixel7 = np.c_[pixel7[:,:2]/pixel7[:,2,None],np.ones(len(pixel7))] @ np.linalg.inv(affine).T
        row['mesh_boundary_segments_uv'] = pixel7[boundary,:2].tolist()
        cam = np.array(mesh['vertices']) @ e[0, :, :3].T + e[0, :, 3]*est['scale']
        pix = cam @ k[0].T; pix = pix[:, :2]/pix[:, 2, None]
        original = np.c_[pix, np.ones(len(pix))] @ np.linalg.inv(affine).T
        rgb = cv2.imread(str(root / 'inputs' / clip / 'rgb_00.png')); coverage = rgb.copy()
        polygons = [np.rint(original[f, :2]).astype(np.int32) for f in mesh['faces']]
        cv2.fillPoly(coverage, polygons, (230, 210, 40))
        cv2.imwrite(str(dest / 'mesh.png'), cv2.addWeighted(rgb, .45, coverage, .55, 0))
        if 'center_sequence' in state:
            centers = np.array(state['center_sequence'])
            cams = np.einsum('tij,tj->ti', e[:, :, :3], centers) + e[:, :, 3]*est['scale']
            p = np.einsum('tij,tj->ti', k, cams); p = p[:, :2]/p[:, 2, None]
            row['center_uv'] = (np.c_[p, np.ones(8)] @ np.linalg.inv(affine).T)[:, :2].tolist()
        calibration = json.loads((base/'vision_inputs'/row['case_id']/'calibration.json').read_text())
        E, K = np.array(calibration['world_to_camera_3x4']), np.array(calibration['intrinsic_K'])
        with np.load(gtroot/row['case_id']/'raw/states_xyzw.npz') as a:
            names = list(a['object_names'].astype(str)); assert names.count('pilot_ball') == 1
            idx = names.index('pilot_ball'); gt = a['positions'][:49, idx]
        gtcam = gt @ E[:, :3].T + E[:, 3]
        roll = json.loads((root/'rollouts'/f'{clip}.json').read_text())
        row['rollout_contact_frames'] = sum(bool(c) for c in roll.get('contacts', [])) if roll['status']=='EXECUTED' else None
        row['initial_overlap_status'] = ('OVERLAP' if roll['initial_overlap'] else 'NO_OVERLAP') if 'initial_overlap' in roll else 'NOT_RUN'
        if row['trajectory_metrics']:
            pred = np.array(roll['positions']) @ e[0, :, :3].T + e[0, :, 3]*est['scale']
            p7 = e[0, :, :3] @ state['p7'] + e[0, :, 3]*est['scale']
            v7 = e[0, :, :3] @ state['v7']
            cv = p7 + np.arange(1,42)[:,None]/30*v7
            cv_err = np.linalg.norm(cv-gtcam[8:49], axis=1)
            disp = np.linalg.norm((pred-p7)-(gtcam[8:49]-gtcam[7]), axis=1)
            def project(x):
                p=x@K.T; return p[:, :2]/p[:, 2, None]
            row['cv_future_uv'] = project(cv).tolist()
            uv_err = np.linalg.norm(project(pred)-project(gtcam[8:49]), axis=1)
            row['trajectory_metrics'].update(CV_ADE_m=float(cv_err.mean()), CV_FDE_m=float(cv_err[-1]),
                    displacement_ADE_m=float(disp.mean()), displacement_FDE_m=float(disp[-1]),
                    projected_2D_mean_px=float(uv_err.mean()), projected_2D_FDE_px=float(uv_err[-1]),
                    frame_errors_m=np.linalg.norm(pred-gtcam[8:49], axis=1).tolist())
    records = data['records']; executed = [r for r in records if r['trajectory_metrics']]
    def aggregate(rows):
        run = [r for r in rows if r['trajectory_metrics']]
        return {'cases': len(rows), 'state_admitted': sum(r['state_status']=='ESTIMATED' for r in rows),
                'executed': len(run), 'p7_median_m_all_attempts': float(np.median([r['state_metrics']['p7_error_m'] for r in rows])),
                'v7_median_mps_all_attempts': float(np.median([r['state_metrics']['v7_error_mps'] for r in rows])),
                'radius_median_m_all_attempts': float(np.median([r['state_metrics']['radius_m'] for r in rows])),
                'ADE_mean_m_executed_only': float(np.mean([r['trajectory_metrics']['ADE_m'] for r in run])) if run else None,
                'FDE_mean_m_executed_only': float(np.mean([r['trajectory_metrics']['FDE_m'] for r in run])) if run else None,
                'CV_ADE_mean_m_same_executed': float(np.mean([r['trajectory_metrics']['CV_ADE_m'] for r in run])) if run else None}
    data['summary'].update(aggregate(records))
    data['summary']['initial_overlap'] = {'positive': sum(r['initial_overlap_status']=='OVERLAP' for r in records), 'tested': len(executed), 'not_run': 36-len(executed)}
    data['summary']['initial_contact_cases'] = sum(bool(r['initial_contacts']) for r in executed)
    data['summary']['family_results'] = {f: aggregate([r for r in records if r['family']==f]) for f in sorted({r['family'] for r in records})}
    data['summary']['unique_history_groups'] = len({(r['family'],r['group_id']) for r in records})
    data['summary']['per_group'] = {f'{f}/{g}': aggregate([r for r in records if (r['family'],r['group_id'])==(f,g)]) for f,g in sorted({(r['family'],r['group_id']) for r in records})}
    dump_json(root/'viewer_data.json', data)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'web/grounded_generic_pipeline.html', root/'index.html')
    summary = data['summary']
    report = f'''# Grounding DINO＋SAM2 接入通用 pipeline · 2026-09-23

36例全部尝试；20例执行仿真，16例因球面残差/半径稳定性失败。完整链路准确性 FAIL，不能称深度、平台和穿插已修复。

## 最新流程和输入来源

1. RGB0–7、时间戳。使用先前context调优的单短语（ball / brown ball）；Grounding DINO各帧唯一框，RGB7框提示SAM2双向传播。它是文本指定目标，不是无先验自动运动物体发现。短语与本批组绑定，不能称盲测泛化。
2. VGGT预测深度、相机内外参，保留原始单位；真正读取固定scale=5.819486884015457。该值来自旧pilot尺度中位数，是待验证prior，非新视频米制尺度保证。
3. SAM2 mask映射到VGGT分辨率；mask内可见3D表面联合拟合8个球心和未知共享半径，soft-L1；随后8帧鲁棒直线拟合p7/v7。不固定0.11m、不约束同高度、不用GT对齐。
4. RGB0静态深度按像素邻接生成有限triangle mesh；去掉八帧mask并集及扩张边缘、深度跳变；不补不可见支撑和厚度。各family使用相同机制。几何已生成不等于碰撞几何准确。
5. 下方35%平面prior估计重力方向，跨RGB0/7一致性门；9.81m/s²固定。该prior假定大致正立相机及下方水平面，36例通过观察准入不等于GT方向全合格。
6. 统一sphere state + finite mesh + fixed physics输入PyBullet，zero omega；先刷新碰撞检查，>1mm初始重叠直接FAIL，不对齐、不settle。固定质量1kg、摩擦0.35、恢复0.25、零阻尼；保持原1/240 fixedTimeStep、numSubSteps=8、每帧8调用，41输出/328调用。
7. 估计/重力/rollout先hash冻结，再读GT状态和相机评测；旧D仅作为GT-omega历史展示，不能作为公平因果对照。

## 执行方式

本轮复用已冻结、RGB hash匹配的Grounding DINO/SAM2和VGGT原始输出，CPU两线程重新执行状态/几何/重力/Bullet及评测。未重新运行模型、未占GPU、未修改solver，未训练。完整288帧源身份验证。3个原有数值测试PASS（环境没有pytest，执行测试文件自带入口）。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/run_grounded_generic_pilot36.py --source /data/gaoya/agent-data/outputs/context_generic_pilot36_20260922_v1 --masks /data/gaoya/agent-data/outputs/pilot36_grounding_sam2_20260923_v1 --output {root}
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/publish_grounded_generic.py --root {root}
```

输出路径为首次执行路径；重跑必须换新目录。

## 结果（失败样本未移除）

```json
{json.dumps({k:v for k,v in summary.items() if k!='per_group'}, ensure_ascii=False, indent=2)}
```

36例都通过二维mask形状检查；20例通过局部球面数值准入，仍不满足真实p/v精度。初始重叠0/20不代表平台存在：20例初始接触均为0。动态mask并集会在mesh上留下未知区；其与深度/状态误差分别造成多少缺失支撑，尚未做独立消融，不把全部轨迹错误归于一个因素。

新D与同输入CV按相同20例比较，不把16例未运行算零。所有36例球拟合诊断值都保留，失败拟合值不是可部署估计。GT只在冻结后用于度量。contact accuracy/precision/recall、GT几何穿透、primitive GT边界误差本轮NOT_EVALUATED；30Hz实际接触记录在rollouts中，不能替代GT事件匹配。没有证据说明fixed scale跨视频成立。

## 逐例

| case | state | rollout | p7 error m | v7 error m/s | radius m | ADE m | CV ADE m | failure |
|---|---|---|---:|---:|---:|---:|---:|---|
'''
    for r in records:
        m=r['state_metrics']; t=r['trajectory_metrics']
        ade=f"{t['ADE_m']:.4f}" if t else 'NOT_RUN'; cv=f"{t['CV_ADE_m']:.4f}" if t else 'NOT_RUN'
        report += f"| {r['case_id']} | {r['state_status']} | {r['rollout_status']} | {m['p7_error_m']:.4f} | {m['v7_error_mps']:.4f} | {m['radius_m']:.4f} | {ade} | {cv} | {r['failure']} |\n"
    (root/'report.md').write_text(report)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['per_group','family_results']}, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    publish(p.parse_args().root)
