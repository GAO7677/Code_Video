"""Post-freeze test70 evaluation and original-v3-style six-stage viewer."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np

from context_rgb_pybullet_common import dump_json, crop_transform, resize_crop_mask
from run_grounded_generic_pilot36 import verify

REPO = Path(__file__).resolve().parents[1]
TRUTH = Path('/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v1/cases')


def evaluate(entry, state, rollout, e, k, scale, shape):
    """Single actor eval; renderer camera checked independently against stored pixels."""
    sample = Path(entry['source_sample'])
    truth = TRUTH / sample.name
    meta = json.loads((truth / 'truth_metadata.json').read_text())
    actors = json.loads((sample / 'metadata.json').read_text())['actors']
    with np.load(truth / 'dynamic_masks.npz', allow_pickle=False) as a:
        names = list(a['object_names'].astype(str))
    if len(names) != 1:
        return {'status': 'BLOCKED', 'reason': 'multiple_GT_actors_no_unique_evaluation_mapping'}
    actor = names[0]
    radius = actors[actor]['size_m']['radius']
    with np.load(sample / 'raw/states_xyzw.npz', allow_pickle=False) as a:
        ns = list(a['object_names'].astype(str))
        assert ns.count(actor) == 1
        idx = ns.index(actor)
        gt = a['positions'][:49, idx]
        vel = a['linear_velocities'][7, idx]
    if len(gt) != 49:
        raise ValueError('49 GT states required')
    eye, target = np.array(meta['camera']['location']), np.array(meta['camera']['target'])
    f = target - eye
    f /= np.linalg.norm(f)
    right = np.cross(f, [0, 0, 1.])
    right /= np.linalg.norm(right)
    R = np.stack([right, -np.cross(right, f), f])
    gtcam = (gt - eye) @ R.T
    width, height = meta['resolution']
    focal = height / (2 * np.tan(np.radians(meta['camera']['effective_yfov_deg']) / 2))
    K = np.array([[focal, 0, width/2], [0, focal, height/2], [0, 0, 1.]])
    def projection(x):
        pix = x @ K.T
        return pix[:, :2] / pix[:, 2, None]
    with np.load(truth / 'trajectory_pixels.npz') as a:
        ref = a['centers_tnc'][:8, 0, :2]
    if np.max(abs(ref)) <= 1.5:
        ref = ref * [width, height]
    error = float(np.linalg.norm(projection(gtcam[:8]) - ref, axis=1).max())
    if error > 2:
        return {'status': 'BLOCKED', 'reason': 'GT_camera_projection_check_failed', 'max_pixel_error': error}
    out = {'status': 'EVALUATED', 'projection_check_px': error, 'gt_radius_m': radius,
           'gt_future_uv': (projection(gtcam[8:49]) * [640/width, 360/height]).tolist(),
           'contact_accuracy': 'NOT_EVALUATED', 'GT_geometry_error': 'NOT_EVALUATED'}
    if 'radius' not in state:
        return out
    p7 = e[0, :, :3] @ state['p7'] + e[0, :, 3] * scale
    v7 = e[0, :, :3] @ state['v7']
    truev = R @ vel
    centers = np.array(state['center_sequence']) @ e[0, :, :3].T + e[0, :, 3] * scale
    out['state_metrics'] = {'admitted': state['status'] == 'ESTIMATED',
        'p7_error_m': float(np.linalg.norm(p7 - gtcam[7])),
        'center_sequence_mean_error_m': float(np.linalg.norm(centers - gtcam[:8], axis=1).mean()),
        'v7_vector_error_mps': float(np.linalg.norm(v7 - truev)),
        'radius_error_m': abs(state['radius'] - radius), 'estimated_radius_m': state['radius']}
    denom = np.linalg.norm(v7) * np.linalg.norm(truev)
    out['state_metrics']['v7_direction_error_deg'] = float(np.degrees(np.arccos(np.clip(v7 @ truev / denom, -1, 1)))) if denom > 1e-9 else None
    if rollout['status'] == 'EXECUTED':
        pred = np.array(rollout['positions']) @ e[0, :, :3].T + e[0, :, 3] * scale
        cv = p7 + np.arange(1, 42)[:, None] / 30 * v7
        err = np.linalg.norm(pred - gtcam[8:49], axis=1)
        cverr = np.linalg.norm(cv - gtcam[8:49], axis=1)
        disp = np.linalg.norm((pred-p7) - (gtcam[8:49]-gtcam[7]), axis=1)
        px = np.linalg.norm(projection(pred)-projection(gtcam[8:49]), axis=1)
        out['trajectory_metrics'] = {'ADE_m': float(err.mean()), 'FDE_m': float(err[-1]),
            'CV_ADE_m': float(cverr.mean()), 'CV_FDE_m': float(cverr[-1]),
            'displacement_ADE_m': float(disp.mean()), 'displacement_FDE_m': float(disp[-1]),
            'projection_mean_px': float(px.mean()), 'projection_FDE_px': float(px[-1]), 'frame_errors_m': err.tolist()}
        # Evaluation renderer projection explicitly distinct from inference projection.
        out['pred_eval_uv'] = (projection(pred) * [640/width, 360/height]).tolist()
        out['cv_eval_uv'] = (projection(cv) * [640/width, 360/height]).tolist()
    return out


def publish(root):
    cv2.setNumThreads(2)
    for f in ['input_freeze.json', 'depth_freeze.json', 'estimate_freeze.json', 'rollout_freeze.json']:
        verify(root, f)
    verify(root / 'lower_plane_gravity_v1', 'freeze.json')
    protocol = json.loads((root / 'protocol.json').read_text())
    source = Path(protocol['source'])
    mapping = json.loads((source / 'evaluation_mapping.json').read_text())
    shutil.copyfile(source / 'evaluation_mapping.json', root / 'evaluation_mapping.json')
    gravity = {r['id']: r for r in json.loads((root / 'lower_plane_gravity_v1/candidates.json').read_text())}
    records = []
    for entry in mapping:
        cid = entry['id']
        folder = root / 'estimates' / cid
        est = json.loads((folder / 'result.json').read_text())
        roll = json.loads((root / 'rollouts' / f'{cid}.json').read_text())
        with np.load(folder / 'vggt_raw.npz') as a:
            d, k, e = a['depth'][..., 0], a['intrinsic'], a['extrinsic']
        with np.load(folder / 'tracking.npz') as a:
            masks = a['masks']
        h, w = masks.shape[1:]
        display_scale = np.array([640/w, 360/h])
        transform = crop_transform((h, w))
        affine_inv = np.linalg.inv(transform['affine'])
        scale = est['scale']
        def project(points, t=7):
            cam = np.array(points) @ e[t, :, :3].T + e[t, :, 3] * scale
            pix = cam @ k[t].T
            uv = pix[:, :2] / pix[:, 2, None]
            return (np.c_[uv, np.ones(len(uv))] @ affine_inv.T)[:, :2] * display_scale
        state = est['state']
        row = {'id': cid, 'case_id': Path(entry['source_sample']).name,
               'state': state, 'geometry': est['geometry'], 'gravity': gravity[cid],
               'support_reason': est['support_reason'], 'rollout_status': roll['status'], 'failure': roll.get('reason'),
               'target_detection': est['target_detection'], 'fallback': est['fallback'],
               'initial_penetration_m': roll.get('initial_penetration_m'),
               'initial_contacts': roll.get('initial_contact_count'),
               'mask_contours': [[(c[:, 0] * display_scale).tolist() for c in cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]] for m in masks]}
        b = est['target_detection']['candidates'][0]['box']
        row['prompt_box'] = (np.array(b) * np.tile(display_scale, 2)).tolist()
        if 'center_sequence' in state:
            row['center_uv'] = [project([c], t)[0].tolist() for t, c in enumerate(state['center_sequence'])]
        if roll['status'] == 'EXECUTED':
            row['new_future_uv'] = project(roll['positions']).tolist()
            cv = np.array(state['p7']) + np.arange(1, 42)[:, None] / 30 * np.array(state['v7'])
            row['cv_future_uv'] = project(cv).tolist()
            contacts = roll['api_contacts']
            active = [i for i, cs in enumerate(contacts) if cs]
            row['contacts'] = {'frame_contacts': sum(bool(c) for c in roll['contacts']),
                'first_contact_time_s': (active[0]+1)/240 if active else None,
                'max_penetration_vs_rollout_geometry_m': max([max(0., -c['distance']) for cs in contacts for c in cs], default=0.),
                'contact_log_hz': 240, 'hidden_internal_substeps_observed': False}
        dest = root / 'diagnostics' / cid
        dest.mkdir(parents=True, exist_ok=True)
        lo, hi = np.percentile(d[np.isfinite(d)], [2, 98])
        for t in range(8):
            color = cv2.applyColorMap(np.uint8(np.clip((d[t]-lo)/max(hi-lo, 1e-8), 0, 1)*255), cv2.COLORMAP_TURBO)
            m = resize_crop_mask(masks[t], transform)
            cv2.drawContours(color, cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 100, 210), 1)
            cv2.imwrite(str(dest / f'depth_{t}.png'), color)
        if (folder / 'collision_primitive.json').exists():
            mesh = json.loads((folder / 'collision_primitive.json').read_text())
            faces = np.array(mesh['faces'], dtype=int).reshape(-1, 3)
            vertices = np.array(mesh['vertices'])
            uv = project(vertices)
            edge = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
            unique, count = np.unique(edge, axis=0, return_counts=True)
            row['mesh_segments'] = uv[unique[count == 1]].tolist()
            inferred = faces[np.array(mesh.get('face_inferred', [False]*len(faces)), bool)]
            ie = np.unique(np.sort(np.concatenate([inferred[:, [0, 1]], inferred[:, [1, 2]], inferred[:, [2, 0]]]), axis=1), axis=0)
            row['completion_segments'] = uv[ie].tolist()
            row['completion_audit'] = mesh.get('completion_audit')
            rgb = cv2.resize(cv2.imread(str(root / 'inputs' / cid / 'rgb_07.png')), (640, 360))
            overlay = rgb.copy()
            valid = np.isfinite(uv).all(axis=1) & (np.abs(uv).max(axis=1) < 1e5)
            polys = [np.rint(uv[f]).astype(np.int32) for f in faces if valid[f].all()]
            cv2.fillPoly(overlay, polys, (210, 160, 230))
            for f in inferred:
                if valid[f].all():
                    cv2.fillPoly(overlay, [np.rint(uv[f]).astype(np.int32)], (40, 220, 255))
            cv2.imwrite(str(dest / 'mesh.png'), cv2.addWeighted(rgb, .5, overlay, .5, 0))
        else:
            cv2.imwrite(str(dest / 'mesh.png'), cv2.imread(str(root / 'inputs' / cid / 'rgb_07.png')))
        if est['support_reason']:
            row['evaluation'] = {'status': 'NOT_RUN', 'reason': est['support_reason']}
        else:
            try:
                row['evaluation'] = evaluate(entry, state, roll, e, k, scale, (h, w))
            except (FileNotFoundError, KeyError, ValueError, AssertionError) as exc:
                row['evaluation'] = {'status': 'BLOCKED', 'reason': f'{type(exc).__name__}: {exc}'}
        dump_json(root / 'cases' / f'{cid}.json', row)
        records.append(row)
        print('PUBLISH', cid, row['rollout_status'], row['evaluation']['status'], flush=True)
    valid = [r for r in records if 'trajectory_metrics' in r['evaluation']]
    summary = {'cases': len(records), 'state': dict(Counter(r['state']['status'] for r in records)),
        'gravity': dict(Counter(r['gravity']['gravity']['status'] for r in records)),
        'rollout': dict(Counter(r['rollout_status'] for r in records)),
        'evaluated_rollouts': len(valid),
        'ADE_m': float(np.mean([r['evaluation']['trajectory_metrics']['ADE_m'] for r in valid])) if valid else None,
        'FDE_m': float(np.mean([r['evaluation']['trajectory_metrics']['FDE_m'] for r in valid])) if valid else None,
        'CV_ADE_m_same_cases': float(np.mean([r['evaluation']['trajectory_metrics']['CV_ADE_m'] for r in valid])) if valid else None,
        'initial_overlap_tested': sum(r['initial_penetration_m'] is not None for r in records),
        'initial_overlap_positive': sum((r['initial_penetration_m'] or 0) > .001 for r in records)}
    dump_json(root / 'viewer_data.json', {'summary': summary, 'records': records})
    html = Path('/data/gaoya/agent-data/outputs/context_grounded_generic_pilot36_20260923_v1/legacy_v3_index.html').read_text()
    html = html.replace('styles.css?v=contact-gate-1', '../overlay_viewer_v3/styles.css?v=contact-gate-1').replace('app.js?v=contact-gate-1', 'test70_viewer.js')
    html = html.replace('36 CASE PILOT', 'TEST70 · CONTEXT PIPELINE').replace('SAM2 + sphere', 'DINO + SAM2')
    html = re.sub(r'<body><div.*?</div>', '<body><div style="padding:12px;background:#fff2c4">test70 全70例；未知半径单球链路。非球体/关节/多动态物体仅诊断，不套用球体rollout。<a href="../overlay_viewer_v3/">36例旧页面</a></div>', html, count=1)
    html = re.sub(r'<nav class="artifact-links".*?</nav>', '<nav class="artifact-links"><a href="report.md">报告</a><a href="viewer_data.json">逐例指标</a><a href="protocol.json">协议</a></nav>', html, flags=re.S)
    html = re.sub(r'<div class="family-tabs".*?</div>', '<div class="family-tabs"><button class="family-tab is-active" data-filter="all">全部</button><button class="family-tab" data-filter="EXECUTED">已推进</button><button class="family-tab" data-filter="FAIL">失败</button><button class="family-tab" data-filter="UNSUPPORTED">不支持</button></div>', html, flags=re.S)
    layers = [('gt', 'GT trajectory · eval only', '#42df91'), ('cv', '同输入 CV', '#37aef5'), ('D', 'D · zero omega', '#ff514c'), ('mesh', 'Estimated mesh', '#f453dc'), ('completion', 'Inferred local plane', '#ffdd44')]
    controls = '<section id="layerControls" class="layer-controls"><h2>Overlay layers</h2>' + ''.join(f'<label><input type="checkbox" data-layer="{key}" checked><i style="color:{color}">●</i> {label}</label>' for key, label, color in layers) + '<p>A/B/C、GT geometry：NOT_RUN，本轮未适配 test70 原世界构建器。</p></section>'
    html = re.sub(r'<section id="layerControls".*?</section>', controls, html, flags=re.S)
    reference = '<section class="layer-reference"><h2>图层说明 · 常驻参考</h2><p>绿：GT轨迹（仅评测）；蓝：与D相同p7/v7的恒速外推；红：估计半径/状态/几何＋zero omega的D。紫色：观测mesh边界；黄色：显式局部平面推断。补全面不是真值。</p><p>未来轨迹叠加在RGB7静止背景。默认估计相机投影；勾选GT评测相机投影后才叠加GT并比较像素误差。失败无轨迹。A/B/C和GT几何本轮NOT_RUN。UNSUPPORTED场景的mesh仅作视觉诊断，可能包含其他运动物体。</p><label><input id="evalCamera" type="checkbox"> GT评测相机投影（冻结后，仅评测）</label></section>'
    html = re.sub(r'<section class="layer-reference".*?</section>', reference, html, flags=re.S)
    (root / 'index.html').write_text(html)
    shutil.copyfile(REPO / 'web/test70_context_viewer.js', root / 'test70_viewer.js')
    report = '# test70 Context → PyBullet · 2026-09-23\n\n全70例保留；复用冻结Grounding DINO＋SAM2，GPU6新跑VGGT；其余CPU两线程。\n\n'
    report += '支持范围按此前context视觉筛查预先固定：30个单球非关节候选；20个非球体、5个多动态物体、15个关节/动态支撑场景仅诊断。序号仅用于范围声明，未用于提供几何参数。这不是自动场景理解。\n\n'
    report += '未知半径球面联合拟合 → 固定scale=5.819486884015457 → 多帧静态融合/局部平面补全 → 下方平面重力prior → zero-omega Bullet。没有GT补齐或位置对齐。固定scale来自旧pilot，不保证跨视频米制准确。\n\n'
    report += '估计/重力/rollout全部hash冻结后才读取GT。GT相机投影先核对保存的轨迹像素，误差>2px标BLOCKED。D默认用估计相机投影；GT相机叠加是单独评测开关。\n\n'
    report += 'A/B/C、GT几何对照、contact accuracy及GT几何穿透NOT_RUN/NOT_EVALUATED。未用GT未来筛选场景；40例unsupported不冒充失败拟合；各分母独立。\n\n```json\n' + json.dumps(summary, ensure_ascii=False, indent=2) + '\n```\n\n'
    report += '| case | state | gravity | rollout | failure |\n|---|---|---|---|---|\n'
    for r in records:
        report += f"| {r['case_id']} | {r['state']['status']} | {r['gravity']['gravity']['status']} | {r['rollout_status']} | {r['failure']} |\n"
    (root / 'report.md').write_text(report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    publish(p.parse_args().root)
