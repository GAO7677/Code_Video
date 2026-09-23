"""Post-freeze paired comparison; no selection of inference parameters."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from context_rgb_pybullet_common import dump_json, sha256_file


def report(source, out):
    old = json.loads((source/'viewer_data.json').read_text())
    new = json.loads((out/'viewer_data.json').read_text())
    originals = {r['id']: r for r in old['records']}
    rows = []
    reasons = Counter()
    for r in new['records']:
        cid = r['id']
        b = originals[cid]
        bm, nm = b['evaluation'].get('trajectory_metrics'), r['evaluation'].get('trajectory_metrics')
        before = json.loads((source/'estimates'/cid/'collision_primitive.json').read_text())
        after = json.loads((out/'estimates'/cid/'collision_primitive.json').read_text())
        old_est = json.loads((source/'estimates'/cid/'result.json').read_text())
        new_est = json.loads((out/'estimates'/cid/'result.json').read_text())
        assert old_est['state'] == new_est['state'], 'state changed'
        assert sha256_file(source/'estimates'/cid/'vggt_raw.npz') == sha256_file(out/'estimates'/cid/'vggt_raw.npz')
        if not new_est['support_reason']:
            reasons.update(c.get('reason') for c in after['completion_audit']['components'])
        row = {'id': cid, 'case': r['case_id'], 'scope': new_est['support_reason'] or 'sphere_candidate',
               'old_status': b['rollout_status'], 'new_status': r['rollout_status'],
               'failure': r['failure'], 'mesh_changed': before != after,
               'vertices_changed': before['vertices'] != after['vertices'],
               'faces_changed': before['faces'] != after['faces'],
               'old_inferred_pixels': before.get('completion_audit', {}).get('inferred_pixels', 0),
               'new_inferred_pixels': after.get('completion_audit', {}).get('inferred_pixels', 0),
               'old_ADE_m': bm['ADE_m'] if bm else None, 'new_ADE_m': nm['ADE_m'] if nm else None,
               'old_FDE_m': bm['FDE_m'] if bm else None, 'new_FDE_m': nm['FDE_m'] if nm else None,
               'old_overlap_m': b['initial_penetration_m'], 'new_overlap_m': r['initial_penetration_m'],
               'old_contacts': b.get('contacts'), 'new_contacts': r.get('contacts')}
        rows.append(row)
    pairs = [r for r in rows if r['old_ADE_m'] is not None and r['new_ADE_m'] is not None]
    summary = {'old': old['summary'], 'new': new['summary'], 'common_rollouts': len(pairs),
               'paired_ADE_old_new_m': [float(np.mean([r[k] for r in pairs])) for k in ['old_ADE_m', 'new_ADE_m']],
               'paired_FDE_old_new_m': [float(np.mean([r[k] for r in pairs])) for k in ['old_FDE_m', 'new_FDE_m']],
               'improved_over_1cm': sum(r['new_ADE_m'] < r['old_ADE_m']-.01 for r in pairs),
               'worsened_over_1cm': sum(r['new_ADE_m'] > r['old_ADE_m']+.01 for r in pairs),
               'lost_rollouts': [r['case'] for r in rows if r['old_status']=='EXECUTED' and r['new_status']!='EXECUTED'],
               'new_rollouts': [r['case'] for r in rows if r['old_status']!='EXECUTED' and r['new_status']=='EXECUTED'],
               'completion_reasons': dict(reasons), 'all_state_and_vggt_unchanged': True,
               'geometry_face_changes': sum(r['faces_changed'] for r in rows),
               'contact_accuracy': 'NOT_EVALUATED; contact logs are not GT event accuracy'}
    dump_json(out/'mesh_comparison.json', {'summary': summary, 'cases': rows})
    text = '# 有限平面分离与遮挡补全测试 · 2026-09-23\n\n'
    text += '独立几何实验，baseline为recovery v2。固定球状态、相机、尺度、重力和solver。30个预声明单球候选全部尝试（包括2个状态失败）；40个不支持样本保留，不强行rollout。GT只在输出冻结后评测，不按GT逐例挑参数。\n\n'
    text += '本轮实现局部RANSAC共面点分离、连通性/8方向包围/边界证据检查，再补全目标遮挡未知像素。没有实施全场景物体级分割、修正VGGT尺度或移动已有观测点；不能称为完整场景恢复已解决。\n\n'
    text += '```json\n'+json.dumps(summary, ensure_ascii=False, indent=2)+'\n```\n\n'
    text += '第一版误将参考目标区域中、后续帧已揭露的静态点再次排除，30例补全均拒绝。该实现问题已修复并加入测试；第一版产物保留。未放宽残差、包围和边界门限。\n\n'
    text += '穿插通过不等于支撑正确；contact accuracy以及关键桌沿GT几何误差尚未评测，不能凭ADE断言物理场景已准确恢复。合成可见缺口不被填补，不证明隐藏的真实缺口总能识别。\n\n'
    text += '| case | old → new | 补全像素 old → new | ADE old → new m | FDE old → new m | failure |\n|---|---|---|---|---|---|\n'
    for r in rows:
        text += f"| {r['case']} | {r['old_status']} → {r['new_status']} | {r['old_inferred_pixels']} → {r['new_inferred_pixels']} | {r['old_ADE_m']} → {r['new_ADE_m']} | {r['old_FDE_m']} → {r['new_FDE_m']} | {r['failure']} |\n"
    text += '\n复现命令（使用新的output目录）：\n```bash\n'
    text += f"CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/run_segmented_mesh_ablation.py --source {source} --output {out}\n```\n"
    (out/'mesh_comparison.md').write_text(text)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    report(a.source, a.output)
