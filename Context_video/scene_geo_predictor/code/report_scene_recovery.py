"""All-attempt paired report, never selects per-case inference methods."""
import json
from pathlib import Path
import numpy as np
from context_rgb_pybullet_common import dump_json

def main():
    base=Path('/data/gaoya/agent-data/outputs')
    roots=[base/'test70_context_pipeline_20260923_v1']+[base/f'test70_scene_recovery_20260923_v{i}' for i in [1,2,3]]
    datasets=[json.loads((r/'viewer_data.json').read_text()) for r in roots]
    baseline={r['id']:r for r in datasets[0]['records']};results=[]
    for root,data in zip(roots,datasets):
        paired=[];rows=[]
        for row in data['records']:
            old=baseline[row['id']];m=row['evaluation'].get('trajectory_metrics');om=old['evaluation'].get('trajectory_metrics')
            pair={'id':row['id'],'case':row['case_id'],'old_status':old['rollout_status'],'new_status':row['rollout_status'],
                'old_ADE_m':om['ADE_m'] if om else None,'new_ADE_m':m['ADE_m'] if m else None,'failure':row['failure']}
            if m and om:paired.append([om['ADE_m'],m['ADE_m']])
            rows.append(pair)
        common_states=[(baseline[r['id']]['evaluation']['state_metrics'],r['evaluation']['state_metrics']) for r in data['records'] if baseline[r['id']]['evaluation'].get('state_metrics') and r['evaluation'].get('state_metrics')]
        states={key:{'n':len(common_states),'old_median':float(np.median([a[key] for a,b in common_states])),
                       'new_median':float(np.median([b[key] for a,b in common_states]))} for key in ['p7_error_m','radius_error_m','v7_vector_error_mps']}
        result={'root':str(root),'summary':data['summary'],'common_rollouts':len(paired),
            'paired_old_new_ADE_m':np.mean(paired,axis=0).tolist() if paired else None,
            'paired_improved_gt_1cm':sum(b<a-.01 for a,b in paired),'paired_worsened_gt_1cm':sum(b>a+.01 for a,b in paired),
            'lost_baseline_rollouts':[r['case'] for r in rows if r['old_status']=='EXECUTED' and r['new_status']!='EXECUTED'],
            'common_state_diagnostics_including_failed':states,'cases':rows}
        results.append(result)
    out=roots[-1]
    dump_json(out/'recovery_comparison.json',results)
    report='# 通用输入恢复修复 · 2026-09-23\n\n本轮不逐case挑选算法、不回填GT、不调solver、不改变固定尺度，不隐藏失败样本。以下均为已见test70开发集结果，不是未见场景泛化证明。\n\n'
    report+='| 版本 | state通过/30 | rollout/30 | 有效ADE | 与旧版共同样本 | 配对旧→新ADE |\n|---|---:|---:|---:|---:|---|\n'
    for i,r in enumerate(results):
        report+=f"| {['旧baseline','轮廓+深度诊断','加入前表面约束','固定相机检查+有限平面去噪'][i]} | {r['summary']['state'].get('ESTIMATED',0)} | {r['summary']['rollout'].get('EXECUTED',0)} | {r['summary']['ADE_m']:.4f} | {r['common_rollouts']} | {r['paired_old_new_ADE_m']} |\n"
    report+='\n剩余40例UNSUPPORTED全部保留。有效均值分母不相同，必须结合共同样本与丢失的旧成功样本检查；状态误差包括失败拟合诊断，不代表全部可部署。\n\n'
    report+='## 修复与边界\n\n- 轮廓切线+置信度筛选的深度联合估计未知球半径/中心，前表面约束排除背面拟合；运动优化状态/残差纳入准入。合成压平球面测试：旧半径0.542m，新约0.299m，真值0.3m。\n- 固定相机先检查静态特征p90位移≤1.5px，再共用VGGT clip中位K和参考E；不能纠正绝对焦距偏差，检查失败不回退。\n- 平面去噪仅作用于已有观测顶点，位移≤1cm，不增加三角形、不封缺口、不移动球；不是完整物体级场景布局恢复。\n- 第5步采用可见球mask做前景遮挡合成，仅展示修复，不改变碰撞几何。\n- 原始VGGT深度、绝对焦距和尺度偏差仍在；球和支撑面联合布局、物体级几何准入仍未完成。低误差目标不能仅以EXECUTED或平均值下降判定成功。\n\n'
    report+='## 最新配对状态指标\n\n```json\n'+json.dumps(results[-1]['common_state_diagnostics_including_failed'],indent=2)+'\n```\n\n'
    report+='## 最新逐例\n\n| case | 旧状态 | 新状态 | 旧ADE | 新ADE | 失败原因 |\n|---|---|---|---|---|---|\n'
    for r in results[-1]['cases']:report+=f"| {r['case']} | {r['old_status']} | {r['new_status']} | {r['old_ADE_m']} | {r['new_ADE_m']} | {r['failure']} |\n"
    (out/'recovery_report.md').write_text(report)
    for r in results:print(json.dumps({k:v for k,v in r.items() if k!='cases'},ensure_ascii=False))

if __name__=='__main__':main()
