"""Publish true A/B/C/D zero-omega results with all six requested layers."""
import argparse,json,re,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from context_rgb_pybullet_common import dump_json

def publish(root,template):
    data=json.loads((root/'viewer_data.json').read_text());records=data['records']
    html=template.read_text()
    html=html.replace('href="styles.css?v=contact-gate-1"','href="../overlay_viewer_v3/styles.css?v=contact-gate-1"')
    html=html.replace('src="app.js?v=contact-gate-1"','src="abcd_viewer.js?v=20260923-1"')
    html=re.sub(r'<body><div.*?</div>','<body><div style="padding:12px;background:#fff2c4">A/B/C/D · zero omega；B含估计半径。共享估计重力与固定物性。严格C、aligned C单列；执行完成不等于预测正确。</div>',html,count=1)
    html=re.sub(r'<nav class="artifact-links".*?</nav>','<nav class="artifact-links"><a href="report.md">对照报告</a><a href="viewer_data.json">全部指标</a><a href="protocol.json">协议</a></nav>',html,flags=re.S)
    html=html.replace('SAM2 + sphere','DINO + SAM2').replace('B · State error','B · State + radius error')
    table='''<section class="layer-reference"><h2>图层说明 · 常驻参考</h2><table><thead><tr><th>图层</th><th>输入</th><th>含义</th></tr></thead><tbody>
<tr><td>A · Oracle</td><td>GT p7/v7/半径＋GT几何＋zero omega</td><td>GT输入参照，非完整GT动力学重放</td></tr>
<tr><td>B · State + radius error</td><td>估计p7/v7/半径＋GT几何＋zero omega</td><td>包含尺寸与运动状态误差</td></tr>
<tr><td>C · Geometry / aligned</td><td>GT p7/v7/半径＋估计mesh＋zero omega</td><td>严格C与最多55mm向上接触对齐独立；右侧切换</td></tr>
<tr><td>D · End-to-end</td><td>估计p7/v7/半径＋估计mesh＋zero omega</td><td>与原部署D逐帧一致</td></tr>
<tr><td>Estimated geometry</td><td>紫色实际mesh边界，黄色局部推断三角形</td><td>不套用旧family碰撞盒，不把推断面当观测</td></tr>
<tr><td>GT geometry · eval only</td><td>绿色真实静态碰撞盒边界</td><td>冻结后读取；地面不绘制伪造有限边框</td></tr>
</tbody></table><p>所有组共享估计重力向量与固定物性。A/B/C均为冻结后的评测分支，不回流修改视觉估计。状态、尺度、几何共享视觉来源，属于coupled ablation。各分支有效样本数不同，不能直接把均值差当因果贡献。</p>
<p>C aligned不吸附悬空球、不处理侧向穿插、不超过55mm。拒绝时没有轨迹；不自动回退严格C，使用右侧选框显式切换。接触结果按法向区分支撑/障碍，不是family语义GT contact accuracy。</p></section>'''
    html=re.sub(r'<section class="layer-reference".*?</section>',table,html,flags=re.S)
    html=html.replace('<h2>Overlay layers</h2>','<h2>Overlay layers</h2><label>C模式 <select id="cMode"><option value="C_strict">严格C</option><option value="C_aligned">C aligned（显式对齐）</option></select></label>')
    html=html.replace('C · Geometry / aligned</label>','C · Geometry（模式见上）</label>')
    html=html.replace('data-layer="B">','data-layer="B" checked>')
    (root/'index.html').write_text(html)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'web/grounded_abcd_viewer.js',root/'abcd_viewer.js')
    arms=['A','B','C_strict','C_aligned','D']
    common=[r for r in records if all(r['abcd'][a]['status']=='EXECUTED' for a in ['A','B','C_strict','D'])]
    data['summary']['common_ABCD_strict_cases']=len(common)
    data['summary']['common_ABCD_strict_ADE_m']={a:float(np.mean([r['abcd'][a]['ADE_m'] for r in common])) if common else None for a in ['A','B','C_strict','D']}
    data['summary']['family_results']={}
    for family in sorted({r['family'] for r in records}):
        subset=[r for r in records if r['family']==family]
        data['summary']['family_results'][family]={a:{'executed':sum(r['abcd'][a]['status']=='EXECUTED' for r in subset),
            'ADE_m':float(np.mean([r['abcd'][a]['ADE_m'] for r in subset if r['abcd'][a]['status']=='EXECUTED'])) if any(r['abcd'][a]['status']=='EXECUTED' for r in subset) else None} for a in arms}
    dump_json(root/'viewer_data.json',data)
    text='''# 最新补全输入 A/B/C/D zero-omega 对照 · 2026-09-23

B使用估计半径+p7/v7；A/C使用GT半径+p7/v7。所有分支zero omega、同一冻结估计重力和固定物性；A不是旧GT omega的完整oracle重放。GT只在视觉/几何估计freeze验证后读取，坐标注册仅用GT相机外参到VGGT camera0的刚体变换，不拟合尺度或轨迹。
GT几何由原builder精确构建（包括plane.urdf），静态物性覆写为与部署D相同的固定配置，不从GT估计物性。C/D是最新局部补全mesh，不复用旧family primitives。

C strict只做穿插准入；C aligned使用A初始接触确认支撑，要求近水平接触法向、法向速度≤0.01m/s，最多向上55mm，不改变速度、半径或几何。缺支撑不吸附、侧向穿插不修。两种C独立保存，拒绝均0step。
D独立重跑后与已冻结部署结果最大差<1e-8m；执行20例全部满足。各组物理步、solver与部署协议一致。接触日志每API调用记录（240Hz），不声称观测内部子步。generic contact roles来自法向，不能当成真实物体语义标签的precision/recall。

## 执行结果

```json
'''+json.dumps(data['summary'],ensure_ascii=False,indent=2)+'''
```

各分支均值分母不同；给出共同有效样本汇总，但非严格独立因果分解。所有失败保留在下表。

| case | arm | status | ADE m | FDE m | 初始化 | reason |
|---|---|---|---:|---:|---|---|
'''
    for r in records:
        for a in arms:
            m=r['abcd'][a];fmt=lambda v:'NOT_RUN' if v is None else f'{v:.4f}'
            text+=f"| {r['case_id']} | {a} | {m['status']} | {fmt(m['ADE_m'])} | {fmt(m['FDE_m'])} | {m['initialization']['status']} | {m['reason']} |\n"
    text+='''
## 复现命令

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 CUDA_VISIBLE_DEVICES='' /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/run_grounded_abcd.py --source /data/gaoya/agent-data/outputs/context_local_plane_completion_20260923_v1 --output '''+str(root)+'''
```

重新执行必须改用新目录，不能覆盖旧产物。代码：run_grounded_abcd.py、mesh_contact_alignment.py、generic_bullet_input.py；展示：publish_grounded_abcd.py、grounded_abcd_viewer.js。模型推理未重复执行，不占GPU。
'''
    (root/'report.md').write_text(text)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--template',type=Path,required=True);a=p.parse_args();publish(a.root,a.template)
