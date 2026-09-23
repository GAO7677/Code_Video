"""Keep the original v3 interaction/layout while presenting new frozen results."""
from pathlib import Path
import re
import shutil
import argparse
import json

ROOT = Path('/data/gaoya/agent-data/outputs/context_grounded_generic_pilot36_20260923_v1')
REPO = Path(__file__).resolve().parents[1]

def main(root):
    ROOT = root
    html = (ROOT/'legacy_v3_index.html').read_text()
    html = html.replace('href="styles.css?v=contact-gate-1"', 'href="../overlay_viewer_v3/styles.css?v=contact-gate-1"')
    html = html.replace('src="app.js?v=contact-gate-1"', 'src="v3_layout.js?v=20260923-1"')
    summary = json.loads((ROOT/'viewer_data.json').read_text())['summary']
    banner = f"新结果：{summary['executed']}例41帧已完成；{summary['cases']-summary['executed']}例未推进。几何模式：{summary.get('geometry_mode','legacy')}。"
    html = re.sub(r'<body><div.*?</div>', '<body><div style="padding:12px;background:#fff2c4">'+banner+'</div>', html, count=1)
    html = re.sub(r'<nav class="artifact-links".*?</nav>', '<nav class="artifact-links"><a href="report.md">本轮报告</a><a href="index.html">四面板诊断</a><a href="viewer_data.json">逐例指标</a></nav>', html, flags=re.S)
    html = html.replace('SAM2 + sphere','DINO + SAM2')
    labels = {'A · Oracle':'GT · eval only','B · State error':'CV · 同输入恒速','C · Geometry / aligned':'旧 D · GT omega','D · End-to-end':'新 D · zero omega'}
    for a,b in labels.items(): html=html.replace(a,b)
    table='''<section class="layer-reference"><h2>图层说明 · 常驻参考</h2><table><thead><tr><th>图层</th><th>来源</th><th>用途</th></tr></thead><tbody>
<tr><td>绿色 GT</td><td>真值未来轨迹，冻结后读取</td><td>评测，不是A仿真</td></tr>
<tr><td>蓝色 CV</td><td>与新D相同p7/v7的恒速外推</td><td>同输入基线</td></tr>
<tr><td>橙色 旧D</td><td>旧v3结果，GT omega</td><td>历史参照，不是本轮C消融</td></tr>
<tr><td>红色 新D</td><td>新估计状态＋观测mesh＋zero omega</td><td>本次41帧结果；FAIL无轨迹</td></tr></tbody></table>
<p>沿用原页面布局，图层名称按本次实际实验更新；未运行的A/B/C消融不伪造。未来轨迹叠加在RGB7背景；中心标记非球体实际像素半径。无初始穿插不代表恢复了有效支撑。</p></section>'''
    html = re.sub(r'<section class="layer-reference".*?</section>', table, html, flags=re.S)
    if summary.get('geometry_mode')=='local_plane_completion':
        html=html.replace('<h2>Overlay layers</h2>','<p>黄色三角形：局部平面先验推断的表面，不是观测真值。</p><h2>Overlay layers</h2>')
        html=html.replace('href="report.md">本轮报告','href="completion_report.md">补全对照报告')
    html = re.sub(r'<label><input type="checkbox" data-layer="(?:estimatedGeometry|gtGeometry)".*?</label>', '', html)
    html = html.replace('<h2>Overlay layers</h2>', '<h2>Overlay layers</h2><label><input type="checkbox" data-layer="mesh" checked><i class="swatch swatch-est"></i>Estimated mesh boundary</label>')
    (ROOT/'v3_layout.html').write_text(html)
    shutil.copyfile(REPO/'web/grounded_v3_layout.js',ROOT/'v3_layout.js')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    main(p.parse_args().root)
