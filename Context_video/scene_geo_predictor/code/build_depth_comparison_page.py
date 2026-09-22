"""Create a local synchronized depth diagnostic viewer and reproducibility report."""
from compare_depth_diagnostics import OUT,ROOT,DA,VDA,WEIGHTS,read,write,cases
import hashlib,subprocess

def main():
    provenance={}
    for name,path in WEIGHTS.items():
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
        provenance[name]={'weight':str(path),'sha256':h.hexdigest(),'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=DA if name=='da' else VDA,text=True).strip()}
    write(OUT/'provenance.json',provenance)
    protocol=read(OUT/'protocol.json')
    protocol['approved_runtime']={'attention':'official ordinary PyTorch for both models; no model or resolution fallback','render_max_rgb_error_255':1,'render_mean_rgb_error_255':.001,'user_approved':True}
    write(OUT/'protocol.json',protocol)
    data=read(OUT/'comparison.json')
    lines=['# Context depth diagnostic · 12 cases','',
      '12 cases, one history group each across aperture/deflector/support_edge; 8 observed frames each. DA V1 Large and Video Depth Anything Large ran on physical GPU6. No future RGB, DiT changes, predictor training, or automatic model substitution.',
      '', '## Interpretation', '',
      'DA/VDA are relative inverse-depth estimators. All three models are fitted to GT inverse depth using RGB0 checkerboard calibration pixels, then evaluated on opposite checkerboard pixels in RGB1–7. This is an oracle alignment diagnostic, not a deployable metric-depth result. RGB frames are correlated and checkerboard splits do not imply independent samples. Aggregates are unweighted case means. Predicted metric depth is clipped to [0.2,15] m; invalid inverse-depth rate is separately recorded.',
      '', 'GT is rerendered from the original context poses. Maximum RGB difference must be ≤1/255 and mean ≤0.001/255, approved before continuing. GT never enters model forward. VDA repeats RGB7 to fill its official 32-frame input window. The comparison therefore tests short-context inference, not long-video temporal quality.',
      '', '| Model | All AbsRel ↓ | Object AbsRel ↓ | Static object RMSE m ↓ |', '|---|---:|---:|---:|']
    for name,s in data['summary'].items():lines.append(f"| {name} | {s['all']['AbsRel']:.5f} | {s['objects']['AbsRel']:.5f} | {s['static_objects']['RMSE_m']:.5f} |")
    lines+=['','## Findings','',
      'VDA has the lowest whole-image AbsRel; DA V1 has the lowest object-region macro mean. This aggregate improvement is driven mainly by deflector cases. VGGT is better on the aperture and support_edge family object means. No model is uniformly best.',
      'On support_edge_g03_v0620, object AbsRel is VGGT 0.04015, DA 0.05116, VDA 0.04434. VDA has slightly lower object RMSE (0.41885 m vs VGGT 0.43437 m), but neither alternative resolves the platform geometry problem. Whole-image VGGT error falls from 0.13097 with the original sphere scale to 0.00684 after GT affine alignment: calibration remains a major issue. These protocols are different, so this is a diagnostic comparison only.',
      '', '| Family | VGGT object AbsRel | DA object AbsRel | VDA object AbsRel |','|---|---:|---:|---:|']
    for family in ['aperture','deflector','support_edge']:
        vals=[]
        for model in ['vggt','da','vda']:
            values=[r['metrics']['objects']['AbsRel'] for r in data['rows'] if r['family']==family and r['model']==model]
            vals.append(sum(values)/len(values))
        lines.append('| '+family+' | '+' | '.join(f'{v:.5f}' for v in vals)+' |')
    lines+=['','## Per-case results','','| Case | Model | Object AbsRel | Object RMSE m |','|---|---|---:|---:|']
    for row in data['rows']:
        m=row['metrics']['objects'];lines.append(f"| {row['case_id']} | {row['model']} | {m['AbsRel']:.5f} | {m['RMSE_m']:.5f} |")
    lines+=['','## Limits','','No DA/VDA primitive fitting or Bullet rollouts are performed here. A lower aligned depth error alone does not prove improved collision outcomes or solve the known sphere-calibration error. Inspect object-region results rather than background-heavy global averages. Timing in execution JSON includes per-case inference and NPZ serialization, excludes model load, and includes first-case warmup; not end-to-end visual-to-physics latency.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    (OUT/'index.html').write_text(PAGE)

PAGE='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Depth diagnostics · 12 cases</title>
<style>body{margin:0;background:#f3f5f2;color:#1e2927;font:14px/1.6 system-ui,sans-serif}header{background:#20312e;color:white;padding:18px 24px}main{max-width:1600px;margin:auto;padding:20px}h1{margin:0;font-size:23px}a{color:#16878c}header a{color:#9ee1db}.notice{padding:12px;border-left:4px solid #ce9929;background:#fff3d9}select,button,input{padding:8px;font:inherit}nav{display:flex;gap:12px;flex-wrap:wrap;align-items:center;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.panel{background:white;border:1px solid #ccd4cd;padding:10px}.panel img{width:100%;display:block}.panel h2{font-size:15px;margin:0 0 8px}.stats{font-family:monospace}table{border-collapse:collapse;width:100%;background:white;margin:16px 0}th,td{border:1px solid #ccd4cd;padding:8px;text-align:left}tr.active{background:#d8efea}.scroll{overflow:auto}#timeline{flex:1;min-width:100px}.legend{height:12px;background:linear-gradient(90deg,#30123b,#4662d7,#1ac7c2,#a4fc3c,#f9ba38,#e64617,#7a0403)}@media(max-width:900px){.grid{grid-template-columns:repeat(2,1fr)}}@media(max-width:580px){.grid{grid-template-columns:1fr}main{padding:10px}}</style>
<header><h1>Context 深度诊断 · VGGT / Depth Anything / Video Depth Anything</h1><a href="report.md">完整报告</a> · <a href="comparison.json">逐 case 数值</a> · <a href="protocol.json">评测协议</a> · <a href="../overlay_viewer_v2/">轨迹页面</a></header>
<main><p class="notice">所有模型使用 RGB0–7。GT 只用于评测及尺度/偏移对齐：在 RGB0 标定，RGB1–7 评测。这里的米制图是 <strong>GT 对齐诊断</strong>，不能当作无需 GT 的深度预测，更不能直接推断 Bullet 轨迹更好。</p>
<p class="notice">结论：VDA 全图均值较好，DA V1 物体均值较好；收益主要来自挡板场景。门框和平台组的物体 AbsRel 均值仍是 VGGT 更低，不能据此全面替换 VGGT。当前问题 case 中，DA/VDA 未一致优于 VGGT。</p><div id="summary" class="scroll"></div><nav><label>Case <select id="case"></select></label><button id="play">播放</button><input id="timeline" type="range" min="0" max="7" value="7"><strong id="frame">RGB7</strong><label><input id="overlay" type="checkbox" checked>叠加 RGB</label></nav>
<p id="description"></p><div class="grid" id="grid"></div><p>深度图统一色标：远（蓝）→ 近（红），按此 case 的 GT 深度 2%–98% 范围显示，所有模型/帧共用。误差图：黑色小、黄色大，0–50% 相对误差；GT 无效像素置黑。</p><div class="legend"></div>
<h2>RGB7 相对误差图</h2><div class="grid" id="errors"></div><h2>每个 case 的物体区域误差（AbsRel，越低越好）</h2><div id="cases" class="scroll"></div></main>
<script>
const names={vggt:'VGGT',da:'Depth Anything V1 Large',vda:'Video Depth Anything Large'},$=s=>document.getElementById(s);let data,frame=7,timer;
const n=x=>x==null?'—':Number(x).toFixed(4);const panel=(name,src,detail='')=>`<section class="panel"><h2>${name}</h2><img src="${src}"><div class="stats">${detail}</div></section>`;
function draw(){let cid=$('case').value,base='comparison_assets/'+cid+'/';$('frame').textContent='RGB'+frame;$('timeline').value=frame;let rows=data.rows.filter(r=>r.case_id===cid);$('grid').innerHTML=panel('输入 RGB',base+'rgb_'+frame+'.webp')+panel('GT camera-Z 深度',base+'gt_'+frame+'.webp')+rows.map(r=>panel(names[r.model]+' · GT aligned',base+r.model+($('overlay').checked?'_':'_depth_')+frame+'.webp',`物体 AbsRel ${n(r.metrics.objects.AbsRel)} · RMSE ${n(r.metrics.objects.RMSE_m)} m`)).join('');$('errors').innerHTML=rows.map(r=>panel(names[r.model],base+r.model+'_error7.webp')).join('');$('description').textContent=cid+' · 指标固定评测 RGB1–7；RGB0 为对齐帧，不计入评测。';document.querySelectorAll('[data-case]').forEach(r=>r.classList.toggle('active',r.dataset.case===cid));}
(async()=>{data=await(await fetch('comparison.json')).json();let ids=[...new Set(data.rows.map(r=>r.case_id))];$('case').innerHTML=ids.map(id=>`<option>${id}</option>`).join('');$('case').value='phase1_support_edge_g03_v0620';$('summary').innerHTML='<table><tr><th>12 case 等权均值</th><th>全图 AbsRel ↓</th><th>物体 AbsRel ↓</th><th>静态物体 RMSE m ↓</th></tr>'+Object.entries(data.summary).map(([k,s])=>`<tr><th>${names[k]}</th><td>${n(s.all.AbsRel)}</td><td>${n(s.objects.AbsRel)}</td><td>${n(s.static_objects.RMSE_m)}</td></tr>`).join('')+'</table>';$('cases').innerHTML='<table><tr><th>Case</th><th>VGGT</th><th>DA</th><th>VDA</th></tr>'+ids.map(id=>`<tr data-case="${id}"><td>${id}</td>`+['vggt','da','vda'].map(m=>`<td>${n(data.rows.find(r=>r.case_id===id&&r.model===m).metrics.objects.AbsRel)}</td>`).join('')+'</tr>').join('')+'</table>';draw()})();$('case').onchange=draw;$('overlay').onchange=draw;$('timeline').oninput=e=>{frame=+e.target.value;draw()};$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='播放'}else{timer=setInterval(()=>{frame=(frame+1)%8;draw()},250);$('play').textContent='暂停'}};$('cases').onclick=e=>{let r=e.target.closest('[data-case]');if(r){$('case').value=r.dataset.case;draw()}};
</script></html>'''

if __name__=='__main__':main()


