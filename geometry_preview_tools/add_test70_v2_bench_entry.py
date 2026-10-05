from pathlib import Path
import json,shutil
P=Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/generic-full-2175-lora1500-lineage/test89-unified').resolve()
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
B=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261005_test70_v2_bench_entry');B.mkdir(exist_ok=True)
for n in ['bench.js','bench.html']:
 if not (B/n).exists():shutil.copy2(P/n,B/n)
groups=json.loads((D/'group_assignments.json').read_text());cases={}
for f,g in groups.items():
 for c in g['cases']:
  p=D/'samples'/c;m=json.loads((p/'metadata.json').read_text());record=json.loads((p/'complete.json').read_text());control=m['control'];value=format(control['value'],'.10g')+('°' if control['units']=='deg' else ' '+control['units'])
  cases[c]=dict(family=f,gt_url='/test70_v2/samples/'+c+'/videos/rgb_cycles.mp4?v='+record['video_sha256'],poster_url='/test70_v2/samples/'+c+'/frames/00000.png?v='+record['video_sha256'],variant=value,prompt=json.loads((p/'captions/captions.json').read_text())['input_caption'],control_label=control['variable']+' = '+value,scene=g['scene']['label'])
data=dict(dataset='test70_v2',family_labels={f:g['scene']['task_label']+' · '+g['scene']['label'] for f,g in groups.items()},cases=cases)
(P/'test70_v2_bench.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
p=P/'bench.js';s=p.read_text();s=s.replace("const bench=make('section');", "const titleName=key==='test70-v2'?'Test70 v2 · 新版仿真':key==='test89'?'Test89':'Test70';\n const bench=make('section');")
s=s.replace("key==='test89'?'Test89':'Test70'",'titleName') if False else s
s=s.replace("bench.append(make('h2',key==='test89'?'Test89':'Test70'));", "bench.append(make('h2',titleName));")
s=s.replace("key==='test89'?'Test89 使用", "key==='test70-v2'?'Test70 v2：14组×5个变体，Test70几何设计 / Test89物理设置重新仿真；90帧、30fps、896×512。包含新版坡度住宅背景、缝隙暖光和多米诺26°近景。此处为仿真参考视频，未接入模型评测排名。':key==='test89'?'Test89 使用")
s=s.replace("'两个 benchmark 的样本和时间协议不同，指标分别统计，不跨数据集直接排名。'", "'各 benchmark / 数据版本的样本、渲染及时间协议可能不同，指标分别统计，不直接混合排名。'")
s=s.replace("link('查看该 Bench 的模型结果 →','./?bench='+key)", "link(key==='test70-v2'?'完整新版数据集 →':'查看该 Bench 的模型结果 →',key==='test70-v2'?'/test70_v2/dataset.html':'./?bench='+key)")
s=s.replace("bench.append(intro);", "if(key==='test70-v2'){intro.append(document.createTextNode(' · '),link('Bench 说明 PDF','/test70_v2/docs/test70_v2_bench.pdf'),document.createTextNode(' · '),link('Markdown＋图片','/test70_v2/docs/test70_v2_bench_md_assets.zip'));}bench.append(intro);")
s=s.replace("${key==='test89'?'Test89':'Test70'}", "${titleName}")
s=s.replace("const title=(labels[family]?labels[family]+' · ':'')+family", "const title=(data.family_labels?.[family]||labels[family]||'')+' · '+family")
s=s.replace("video.setAttribute('aria-label'", "if(info.poster_url)video.poster=info.poster_url;video.setAttribute('aria-label'")
s=s.replace("caption.append(make('small',info.prompt||'暂无提示词'));", "if(info.control_label)caption.append(make('small',info.control_label));caption.append(make('small',info.prompt||'暂无提示词'));")
s=s.replace("loadBench('test70','test70_dashboard.ui.json')", "loadBench('test70','test70_dashboard.ui.json'),loadBench('test70-v2','test70_v2_bench.json')")
s=s.replace('已加载 Test89 / Test70，共 ','已加载 Test89 / Test70 / Test70 v2，共 ')
p.write_text(s)
p=P/'bench.html';s=p.read_text().replace('Bench 介绍 · Test89 / Test70','Bench 介绍 · Test89 / Test70 / Test70 v2').replace('bench.js?v=20261002-1','bench.js?v=20261005-v2');p.write_text(s)
print('Updated',P,'new cases',len(cases))
