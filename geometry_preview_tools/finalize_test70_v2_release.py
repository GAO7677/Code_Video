from pathlib import Path
import json,hashlib,collections,time,re,html
from PIL import Image
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
cases=json.loads((R/'case_order.json').read_text());groups=json.loads((R/'group_assignments.json').read_text());cfg=json.loads((R/'config.json').read_text());rows=[];total_bytes=0
for c in cases:
 p=R/'samples'/c;d=json.loads((p/'complete.json').read_text());assert d['qa_warning_count']==0,(c,d)
 frames=sorted((p/'frames').glob('[0-9][0-9][0-9][0-9][0-9].png'));assert len(frames)==90,c
 for i,f in enumerate(frames):
  assert f.name==f'{i:05d}.png'
  with Image.open(f) as im:assert im.size==(896,512);im.verify()
 for f in ['videos/rgb_cycles.mp4','context/context8_cycles.mp4','context/context16_cycles.mp4','raw/trajectories.npz','metadata.json','camera.json','contacts.json','captions/captions.json']:
  assert (p/f).is_file() and (p/f).stat().st_size>0,(c,f)
 assert hashlib.sha256((p/'videos/rgb_cycles.mp4').read_bytes()).hexdigest()==d['video_sha256'],c
 test=R/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json';t=json.loads(test.read_text())
 for k,v in t.items():
  if k in ['source_video','input_video','input_video_8f','input_video_16f','input_image','metadata_json','manifest_json','captions_json','contacts_json','trajectories_npz']:assert Path(v).is_file(),(c,k,v)
 size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink());total_bytes+=size
 rows.append(dict(sample_id=c,family=d['family_key'],scene=cfg['scene_by_family'][d['family_key']]['label'],video=f'samples/{c}/videos/rgb_cycles.mp4',context8=f'samples/{c}/context/context8_cycles.mp4',context16=f'samples/{c}/context/context16_cycles.mp4',test_json=str(test.relative_to(R)),bytes=size,sha256=d['video_sha256']))
assert len(cases)==len(set(cases))==70
assert len((R/'testjsons/input_list.txt').read_text().splitlines())==70
release=dict(dataset='test70_v2',release='20261003_final_warm_slope_gap_light_domino'+str(cfg['scene_by_family']['V2V_DOMINO']['camera_adjustment']['yfov']),status='PASS',cases=70,groups=14,frames=6300,videos=70,context_videos=140,width=896,height=512,fps=30,sample_bytes=total_bytes,updated=time.time(),included=['RGB PNG frames','RGB MP4 videos','8/16-frame context videos','initial-state captions','camera','trajectories','contacts','metadata','test JSONs'],not_included=['aligned depth','segmentation masks','physics_supervision archives','model caches'],samples=rows)
(R/'dataset_release.json').write_text(json.dumps(release,ensure_ascii=False,indent=2))
# Synchronize visible labels and explicitly show the finalized release.
for family in groups:groups[family]['scene']=cfg['scene_by_family'][family]
(R/'group_assignments.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
p=R/'index.html';s=p.read_text();s=re.sub(r'const groups=.*?;const cards=',lambda _:'const groups='+json.dumps(groups,ensure_ascii=False)+';const cards=',s,count=1)
s=s.replace('<h1>Test70 v2</h1>','<h1>Test70 v2 · 新版完整数据集</h1>')
s=re.sub(r'<p>使用已确认的写实场景。.*?</p>', '<p>已完成 70/70：坡度组采用暖色素墙木地板，缝隙组采用前上方暖色柔光。<a href="dataset.html">完整数据集与文件列表</a> · <a href="testjsons/input_list.txt" download>下载70例测试列表</a></p>',s,count=1)
if cfg['scene_by_family']['V2V_DOMINO']['camera_adjustment']['yfov']==26:
 s=s.replace('缝隙组采用前上方暖色柔光。', '缝隙组采用前上方暖色柔光，多米诺组采用26°近景。')
p.write_text(s)
body=''.join('<tr><td>'+html.escape(x['sample_id'])+'</td><td>'+html.escape(x['scene'])+'</td>'+''.join('<td><a href="'+x[k]+'">'+label+'</a></td>' for k,label in [('video','视频'),('context8','ctx8'),('context16','ctx16'),('test_json','测试JSON')])+'</tr>' for x in rows)
(R/'dataset.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Test70 v2 完整数据集</title><style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:16px system-ui;background:#f7f5ef;color:#263334}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}a{color:#146c72}code{overflow-wrap:anywhere}table{width:100%;font-size:14px}</style><h1>Test70 v2 · 新版完整渲染数据集</h1><p>70 case / 14组 · 6300张RGB帧 · 70段完整视频 · 140段context视频 · 896×512 · 30 fps</p><p>已包含坡度组暖色素墙木地板、缝隙组新打光；各组使用当前确认的相机配置。每组5个case保持相同场景配置。</p><p>本机目录：<code>/data/gaoya/AAA_test_video/test70_v2</code></p><p><a href="./">全部视频</a> · <a href="dataset_release.json" download>完整清单及视频SHA256</a> · <a href="testjsons/input_list.txt" download>70例测试列表</a> · <a href="README.md">目录说明</a></p><p>包含逐帧RGB、视频、8/16帧context、初始状态提示词、相机、轨迹、接触记录及元数据。本版不包含重新对齐的深度、分割标注或模型缓存。</p><p>测试JSON使用本机绝对路径，迁移到其他机器时须更新路径。</p><table><thead><tr><th>Case</th><th>场景</th><th>视频</th><th>8帧</th><th>16帧</th><th>测试配置</th></tr></thead><tbody>'''+body+'</tbody></table></html>')
hub=Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/index.html');s=hub.read_text();marker='<!-- TEST70_V2_FINAL_RELEASE -->'
entry=marker+'<section class="entry" style="border-left:4px solid #168e98"><div><h2>Test70 v2 · 新版完整渲染数据集</h2><div class="meta">14组 / 70 case；暖色住宅坡度场景、缝隙组新打光；视频、context、轨迹及测试列表。</div><a href="/test70_v2/?v=20261003-final">查看全部70例</a><a href="/test70_v2/dataset.html">完整数据集</a></div><div class="status"><strong>70 / 70 完成</strong><small>6300帧 · 90帧/30fps</small></div></section>'
if marker not in s:s=s.replace('<div class="entries">','<div class="entries">'+entry,1);hub.write_text(s)
print(json.dumps({k:v for k,v in release.items() if k!='samples'},ensure_ascii=False))
