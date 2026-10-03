from pathlib import Path
import json,hashlib,zipfile,html
from PIL import Image
import markdown
D=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
R=D.parent.parent/'20261003_test70_v2_bench_documentation';R.mkdir(exist_ok=True);(R/'assets').mkdir(exist_ok=True)
revision=json.loads((D.parent.parent/'20261003_test70_v2_domino_closeup/revision_status.json').read_text());assert revision['stage']=='PUBLISHED','Wait for latest domino closeups'
groups=json.loads((D/'group_assignments.json').read_text());cfg=json.loads((D/'config.json').read_text());assert cfg['scene_by_family']['V2V_DOMINO']['camera_adjustment']['yfov']==26
labels={'F11':('桌面高度','观察小球离开桌缘后的运动。'),'F12':('斜坡倾角','观察木块沿不同坡度斜面滑动。'),'F12_RAMP_LENGTH':('斜坡长度','观察斜面路径长度变化后的运动。'),'SCENE_DOOR_FRAME_BALL':('门洞净宽（球）','观察小球与不同宽度门洞的相互作用。'),'SCENE_DOOR_FRAME':('门洞净宽（木块）','观察木块与不同宽度门洞的相互作用。'),'SCENE_PUCK_BARRIER':('挡板法线角度','观察圆饼与不同朝向挡板的碰撞。'),'V2V_BOWL':('碗面半径 R','元数据控制半径，数值不是曲率；若采用圆弧曲率记法，曲率为 1/R。'),'V2V_DOMINO':('多米诺间距','观察触发小球推动多米诺后的连锁运动。'),'V2V_GAP':('台面缝隙宽度','观察小球到达两段台面之间缝隙时的运动。'),'V2V_OBSTACLE_SIZE':('小球半径 r','页面原标签为“障碍尺寸”，实际控制球半径，障碍物尺寸固定。'),'V2V_OBSTACLE':('设定初始速度','控制初始速度，不是测量碰撞发生瞬间的速度。'),'V2V_PENDULUM_CABINET':('摆锤悬点高度','柜体主体高度固定为 2.55 m；主控制量为摆锤悬点高度。'),'V2V_PENDULUM':('摆长','观察不同摆长条件下的摆动。'),'V2V_SEESAW':('载荷位置 x','控制载荷沿世界 x 轴的位置；支点位于 x=0 附近。')}
def val(v,unit):
 n=format(v,'.10g');return n+('°' if unit=='deg' else ' '+unit)
records=[]
for number,(family,g) in enumerate(groups.items(),1):
 variants=[]
 for c in g['cases']:
  p=D/'samples'/c;m=json.loads((p/'metadata.json').read_text());control=m['control'];img=p/'frames/00000.png';target=R/'assets'/f'{c}.jpg'
  Image.open(img).convert('RGB').save(target,quality=93,subsampling=0)
  variants.append(dict(case_id=c,variable=control['variable'],value=control['value'],unit=control['units'],label=val(control['value'],control['units']),image=str(target.relative_to(R)),source_image=str(img),source_sha256=hashlib.sha256(img.read_bytes()).hexdigest()))
 records.append(dict(number=number,family=family,title=labels[family][0],note=labels[family][1],scene=cfg['scene_by_family'][family]['label'],variants=variants))
intro='''# Test70 v2：可控物理视频续写 Bench

**版本：2026-10-03 · 14 组 × 5 个变体 = 70 个 case · 最新写实渲染版**

Test70 v2 是用于观察视频续写模型能否响应物理条件变化的一组受控场景。它沿用 Test70 的几何设计，以已有的 Test89 物理设置重新仿真，再采用写实室内背景、材质和光照进行渲染。每组围绕一个主控制参数构造 5 个变体；各组分别覆盖桌缘运动、斜坡、门洞、碰撞、多米诺、缝隙、摆锤和跷跷板等情形。

## 数据与使用方式

每个 case 提供 **90 帧、896×512、30 fps** 的 RGB 视频及 PNG 帧，并提供前 **8 / 16 帧** context、初始状态提示词、相机、轨迹、接触记录和元数据。编码视频长 3 秒，首末采样时刻跨度为 89/30≈2.967 秒。物理状态采样配置为 5688 Hz。使用时以 context 和初始状态提示词为条件，比较模型后续生成与对应仿真参考视频；跨变体比较应保持模型、推理步数、随机种子策略和指标配置一致。

本版包含坡度组暖色素墙与木地板、缝隙组前上方暖色柔光，以及多米诺组 **26° 垂直视角的近景镜头**。同组采用一致的背景和材质配置；相机保留每例的既定取景及已确认的组级调整。本文不宣称所有组都严格只改变单一因素，几何派生量或各例取景可能不同，精确复现应读取对应 metadata 与 camera 文件。

## 主控制变量与具体数值

以下数值直接读取 `metadata.json → control.value`，单位读取 `control.units`，不用 case 名称或经过舍入的 `value_label` 反推。数值是设计控制参数，不等同于离散仿真帧中的测量值。每组的 5 个变体在图示中按以下顺序从左向右排列。

| 编号 | 场景组 / 主控制量 | 元数据字段 | 5 个变体数值 |
|---|---|---|---|
'''
for g in records:
 intro+='| '+str(g['number'])+' | '+g['title']+' | `'+g['variants'][0]['variable']+'` | '+' / '.join(v['label'] for v in g['variants'])+' |\n'
intro+='''
**口径说明：**“障碍尺寸”组控制的是小球半径；“碰撞速度”组控制的是设定初始速度；“摆锤与柜体高度”组控制悬点高度，柜体主体高度固定为 2.55 m。多米诺间距包含 0.045、0.135 m；跷跷板位置包含 0.2925、0.585、0.8775 m，均保留原始精度。`v2v_bowl_r229` 的控制值为 2.3 m，以元数据为准。

**范围：**本说明展示初始条件，不依据首帧推断后续物理结果。本版尚未包含与新渲染对齐的深度、分割标注、physics-supervision 归档或模型缓存；这些不能沿用旧渲染版本。
'''
md=intro
for start in range(0,14,4):
 chunk=records[start:start+4]
 md+='\n<div class="page-break"></div>\n\n## 全变体首帧图谱 · 第 '+str(start+1)+'–'+str(start+len(chunk))+' 组\n\n每组一行；五列从左到右对应控制参数值。图片均取自该 case 的 `frames/00000.png`，未裁剪。\n\n'
 md+='| 场景组与控制变量 | 变体 1 | 变体 2 | 变体 3 | 变体 4 | 变体 5 |\n|---|---|---|---|---|---|\n'
 for g in chunk:
  left='**'+str(g['number'])+'. '+g['title']+'**<br>'+g['scene']+'<br><code>'+g['variants'][0]['variable']+'</code><br>'+g['note']
  cells=['**'+v['label']+'**<br>!['+v['case_id']+']('+v['image']+')<br><code>'+v['case_id']+'</code>' for v in g['variants']]
  md+='| '+left+' | '+' | '.join(cells)+' |\n'
md+='''

## 文件与复现入口

- 本机数据集：`/data/gaoya/AAA_test_video/test70_v2`。
- 可视化：http://10.176.42.45:8844/test70_v2/ 。
- 测试列表：`testjsons/input_list.txt`；完整清单：`dataset_release.json`。
- 每例元数据：`samples/<case_id>/metadata.json`；相机：`camera.json`；轨迹：`raw/trajectories.npz`。
- 文档附带 `bench_controls.json`，记录每个变体的精确控制值、首帧来源与 SHA256。Markdown 图片位于同目录的 `assets/`；移动文档时请一并保留。
'''
(R/'test70_v2_bench.md').write_text(md)
(R/'bench_controls.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
body=markdown.markdown(md,extensions=['tables','fenced_code'])
css='''@page{size:A3 landscape;margin:12mm 12mm 14mm}*{box-sizing:border-box}body{font-family:"Noto Sans CJK SC",sans-serif;font-size:12px;line-height:1.48;color:#203536;margin:0}h1{font-size:26px;color:#164c4e;margin:0 0 10px}h2{font-size:17px;margin:12px 0 7px}p{margin:7px 0}table{border-collapse:collapse;width:100%;table-layout:fixed;margin:8px 0}th{background:#174e50;color:white;font-size:11px;padding:6px}td{border:1px solid #d4ddda;padding:6px;vertical-align:top;font-size:11px;overflow-wrap:anywhere}tr{break-inside:avoid}img{display:block;width:100%;height:auto;margin:5px 0}code{font-family:"Noto Sans Mono CJK SC",monospace;font-size:9px;overflow-wrap:anywhere;white-space:normal}a{color:#146c72}table:first-of-type th:nth-child(1){width:5%}table:first-of-type th:nth-child(2){width:20%}table:first-of-type th:nth-child(3){width:26%}table:first-of-type th:nth-child(4){width:49%}table:not(:first-of-type) th:first-child{width:18%}table:not(:first-of-type) td:first-child{font-size:11px}table:not(:first-of-type) td:not(:first-child){text-align:center}table:not(:first-of-type) td:not(:first-child) strong{font-size:14px;color:#174e50}.page-break{break-before:page}li{margin:2px 0}'''
(R/'test70_v2_bench.html').write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>Test70 v2 Bench</title><style>'+css+'</style></head><body>'+body+'</body></html>')
with zipfile.ZipFile(R/'test70_v2_bench_md_assets.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in [R/'test70_v2_bench.md',R/'bench_controls.json',*sorted((R/'assets').glob('*.jpg'))]:z.write(p,p.relative_to(R))
print('BUILT',R,'groups',len(records),'images',sum(len(x['variants']) for x in records))
