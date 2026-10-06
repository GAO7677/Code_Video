from pathlib import Path
import json,os,shutil,hashlib,time,html,collections,subprocess
BASE=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer')
OLD=BASE/'20261003_test70_v2/test70_v2'
MOTION=BASE/'20261005_test70_v2_approved_motion_render'
RAMP=BASE/'20261006_test70_v2_flat_ramp_render'
OUT=BASE/'20261006_test70_v2_latest/test70_v2'
ENTRY=Path('/data/gaoya/AAA_test_video/test70_v2')
HUB=Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub')
WEB=Path('/data/gaoya/agent-data/outputs_v1/geometry_sweep_bench/20260929_unified_evaluation')
def read(p):return json.loads(p.read_text())
def put(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.new');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');t.replace(p)
def clone(src,dst):
 src=Path(src);dst=Path(dst)
 if src.suffix in {'.png','.mp4','.npz'}:os.link(src,dst)
 else:shutil.copy2(src,dst)
 return dst
def rewrite(v,root):
 if isinstance(v,str):return v.replace(str(root)+'/samples/',str(ENTRY)+'/samples/').replace(str(root)+'/source_cases/',str(ENTRY)+'/source_cases/').replace(str(root)+'/testjsons/',str(ENTRY)+'/testjsons/')
 if isinstance(v,list):return [rewrite(x,root) for x in v]
 if isinstance(v,dict):return {k:rewrite(x,root) for k,x in v.items()}
 return v
FIX=BASE/'20261006_test70_v2_door_room_fix'
fix_ready=FIX.exists() and all((FIX/'samples'/c/'complete.json').exists() for c in read(FIX/'case_order.json'))
previous_sources={x['sample_id']:x['source_root'] for x in read(OUT/'provenance/sources.json')} if (OUT/'provenance/sources.json').exists() else {}
roots={};groups=read(OLD/'group_assignments.json');replaced=set()
for root in [MOTION,RAMP]+([FIX] if fix_ready else []):
 for c in read(root/'case_order.json'):
  family=read(root/'source_cases'/c/'metadata.json')['family_key'];replaced.add(family);roots[c]=root
order=[]
for family,g in groups.items():
 ids=[c for c,r in roots.items() if read(r/'source_cases'/c/'metadata.json')['family_key']==family] if family in replaced else g['cases']
 assert len(ids)==5
 for c in ids:roots.setdefault(c,OLD)
 order+=ids
assert len(order)==70 and len(set(order))==70
OUT.mkdir(parents=True,exist_ok=True)
for d in ['samples','source_cases','testjsons/v2v_jsons/test70_v2_all_cycles','reports','provenance']:(OUT/d).mkdir(parents=True,exist_ok=True)
bench=read(WEB/'test70_v2_bench.json');bench['cases']={};bench['consolidated_release']=True
bench['family_labels'].update(V2V_OBSTACLE='平地转上坡／初速度 · 窗光开阔房间',V2V_OBSTACLE_SIZE='固定门宽0.46 m／小球半径 · 经典橱柜厨房',V2V_PENDULUM_CABINET='固定摆锤／柜体位置 · 简洁石材房间')
records=[];release_rows=[];pending=[];tests=[];cfg=read(OLD/'config.json');assignments={};provenance=[]
for c in order:
 root=roots[c];src=root/'samples'/c;dst=OUT/'samples'/c;sm=read(root/'source_cases'/c/'metadata.json');family=sm['family_key'];done=(src/'complete.json').exists()
 cfg_src=read(root/'config.json')
 for key in ['scene_by_family','actor_materials']:
  cfg[key][family]=cfg_src[key][family]
 assignments.setdefault(family,dict(scene=cfg_src['scene_by_family'][family],cases=[]))['cases'].append(c)
 source=OUT/'source_cases'/c
 if previous_sources.get(c,str(root))!=str(root):
  if dst.exists():shutil.rmtree(dst)
  if source.exists():shutil.rmtree(source)
 if not source.exists():shutil.copytree(root/'source_cases'/c,source,copy_function=clone,symlinks=True)
 control=sm['control'];caption=sm.get('input_caption','')
 if done:
  if not dst.exists():shutil.copytree(src,dst,copy_function=clone,symlinks=True)
  for p in dst.rglob('*.json'):put(p,rewrite(read(p),root))
  test=read(root/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json');test=rewrite(test,root)
  for key in ['source_video','input_video','input_video_8f','input_video_16f','input_image','metadata_json','manifest_json','captions_json','contacts_json','trajectories_npz']:
   assert test[key].startswith(str(ENTRY)+'/'),(c,key,test[key])
   assert (OUT/Path(test[key]).relative_to(ENTRY)).exists(),(c,key)
  put(OUT/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json',test);tests.append(str(ENTRY/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json'))
  rec=read(dst/'complete.json');assert len(list((dst/'frames').glob('[0-9]*.png')))==90
  sha=hashlib.sha256((dst/'videos/rgb_cycles.mp4').read_bytes()).hexdigest();assert sha==rec['video_sha256'];records.append(rec)
  caption=test['input_caption'];size=sum(p.stat().st_size for p in dst.rglob('*') if p.is_file() and not p.is_symlink())
  release_rows.append(dict(sample_id=c,family=family,scene=cfg_src['scene_by_family'][family]['label'],video=f'samples/{c}/videos/rgb_cycles.mp4',context8=f'samples/{c}/context/context8_cycles.mp4',context16=f'samples/{c}/context/context16_cycles.mp4',test_json=f'testjsons/v2v_jsons/test70_v2_all_cycles/{c}.json',bytes=size,sha256=sha))
 else:
  pending.append(dict(sample_id=c,family=family,status='needs_visibility_fix',reason='末尾小球被背景地板遮挡；不进入测试列表',source=str(src)))
  dst.mkdir(exist_ok=True);put(dst/'pending.json',pending[-1]);sha='pending'
 bench['cases'][c]=dict(family=family,variant=control.get('value_label',str(control['value'])+' '+control.get('units','m')),control_label=f"{control['variable']} = {control['value']} {control.get('units','m')}",prompt=caption,gt_url=f'/test70_v2/samples/{c}/videos/rgb_cycles.mp4?v={sha}' if done else '',poster_url=f'/test70_v2/samples/{c}/frames/00000.png?v={sha}' if done else '',pending=not done,render_state='新版完整视频已完成 · 90/90帧' if done else '待修复：末尾小球被背景地板遮挡')
 provenance.append(dict(sample_id=c,source_root=str(root),published=done))
for name in ['input_list.txt','test70_v2_all_cycles_test70_ctx8.txt']:(OUT/'testjsons'/name).write_text('\n'.join(tests)+'\n')
put(OUT/'case_order.json',order);put(OUT/'group_assignments.json',assignments);put(OUT/'config.json',cfg)
put(OUT/'manifest.json',dict(dataset='test70_v2',expected_count=70,completed_count=len(records),samples=records,pending=pending))
release=read(OLD/'dataset_release.json');release.update(release='20261006_latest_approved_motion_flat_ramp',status='PASS' if not pending else 'PARTIAL_PENDING_VISIBILITY_FIX',cases=70,completed_cases=len(records),pending_cases=pending,frames=len(records)*90,videos=len(records),context_videos=len(records)*2,sample_bytes=sum(x['bytes'] for x in release_rows),updated=time.time(),samples=release_rows)
put(OUT/'dataset_release.json',release);meta=read(OLD/'dataset_meta.json');meta.update(completed_count=len(records),pending_count=len(pending),source_physics='source_cases/',approved_preview='See provenance/sources.json');put(OUT/'dataset_meta.json',meta);put(OUT/'provenance/sources.json',provenance)
for r in [OLD,MOTION,RAMP]+([FIX] if fix_ready else []):
 d=OUT/'provenance'/r.parent.name if r==OLD else OUT/'provenance'/r.name;d.mkdir(exist_ok=True)
 for p in r.iterdir():
  if p.is_file() and p.suffix in {'.py','.json'} and p.name not in ['manifest.json','dataset_release.json']:shutil.copy2(p,d/p.name)
put(OUT/'reports/pipeline_status.json',dict(stage=release['status'],total=70,completed=len(records),failed=pending,updated=time.time()))
(OUT/'README.md').write_text(f'''# Test70 v2 最新统一数据集\n\n入口：`{ENTRY}`\n\n14组、70个定义；目前通过检查 {len(records)} 例，待修复 {len(pending)} 例。`testjsons/input_list.txt` 只收录通过检查的样本，不混入旧版本。\n\n每例90帧，896×512，30fps，含完整视频、8/16帧context、初始状态提示词、相机、轨迹和接触记录。无新版深度、分割或模型缓存。\n\n- `case_order.json`：全部70例定义。\n- `manifest.json`：通过检查和待修复清单。\n- `source_cases/`：最新版物理输入。\n- `samples/`：已通过检查的最新渲染及元数据；未通过者仅含pending标记。\n- `testjsons/v2v_jsons/test70_v2_all_cycles/`：正式测试配置，路径统一使用上述稳定入口。\n- `provenance/`：各样本来源及渲染代码/配置快照。\n\n最新调整包括真实门洞、固定0.46m门宽改变球半径、固定1.2m摆长改变障碍物位置、平地转上坡（1.2/2.2/3.2/4.2/5.2m/s）。原Test70未修改。\n''')
rows=[]
for c,i in bench['cases'].items():
 media=f'<video controls preload="none" width="320" poster="samples/{c}/frames/00000.png" src="samples/{c}/videos/rgb_cycles.mp4"></video>' if not i['pending'] else '待修复，未发布视频'
 rows.append(f'<tr><td>{html.escape(c)}<br>{html.escape(i["control_label"])}</td><td>{media}</td></tr>')
doc=f'<!doctype html><meta charset="utf-8"><title>Test70 v2 最新版</title><h1>Test70 v2 最新统一数据集</h1><p>14组 / 70例定义；{len(records)}例通过检查，{len(pending)}例待修复。</p><p><a href="testjsons/input_list.txt">有效测试列表</a> · <a href="dataset_release.json">发布清单</a> · <a href="README.md">说明</a></p><table>'+''.join(rows)+'</table>'
for n in ['dataset.html','index.html']:(OUT/n).write_text(doc)
bench['release_status']={'completed':len(records),'total':70,'pending':pending};put(OUT/'bench.json',bench)
# Validate before publishing the stable pointers.
assert collections.Counter(i['family'] for i in bench['cases'].values())=={f:5 for f in groups}
assert len(tests)==len(records) and len(records)+len(pending)==70
for target in [ENTRY,HUB/'test70_v2']:
 assert target.is_symlink(),target
 tmp=target.with_name(target.name+'.latest_tmp');tmp.symlink_to(OUT);tmp.replace(target)
put(WEB/'test70_v2_bench.json',bench)
print(json.dumps(dict(output=str(OUT),completed=len(records),pending=pending),ensure_ascii=False,indent=2))
