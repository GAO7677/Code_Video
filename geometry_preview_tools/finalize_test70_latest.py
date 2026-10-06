from pathlib import Path
import json,subprocess,shutil,time,hashlib
B=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer');F=B/'20261006_test70_v2_door_room_fix';O=B/'20261003_test70_v2/test70_v2';N=B/'20261006_test70_v2_latest/test70_v2';M=B/'20261005_test70_v2_approved_motion_render'
read=lambda p:json.loads(p.read_text())
assert all((F/'samples'/c/'complete.json').exists() for c in read(F/'case_order.json')),'Fix did not complete; no deletion or final publication'
subprocess.run(['/usr/bin/python3','/home/gaoya/Code_Video/geometry_preview_tools/consolidate_test70_latest.py'],check=True)
manifest=read(N/'manifest.json');assert manifest['completed_count']==70 and not manifest['pending']
for rec in manifest['samples']:
 p=N/'samples'/rec['sample_id'];assert hashlib.sha256((p/'videos/rgb_cycles.mp4').read_bytes()).hexdigest()==rec['video_sha256']
 for media,n in [('videos/rgb_cycles.mp4',90),('context/context8_cycles.mp4',8),('context/context16_cycles.mp4',16)]:
  result=json.loads(subprocess.check_output(['/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(p/media)],text=True))['streams'][0]
  assert (result['width'],result['height'],int(result['nb_read_frames']),result['r_frame_rate'])==(896,512,n,'30/1'),(p,result)
replaced={'SCENE_DOOR_FRAME_BALL','SCENE_DOOR_FRAME','V2V_OBSTACLE_SIZE','V2V_OBSTACLE','V2V_PENDULUM_CABINET'}
oldgroups=read(O/'group_assignments.json');targets=[O/'samples'/c for f,g in oldgroups.items() if f in replaced for c in g['cases']]
# The five motion renders have been superseded by the consistent room correction.
targets += [M/'samples'/c for c in read(F/'case_order.json')]
audit=N/'reports/superseded_metadata';audit.mkdir(exist_ok=True);removed=[]
for p in targets:
 assert p.parent in [O/'samples',M/'samples'] and not p.is_symlink(),p
 if not p.exists():continue
 archive=audit/(p.parent.parent.name+'__'+p.name);archive.mkdir(exist_ok=True)
 for f in p.glob('*.json'):shutil.copy2(f,archive/f.name)
 size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file() and not f.is_symlink());shutil.rmtree(p);removed.append(dict(path=str(p),logical_bytes=size))
 if p.parent==M/'samples':p.symlink_to(N/'samples'/p.name,target_is_directory=True)
 else:
  t=O/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{p.name}.json'
  if t.exists():shutil.copy2(t,archive/'test_config.json');t.unlink()
for filename in ['index.html','dataset.html']:(O/filename).write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/test70_v2/dataset.html"><a href="/test70_v2/dataset.html">已更新至最新统一数据集</a>')
(O/'SUPERSEDED.md').write_text('本版本已被 /data/gaoya/AAA_test_video/test70_v2 替代；25例被拒绝的旧渲染已删除。旧清单仅作历史记录，不可用于测试。\n')
report=dict(stage='COMPLETE',validated_cases=70,validated_videos=210,deleted=removed,deleted_logical_bytes=sum(x['logical_bytes'] for x in removed),updated=time.time())
(N/'reports/consolidation_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False),flush=True)
