from pathlib import Path
import json,shutil,math,hashlib
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_v2/test70_v2')
V=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview')
assert not R.exists(),R
R.mkdir(parents=True)
cfg=json.loads((V/'config.json').read_text());groups=json.loads((V/'group_assignments.json').read_text());src=Path(cfg['source_version'])
for case in json.loads((V/'preview_cases.json').read_text()):
 report=json.loads((V/'cases'/case/'calibration_report.json').read_text());family=json.loads((V/'cases'/case/'metadata.json').read_text())['family_key'];p=cfg['scene_by_family'][family];fit=report['background_placement'];p['fixed_anchor']=fit['anchor'];p['forward']=[math.cos(fit['native_yaw']),math.sin(fit['native_yaw'])];p['fixed_room_yaw']=math.radians(report['room_yaw_degrees']);p['approved_reference_case']=case
cfg['dataset']='test70_v2';cfg['camera_policy']='Use per-case source camera with approved family adjustments; freeze background world transform per family.'
(R/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
(R/'group_assignments.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2))
cases=[c for g in groups.values() for c in g['cases']];assert len(cases)==len(set(cases))==70
(R/'case_order.json').write_text(json.dumps(cases,indent=2));(R/'logs').mkdir();(R/'reports').mkdir();(R/'testjsons/v2v_jsons/test70_v2_all_cycles').mkdir(parents=True)
for c in cases:
 p=R/'source_cases'/c;p.mkdir(parents=True);(R/'samples'/c).mkdir(parents=True)
 for name in ['metadata.json','trajectory.json','trajectory.npz','camera.json','contacts.json','simulation_report.json']:shutil.copy2(src/'cases'/c/name,p/name)
for name in ['egl_guard.py','fit_room.py']:shutil.copy2(V/name,R/name)
s=(V/'render.py').read_text().replace("P=R/'cases'/case;", "P=R/'source_cases'/case;")
s=s.replace("yaw=math.atan2(forward.y,forward.x)-fit['native_yaw']", "yaw=placement['fixed_room_yaw']")
s=s.replace("frames=P/('calibration_frames'", "OUT=R/'samples'/case\nframes=OUT/('calibration_frames'")
s=s.replace("scene.render.filepath=str(frames/f'frame_{frame+1:04d}.png');bpy.ops.render.render(write_still=True);print('FRAME_DONE',case,frame+1,flush=True)","""destination=frames/f'{frame:05d}.png'
  if not destination.exists():
   scene.render.filepath=str(frames/f'.{frame:05d}.partial.png');bpy.ops.render.render(write_still=True);Path(scene.render.filepath).replace(destination)
  progress=OUT/'render_progress.json.tmp';progress.write_text(json.dumps(dict(case=case,frame=frame+1,total=90,updated=time.time())));progress.replace(OUT/'render_progress.json');print('FRAME_DONE',case,frame+1,flush=True)""")
s=s.replace("(P/('calibration_report.json'", "(OUT/('calibration_report.json'")
compile(s,'render.py','exec');(R/'render.py').write_text(s)
for file in ['run_test70_v2.py']:
 shutil.copy2(Path(__file__).parent/file,R/file)
readme='''# test70_v2

70 cases / 14 groups. Test70 geometry, existing validated Test89 physics resimulation, approved photorealistic backgrounds. Original Test70 is unchanged.

- samples/<id>/frames/00000.png ... 00089.png: Cycles RGB, 896x512, 30fps.
- samples/<id>/videos/rgb_cycles.mp4 (rgb.mp4 alias).
- samples/<id>/context/context8_cycles.mp4 and context16_cycles.mp4.
- samples/<id>/metadata.json, camera.json, raw/trajectories.npz: effective rendered world coordinates; wxyz rotations.
- source_cases/: immutable pre-render physics inputs. Domino is translated upward 0.8m onto a finite supported tabletop; contacts retain source body IDs and refer to the translated support plane.
- testjsons/: only completed, validated samples are listed; initial-state prompts, no old outcome annotations.
- config.json: all scene, material, camera and GPU settings. Each family uses its approved representative's fixed background world transform.
- reports/pipeline_status.json and manifest.json: live progress and validation. A completed render is not automatically a human visual approval.

This render release does not copy old depth/masks, physics-supervision archives or model caches: these are not aligned with the new cameras and appearance. Fresh image-space supervision and model caches, if needed, must be generated separately.

GPU2 only. Resume: /home/gaoya/miniconda3/envs/sam/bin/python run_test70_v2.py
'''
(R/'README.md').write_text(readme)
(R/'dataset_meta.json').write_text(json.dumps(dict(dataset='test70_v2',sample_count=70,groups=14,frames=90,fps=30,width=896,height=512,source_physics=str(src),approved_preview=str(V),depth_masks='not_generated; never reuse old image-space ground truth',model_caches='not_generated',quaternion_order='wxyz'),indent=2))
print(R)
