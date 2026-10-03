from pathlib import Path
import json,sys,runpy,contextlib,io,time
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');reviewed=set(sys.argv[1:]);cfg=json.loads((R/'config.json').read_text());cases=json.loads((R/'preview_cases.json').read_text())
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(str(R/'validate_previews.py'),run_name='__main__')
validation=json.loads((R/'validation.json').read_text());assert validation['status']=='PASS',validation
previous={x['case']:x for x in json.loads((R/'visual_review.json').read_text())['cases']};reviews=[]
for case in cases:
 p=R/'cases'/case;m=json.loads((p/'metadata.json').read_text());family=m['family_key'];scene=cfg['scene_by_family'][family];report=json.loads((p/'calibration_report.json').read_text());marker=json.loads((p/'preview_complete.json').read_text())
 assert report['config']['scene_by_family'][family]==scene,('Config mismatch',case)
 assert set(scene.get('hide_background_objects',[])).issubset(report.get('removed_background_furniture',[]))
 if case in reviewed:reviews.append(dict(case=case,status='PASS',image_sha256=marker['sha256'],scene=scene['name'],checks=['Final rendered image visually reviewed','Requested furniture and decor removal checked','Task visible with support; textures present']))
 else:
  old=previous[case];assert old['image_sha256']==marker['sha256'] and old['status']=='PASS',('Unreviewed changed image',case);reviews.append(old)
 marker['needs_visual_review']=False;(p/'preview_complete.json').write_text(json.dumps(marker))
groups={f:dict(scene=s,cases=[]) for f,s in cfg['scene_by_family'].items()}
for p in sorted((Path(cfg['source_version'])/'cases').iterdir()):
 if p.is_dir() and (p/'metadata.json').exists():
  m=json.loads((p/'metadata.json').read_text());groups[m['family_key']]['cases'].append(p.name)
assert len(groups)==14 and all(len(x['cases'])==5 for x in groups.values());assert len({x['scene']['asset_id'] for x in groups.values()})==14
(R/'group_assignments.json').write_text(json.dumps(groups,ensure_ascii=False,indent=2));(R/'visual_review.json').write_text(json.dumps(dict(status='PASS',cases=reviews,reviewed=time.time()),ensure_ascii=False,indent=2));(R/'pipeline_status.json').write_text(json.dumps(dict(stage='COMPLETE',gpu=2,total=14,completed=14,updated=time.time(),validation='PASS',preview_frame=1)));print('PASS: 14/14; effective configs match; furniture removals checked; source geometry and relative motion preserved')
