from pathlib import Path
import json,shutil
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');old=R.parent/'20261003_test70_cycles_loft_preview';source=R.parent/'20261003_test70_group_object_preview';cache=Path('/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes');assets={a['displayName']:a for a in json.loads((R/'assets_catalog.json').read_text())}
# Native scene anchors are only initial placement hints. Visibility fitting selects a nearby open floor area.
plans=[
 ('Modern Loft Apartment','开放式阁楼',[0,-7.2,.1038],[0,1]),
 ('Interior Office','明亮办公室',[5,-9,.032],[-.927,.375]),
 ('Coffee Office bar','木地板咖啡吧',[1,-1.5,.06],[-.643,.766]),
 ('Cafe interior loft bar','工业风咖啡厅',[0,-6,.013],[.077,.995]),
 ('LUXURY interior house','挑高暖木住宅',[0,0,-.12],[0,1]),
 ("Kyra's Living Room",'日照木地板客厅',[3.5,0,0],[-1,0]),
 ('Butterfly Mural Room','壁画客厅',[3.8,0,.002],[-1,0]),
 ('Hallway indoor interior','绿植休闲厅',[0,-6.5,0],[.615,.787]),
 ('Home minimal interior','浅色现代起居室',[0,0,.065],[0,1]),
 ('kitchen_Scene_06','经典橱柜厨房',[-2,0,.02],[1,0]),
 ('Coffee Cafe interior','窗边咖啡店',[-.4,-2,-.56],[0,1]),
 ('Luxury Dining Room Interior','石材餐厅',[0,-2.8,.025],[0,1]),
 ('Indoor luxury interior minimal style','深木双层客厅',[.1,-2.8,-1.1],[0,1]),
 ('Studio Apartment Furnished','暖光小公寓',[1,1,.09],[-1,0]),
]
cases=json.loads((source/'preview_cases.json').read_text());appearance=json.loads((source/'appearance.json').read_text());cfg=json.loads((old/'config.json').read_text());cfg['scene_by_family']={};cfg['actor_materials']={};cfg['samples']=32;cfg['preview_samples']=32;cfg['calibration_frames']=[1];cfg['preview_only']=True;cfg['exposure']=0;cfg['source_version']=str(source)
labels=['桌高','坡度','坡长','门洞宽度（球）','门洞宽度（木块）','挡板法线角度','碗曲率','多米诺间距','缝隙宽度','障碍尺寸','碰撞速度','摆锤与柜体高度','摆长','跷跷板位置']
for c,(name,label,anchor,forward),tasklabel in zip(cases,plans,labels):
 p=R/'cases'/c;p.mkdir(parents=True,exist_ok=True)
 for filename in ['metadata.json','trajectory.json','trajectory.npz','camera.json','contacts.json','simulation_report.json']:
  src=source/'cases'/c/filename
  if src.exists():shutil.copy2(src,p/filename)
 m=json.loads((p/'metadata.json').read_text());f=m['family_key'];ap=appearance['actor_by_family'][f];ball=ap.get('puck',ap['sphere']);cfg['actor_materials'][f]=dict(ball_srgb=ball['color_srgb'],ball_accent=ball['accent_srgb'],ball_pattern=ball['pattern'],wood=ap['box']['texture'],support_wood='dark_wood')
 if name=='Modern Loft Apartment':assetid='05057669-22df-4e62-a03a-b665524c7135';blend=cache/'modern_loft/scene.blend'
 else:assetid=assets[name]['assetBaseId'];blend=cache/assetid/'scene.blend'
 cfg['scene_by_family'][f]=dict(name=name,label=label,task_label=tasklabel,asset_id=assetid,scene_blend=str(blend),anchor_hint=anchor,forward=forward,search_radius=1.5)
(R/'config.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False));(R/'preview_cases.json').write_text(json.dumps(cases,indent=2));shutil.copy2(old/'egl_guard.py',R/'egl_guard.py')
s=(old/'render.py').read_text()
s=s.replace("bpy.ops.wm.open_mainfile(filepath=cfg['scene_blend'])", "placement=cfg['scene_by_family'][family];cfg['scene_blend']=placement['scene_blend']\nbpy.ops.wm.open_mainfile(filepath=cfg['scene_blend']);bpy.context.scene.frame_set(bpy.context.scene.frame_start)")
s=s.replace("# Preserve every native scene material and light. Only move the room around the unchanged task.", "# Fit each native background to the unchanged task and camera.\nfrom fit_room import fit_room\nfit=fit_room(scene,placement,cam,m,tr,P);cfg['room_anchor']=fit['anchor']\n# Preserve native scene materials and lights.")
s=s.replace("yaw=math.atan2(forward.y,forward.x)-math.pi/2", "yaw=math.atan2(forward.y,forward.x)-fit['native_yaw']")
s=s.replace("if family=='F11':", "if profile.get('ball_pattern')=='bands':")
s=s.replace("mat=ball_material(name) if a['shape']=='sphere'", "mat=ball_material(name) if a['shape'] in ('sphere','puck')")
s=s.replace("scene.render.filepath=str(frames/f'frame_{frame+1:04d}.png')", "scene.render.filepath=str(frames/f'frame_{frame+1:04d}.png')")
s=s.replace("background_object_count=len(background_names)", "background_object_count=len(background_names),background_placement=fit")
(R/'render.py').write_text(s)
print('Prepared',len(cases),'unique scene assignments')
