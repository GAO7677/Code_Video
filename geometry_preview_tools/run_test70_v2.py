from pathlib import Path
import os,json,time,subprocess,fcntl,traceback,hashlib,math,shutil
import numpy as np
from PIL import Image
R=Path(__file__).resolve().parent
CFG=json.loads((R/'config.json').read_text());CASES=json.loads((R/'case_order.json').read_text())
FF='/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg';PROBE=str(Path(FF).with_name('ffprobe'))
PROMPTS=Path('/data/gaoya/agent-data/physv_v2v_0819/manifests/test70_initial_state_v1')
def put(path,obj):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)
def refresh(stage,current=None,failed=None):
 samples=[];lists=[]
 for c in CASES:
  done=R/'samples'/c/'complete.json'
  if done.exists():
   samples.append(json.loads(done.read_text()));lists.append(str(R/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json'))
 put(R/'manifest.json',dict(dataset='test70_v2',expected_count=70,completed_count=len(samples),samples=samples))
 for n in ['test70_v2_all_cycles_test70_ctx8.txt','input_list.txt']:
  p=R/'testjsons'/n;t=p.with_suffix('.tmp');t.write_text(''.join(x+'\n' for x in lists));t.replace(p)
 put(R/'reports/pipeline_status.json',dict(stage=stage,gpu=2,gpu_uuid=CFG['gpu_uuid'],total=70,completed=len(samples),current=current,failed=failed or [],updated=time.time()))
 return len(samples)
def encode(source,output,frames):
 tmp=output.with_name(output.stem+'.partial.mp4')
 cmd=[FF,'-hide_banner','-loglevel','error','-y','-threads','2','-framerate','30','-start_number','0','-i',str(source/'%05d.png'),'-frames:v',str(frames),'-c:v','libx264','-threads','2','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)]
 subprocess.run(cmd,check=True)
 probe=json.loads(subprocess.check_output([PROBE,'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames,r_frame_rate','-of','json',str(tmp)],text=True))['streams'][0]
 assert (probe['width'],probe['height'],int(probe['nb_read_frames']),probe['r_frame_rate'])==(896,512,frames,'30/1'),probe
 tmp.replace(output)
def export(c):
 p=R/'samples'/c;s=R/'source_cases'/c
 report=json.loads((p/'render_report.json').read_text());assert report['gpu']['uuid']==CFG['gpu_uuid'] and report['engine']=='CYCLES' and not report['missing_images'];assert report['frames_rendered']==list(range(1,91))
 for i in range(90):
  with Image.open(p/'frames'/f'{i:05d}.png') as im:assert im.size==(896,512);im.verify()
 for d in ['videos','context','raw','captions']:(p/d).mkdir(exist_ok=True)
 encode(p/'frames',p/'videos/rgb_cycles.mp4',90)
 for n in [8,16]:encode(p/'frames',p/'context'/f'context{n}_cycles.mp4',n)
 for parent,a,b in [('videos','rgb.mp4','rgb_cycles.mp4'),('context','context8.mp4','context8_cycles.mp4'),('context','context16.mp4','context16_cycles.mp4')]:
  link=p/parent/a
  if not link.exists():link.symlink_to(b)
 m=json.loads((s/'metadata.json').read_text());family=m['family_key'];placement=CFG['scene_by_family'][family];offset=np.array(placement.get('task_origin_offset',[0,0,0]));tr=json.loads((s/'trajectory.json').read_text())
 arrays={k:np.asarray(v) for k,v in tr.items()}
 for k in arrays:
  if k.endswith('_positions'):arrays[k]=arrays[k]+offset
 np.savez_compressed(p/'raw/trajectories.npz',**arrays)
 for name,a in m['actors'].items():
  if 'initial_position_m' in a:a['initial_position_m']=(np.array(a['initial_position_m'])+offset).tolist()
 cam=report['camera'];eye=np.array(cam['position']);forward=np.array(cam['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);up=np.cross(right,forward);matrix=np.eye(4);matrix[:3,:3]=np.stack([right,up,-forward],axis=1);matrix[:3,3]=eye;f=256/math.tan(math.radians(cam['yfov'])/2)
 camera=dict(intrinsics=dict(fx=f,fy=f,cx=448,cy=256,width=896,height=512,yfov_deg=cam['yfov']),extrinsics=dict(eye=cam['position'],target=cam['target'],up=up.tolist(),camera_to_world=matrix.tolist(),convention='Blender/OpenGL camera looks along -Z, +Y up'))
 meta={k:m[k] for k in ['sample_id','task_type','split','source_group','family_key','seed','title','gravity_mps2','simulation','control','actors','resimulation'] if k in m}
 meta.update(dataset='test70_v2',schema_version='test70_v2_render_v1',camera=camera,render_task_offset=offset.tolist(),render_support=placement.get('tabletop'),background=placement,materials=CFG['actor_materials'][family],source_physics=str(s),contact_coordinate_note='Body IDs and contact distances/forces from source simulation; rigid translation leaves them invariant. Domino original floor corresponds to tabletop support. No contact positions supplied.')
 put(p/'metadata.json',meta);put(p/'meta.json',meta);put(p/'camera.json',camera);shutil.copy2(s/'contacts.json',p/'contacts.json');shutil.copy2(s/'simulation_report.json',p/'simulation_report.json')
 old=json.loads((PROMPTS/f'{c}.json').read_text());caption=old['input_caption']
 if family=='V2V_DOMINO':caption=caption.replace('on the floor','on a table').replace('on a floor','on a table')
 captions=dict(input_caption=caption,caption_variant='initial_state_only',source=str(PROMPTS/f'{c}.json'));put(p/'captions/captions.json',captions);(p/'captions/initial_state.txt').write_text(caption+'\n')
 warnings=[dict(frame=d['frame'],**a) for d in report['diagnostics'] for a in d['objects'] if not a['bbox_in_frame'] or a['visible_fraction']==0]
 record=dict(sample_id=c,family_key=family,status='render_complete',sample_dir=str(p),frame_count=90,qa_warning_count=len(warnings),human_visual_review='pending',video_sha256=hashlib.sha256((p/'videos/rgb_cycles.mp4').read_bytes()).hexdigest())
 put(p/'render_validation.json',dict(gpu_verified=True,frames_verified=90,video_contexts_verified=[90,8,16],geometry_visibility_warnings=warnings))
 test={k:meta[k] for k in ['sample_id','dataset','schema_version','family_key','task_type','source_group','control']}
 test.update(input_caption=caption,caption_variant='initial_state_only',source_video=str(p/'videos/rgb_cycles.mp4'),input_video=str(p/'context/context8_cycles.mp4'),input_video_8f=str(p/'context/context8_cycles.mp4'),input_video_16f=str(p/'context/context16_cycles.mp4'),input_image=str(p/'frames/00000.png'),metadata_json=str(p/'metadata.json'),manifest_json=str(p/'manifest.json'),captions_json=str(p/'captions/captions.json'),contacts_json=str(p/'contacts.json'),trajectories_npz=str(p/'raw/trajectories.npz'),frame_counts=dict(source_video=90,input_video_context8=8,input_video_16f=16),video_spec=dict(width=896,height=512,fps=30,source_frame_count=90),conditioning=dict(type='video_context',context_frame_options=[8,16],target_video='videos/rgb_cycles.mp4'))
 put(R/'testjsons/v2v_jsons/test70_v2_all_cycles'/f'{c}.json',test);put(p/'manifest.json',record);put(p/'complete.json',record)

def main():
 failed=[];refresh('WAITING_GPU2')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PREVIEW_PHYSICAL_GPU='2',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',LD_PRELOAD='/data/gaoya/dataset/new_data0826/generic_v3_geometry_sweep_20260929/egl_select_visible.so',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for k in ['DISPLAY','EGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','__GLX_VENDOR_LIBRARY_NAME']:env.pop(k,None)
 with Path('/data/gaoya/agent-data/locks/cosmos_gpu/gpu2.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  active=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if active:refresh('GPU2_OCCUPIED');raise RuntimeError(active)
  for c in CASES:
   p=R/'samples'/c
   if (p/'complete.json').exists():continue
   if shutil.disk_usage(R).free<10*1024**3:refresh('LOW_DISK',c,failed);raise RuntimeError('Less than 10GiB free; stopping safely')
   refresh('RENDERING',c,failed);print('START',c,flush=True)
   try:
    # Reject corrupt frames before resuming; never reuse an incomplete PNG.
    for frame in (p/'frames').glob('*.png') if (p/'frames').exists() else []:
     try:
      with Image.open(frame) as im:assert im.size==(896,512);im.verify()
     except Exception:frame.rename(frame.with_suffix('.invalid'))
    with (R/'logs'/f'{c}.log').open('a') as log:
     subprocess.run([CFG['blender'],'-b','--disable-autoexec','-t','4','--python-exit-code','1','--python',str(R/'render.py'),'--',c],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    refresh('EXPORTING',c,failed);export(c);refresh('RUNNING',c,failed);print('DONE',c,flush=True)
   except Exception:
    failed.append(c);traceback.print_exc();put(p/'failure.json',dict(error=traceback.format_exc(),updated=time.time()));refresh('RUNNING_WITH_FAILURES',c,failed)
  refresh('COMPLETE' if not failed else 'PARTIAL_FAILURE',failed=failed)
if __name__=='__main__':main()
