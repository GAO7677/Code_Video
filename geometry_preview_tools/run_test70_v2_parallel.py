"""Single controller owns GPU2 lock and manifests; each child owns one case."""
from pathlib import Path
import json,os,time,subprocess,fcntl,signal,shutil,traceback
import run_test70_v2 as base
R=Path(__file__).resolve().parent
settings=json.loads((R/'parallel_config.json').read_text())
active={};failed=[];last_launch=0;stopping=False

def gpu_free():
 return int(subprocess.check_output(['nvidia-smi','-i','2','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
def status(stage):
 workers=[dict(case=c,pid=w['process'].pid,started=w['started']) for c,w in active.items()]
 base.refresh(stage,workers[0]['case'] if workers else None,failed)
 p=R/'reports/pipeline_status.json';s=json.loads(p.read_text());s.update(workers=workers,max_workers=settings['max_workers'],scheduler='parallel_gpu2',controller_pid=os.getpid());base.put(p,s)
def stop(sig,frame):
 global stopping
 stopping=True
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)

def main():
 global last_launch
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='2',PREVIEW_PHYSICAL_GPU='2',__EGL_VENDOR_LIBRARY_FILENAMES='/usr/share/glvnd/egl_vendor.d/10_nvidia.json',LD_PRELOAD='/data/gaoya/dataset/new_data0826/generic_v3_geometry_sweep_20260929/egl_select_visible.so',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 for k in ['DISPLAY','EGL_PLATFORM','LIBGL_ALWAYS_SOFTWARE','MESA_LOADER_DRIVER_OVERRIDE','__GLX_VENDOR_LIBRARY_NAME']:env.pop(k,None)
 with Path('/data/gaoya/agent-data/locks/cosmos_gpu/gpu2.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  occupied=subprocess.check_output(['nvidia-smi','-i','2','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
  if occupied:raise RuntimeError('GPU2 has unowned processes: '+occupied)
  pending=[c for c in base.CASES if not (R/'samples'/c/'complete.json').exists()]
  while pending or active:
   for c,w in list(active.items()):
    code=w['process'].poll()
    if code is None:continue
    w['log'].close();del active[c]
    try:
     if code:raise RuntimeError(f'Blender exit code {code}')
     base.export(c);(R/'samples'/c/'failure.json').unlink(missing_ok=True);print('DONE',c,flush=True)
    except Exception:
     failed.append(c);base.put(R/'samples'/c/'failure.json',dict(error=traceback.format_exc(),updated=time.time()));traceback.print_exc()
   if stopping and not active:status('PAUSED');return
   free_disk=shutil.disk_usage(R).free
   if pending and not stopping and free_disk<settings['min_disk_gib']*1024**3:
    if not active:status('LOW_DISK');return
   elif pending and not stopping and len(active)<settings['max_workers'] and time.time()-last_launch>=settings['launch_interval_seconds']:
    # Wait for all existing workers to initialize before allocating another scene.
    warmed=all((R/'samples'/c/'render_progress.json').exists() and (R/'samples'/c/'render_progress.json').stat().st_mtime>w['started'] for c,w in active.items())
    if (not active or warmed) and gpu_free()>=settings['min_free_before_launch_mib']:
     c=pending.pop(0);p=R/'samples'/c
     for frame in (p/'frames').glob('[0-9][0-9][0-9][0-9][0-9].png') if (p/'frames').exists() else []:
      try:
       with base.Image.open(frame) as im:assert im.size==(896,512);im.verify()
      except Exception:frame.rename(frame.with_suffix('.invalid'))
     log=(R/'logs'/f'{c}.log').open('a');started=time.time()
     child=subprocess.Popen([base.CFG['blender'],'-b','--disable-autoexec','-t','4','--python-exit-code','1','--python',str(R/'render.py'),'--',c],env=env,stdout=log,stderr=subprocess.STDOUT)
     active[c]=dict(process=child,log=log,started=started);last_launch=started;print('START',c,child.pid,flush=True)
   status('DRAINING' if stopping else 'RENDERING_PARALLEL')
   if active or pending:time.sleep(10)
  status('COMPLETE' if not failed else 'PARTIAL_FAILURE')
if __name__=='__main__':main()
