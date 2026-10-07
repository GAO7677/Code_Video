#!/usr/bin/env python3
"""Actual integrated simulations, not outcome-conditioned animations.
Usage: python simulate.py --out results
       python simulate.py --families 3 6 --save-frames
       python simulate.py --physics-only --sim-hz 1440 --out fine_check
The included NumPy solvers are simplified prototypes, not validated 3-D GT.
"""
from __future__ import annotations
import argparse, json, sys, time, warnings, importlib.metadata
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio.v2 as imageio
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from scenes import FAMILIES, summarize
from physics import SphereWorld3
from render import Renderer, framed_image, font


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def writer(path,fps):
    return imageio.get_writer(str(path),fps=fps,codec='libx264',quality=8,macro_block_size=2,
        ffmpeg_log_level='error',output_params=['-movflags','+faststart','-threads','2','-pix_fmt','yuv420p'])


def geometry(scene):
    w=scene.world
    if isinstance(w,SphereWorld3):
        return {'sphere':{'position':w.p.tolist(),'velocity':w.v.tolist(),'radius':w.r,
             'mass':w.mass,'inertia':w.inertia,'friction':w.friction,'restitution':w.restitution,
             'angular_velocity':w.omega.tolist()},'static_boxes':[
             {'id':b.name,'center':b.center.tolist(),'half_extents':b.half.tolist(),
              'friction':b.friction,'restitution':b.restitution} for b in w.boxes]}
    return {'gravity':w.gravity.tolist(),'bodies':[
        {'id':b.name,'shape':b.shape,'position':b.p.tolist(),'velocity':b.v.tolist(),
         'angle':b.angle,'angular_velocity':b.omega,'mass':b.mass,'inertia':b.inertia,
         'radius':b.radius,'local_vertices':b.vertices.tolist() if b.vertices is not None else None,
         'friction':b.friction,'restitution':b.restitution,'ideal_fixed_pivot':b.pinned}
        for b in w.bodies]}


def run_case(build,value,idx,a):
    s=build(value);out=a.out/s.key/f'v{idx:02d}';out.mkdir(parents=True,exist_ok=True)
    m=s.metadata();m.update(fps=a.fps,frames=a.frames,context_frames=a.context,simulation_hz=a.sim_hz,
        renderer='CPU orthographic z-buffer',width=a.width,height=a.height,sample_start_s=0.,
        sample_last_s=(a.frames-1)/a.fps,parameters_are_prototype=True,
        source='newly constructed geometry-demo prototypes')
    dump(out/'metadata.json',m);dump(out/'initial_geometry.json',geometry(s))
    rend=None if a.physics_only else Renderer(s,a.width,a.height,a.aa)
    vid=None if a.physics_only else writer(out/'rgb.mp4',a.fps)
    ctx=None if a.physics_only else writer(out/f'context_{a.context}.mp4',a.fps)
    states=[];times=[];outside=[];steps=a.sim_hz//a.fps
    try:
        for k in range(a.frames):
            st=s.world.states();states.append(st);times.append(k/a.fps)
            if not all(np.isfinite(v['position']).all() for v in st.values()):
                raise FloatingPointError(f'Non-finite state: {s.key}, frame{k}')
            if rend:
                im=rend.image();arr=np.asarray(im);vid.append_data(arr)
                if k<a.context:ctx.append_data(arr)
                for index,name in [(0,'first.png'),(a.context-1,'context_last.png'),(48,'frame_48.png'),(a.frames-1,'last.png')]:
                    if k==index:im.save(out/name)
                if a.save_frames:
                    d=out/'frames';d.mkdir(exist_ok=True);im.save(d/f'{k:05d}.png')
                p=np.array(st[s.target]['position'])
                if s.plane=='planar_xy':p=np.r_[p,1.02]
                elif s.plane=='planar_xz':p=np.array([p[0],0.,p[1]])
                q=rend.project([p])[0]/a.aa
                if not(0<=q[0]<a.width and 0<=q[1]<a.height):outside.append(k)
            if k<a.frames-1:
                for _ in range(steps):s.world.step(1/a.sim_hz)
    finally:
        if vid:vid.close()
        if ctx:ctx.close()
    r=summarize(s,states,times);r['target_center_out_of_frame_indices']=outside
    dump(out/'summary.json',r);dump(out/'contacts.json',s.world.events)
    arrays={'time_s':np.array(times)}
    for name in states[0]:
        for field in states[0][name]:arrays[f'{name}__{field}']=np.array([st[name][field] for st in states])
    np.savez_compressed(out/'trajectories.npz',**arrays)
    return {'meta':m,'result':r,'path':out,'states':states}


def compare(records,a,overview):
    fam=records[0]['meta']['family'];out=a.out/fam
    build,_=FAMILIES[int(fam[:2])-1]
    scenes=[build(r['meta']['control']['value']) for r in records]
    readers=[imageio.get_reader(str(r['path']/'rgb.mp4')) for r in records]
    w=a.width*len(records);h=a.height+120;vid=writer(out/'comparison.mp4',a.fps);gifs=[];first=event=None
    try:
        for k in range(a.frames):
            canvas=Image.new('RGB',(w,h),(247,249,249));d=ImageDraw.Draw(canvas)
            d.text((16,7),records[0]['meta']['title'],font=font(24,True),fill=(31,57,65))
            d.text((w-420,14),'几何变体对照 / 同组固定相机与入射状态',font=font(16),fill=(91,114,121))
            for j,(reader,r,s) in enumerate(zip(readers,records,scenes)):
                im=Image.fromarray(reader.get_next_data())
                fr=framed_image(im,s,k/a.fps,k,a.context,r['result']['outcome'],(a.frames-1)/a.fps)
                canvas.paste(fr,(j*a.width,44))
                if j:ImageDraw.Draw(canvas).line((j*a.width,44,j*a.width,h),fill=(200,214,217),width=2)
            arr=np.asarray(canvas);vid.append_data(arr);overview.append_data(arr)
            if k==0:first=canvas.copy();canvas.save(out/'comparison_first.png')
            if k==min(40,a.frames-1):event=canvas.copy();canvas.save(out/'comparison_event.png')
            if k==a.frames-1:canvas.save(out/'comparison_last.png')
            if k%2==0:gifs.append(canvas.resize((1152,int(round(h*1152/w))),Image.Resampling.LANCZOS))
        gifs[0].save(out/'preview.gif',save_all=True,append_images=gifs[1:],duration=round(2000/a.fps),loop=0,optimize=False)
    finally:
        vid.close()
        for rd in readers:rd.close()
    return first,event


def write_index(all_records,a):
    cards=[]
    for rr in all_records:
        m=rr[0]['meta'];fam=m['family']
        rows=''.join(f"<tr><td>{r['meta']['control']['value']:g} {r['meta']['control']['units']}</td><td>{r['result']['outcome']}</td></tr>" for r in rr)
        cards.append(f'''<section><h2>{m['title']}</h2><p>{m['description']}</p>
        <video controls loop muted playsinline preload="metadata" poster="{fam}/comparison_first.png" src="{fam}/comparison.mp4"></video>
        <table><tr><th>{m['control']['name']}</th><th>本次仿真观察结果</th></tr>{rows}</table>
        <p class="note">模型：{m['model']}。{' '.join(m['notes'])}</p></section>''')
    html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>几何变体仿真原型</title><style>
    body{font-family:system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#edf2f3;color:#203941;line-height:1.65}
    main{max-width:1420px;margin:auto;padding:28px}h1{font-size:28px}h2{font-size:22px;margin:0 0 8px}
    section{background:white;padding:22px;margin:24px 0;border-radius:12px;border:1px solid #d9e3e5}
    video{display:block;width:100%;border-radius:6px;background:#e4eaea}table{border-collapse:collapse;margin:14px 0;width:100%;font-size:14px}
    td,th{padding:7px 12px;text-align:left;border-bottom:1px solid #dce6e8}.note{font-size:13px;color:#61757b}</style>
    <main><h1>场景几何变体：__FAMILY_COUNT__组数值仿真</h1>
    <p>本次共__CASE_COUNT__个变体，__FRAMES__帧、__FPS__fps；前__CONTEXT__帧标为CONTEXT。运动来自数值积分与碰撞求解，没有使用图像生成或预设轨迹。</p>
    <p><strong>范围：</strong>自编NumPy求解器的可视化原型：平面刚体与三维球/静态盒模型。不是正式benchmark的高精度3D GT。只调整标明的控制量，允许派生接触和运动发生变化。</p>
    <p><a href="all_scenes.mp4">连续观看本次场景（__SECONDS__秒）</a> · <a href="run_summary.json">运行记录</a></p>'''+''.join(cards)+'</main></html>'
    for token, value in {'__FAMILY_COUNT__':len(all_records),
            '__CASE_COUNT__':sum(len(rr) for rr in all_records),
            '__FRAMES__':a.frames,'__FPS__':a.fps,'__CONTEXT__':a.context,
            '__SECONDS__':f'{len(all_records)*a.frames/a.fps:g}'}.items():
        html=html.replace(token,str(value))
    (a.out/'index.html').write_text(html,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out',type=Path,default=ROOT/'results')
    p.add_argument('--families',nargs='+',default=['all'],help='all, or family IDs 1..6')
    p.add_argument('--list-families',action='store_true',help='Print the six scene families and exit')
    p.add_argument('--fps',type=int,default=30);p.add_argument('--frames',type=int,default=90)
    p.add_argument('--context',type=int,default=8);p.add_argument('--sim-hz',type=int,default=720)
    p.add_argument('--width',type=int,default=640);p.add_argument('--height',type=int,default=360)
    p.add_argument('--aa',type=int,default=2);p.add_argument('--save-frames',action='store_true')
    p.add_argument('--physics-only',action='store_true');a=p.parse_args()
    if a.list_families:
        for i,(build,values) in enumerate(FAMILIES,1):
            scene=build(values[0])
            print(f'{i}: {scene.title} | {scene.variable} | {values} {scene.units}')
        return
    if a.sim_hz<=0 or a.fps<=0 or a.sim_hz%a.fps:p.error('sim-hz must be positive and divisible by fps')
    if not 0<a.context<a.frames:p.error('Require 0<context<frames')
    if a.width<=0 or a.height<=0 or a.width%2 or a.height%2:
        p.error('Video dimensions must be positive and even')
    if a.aa<1:p.error('aa must be a positive integer')
    try:
        selected=list(range(len(FAMILIES))) if a.families==['all'] else [int(s)-1 for s in a.families]
    except ValueError:
        p.error('families must be all, or integer IDs 1..6')
    if any(i<0 or i>=len(FAMILIES) for i in selected):p.error(f'Family IDs must be 1..{len(FAMILIES)}')
    if len(selected)!=len(set(selected)):p.error('Duplicate family IDs are not allowed')
    a.out.mkdir(parents=True,exist_ok=True)
    versions={k:importlib.metadata.version(k) for k in ['numpy','Pillow','imageio','imageio-ffmpeg']};versions['python']=sys.version
    dump(a.out/'environment.json',versions)
    all_records=[];report=[];firsts=[];events=[];start=time.time()
    overview=None if a.physics_only else writer(a.out/'all_scenes.mp4',a.fps)
    try:
        for i in selected:
            build,values=FAMILIES[i];rr=[]
            for j,v in enumerate(values):
                r=run_case(build,v,j,a);rr.append(r)
                print(f"{r['meta']['family']} | {v:g}: {r['result']['outcome']} | penetration {r['result']['max_penetration_m']:.6f} m",flush=True)
            prefix={}
            for count in sorted({a.context,8,16}):
                if count>a.frames:continue
                diff={};common=set.intersection(*(set(r['states'][0]) for r in rr))
                for name in sorted(common):
                    arrays=[]
                    for r in rr:
                        positions=np.array([st[name]['position'] for st in r['states'][:count]])
                        arrays.append(positions)
                    diff[name]=max(float(np.max(np.linalg.norm(x-arrays[0],axis=1))) for x in arrays)
                prefix[str(count)]=diff
            report.append({'family':rr[0]['meta']['family'],'title':rr[0]['meta']['title'],
                'prefix_position_max_difference_m':prefix,'cases':[{'control':r['meta']['control'],**r['result']} for r in rr]})
            all_records.append(rr)
            if not a.physics_only:
                first,event=compare(rr,a,overview);firsts.append(first);events.append(event)
            dump(a.out/'run_summary.json',{'simulation_hz':a.sim_hz,'fps':a.fps,'frames':a.frames,
                'elapsed_wall_seconds':time.time()-start,'families':report})
    finally:
        if overview:overview.close()
    if not a.physics_only:
        for filename,ims in [('contact_sheet_initial.png',firsts),('contact_sheet_events.png',events)]:
            w=1536;h=round(ims[0].height*w/ims[0].width);sheet=Image.new('RGB',(w,h*len(ims)),(247,249,249))
            for i,im in enumerate(ims):sheet.paste(im.resize((w,h),Image.Resampling.LANCZOS),(0,i*h))
            sheet.save(a.out/filename)
        write_index(all_records,a)
    print(f'Completed {sum(len(rr) for rr in all_records)} cases in {time.time()-start:.1f}s: {a.out}',flush=True)

if __name__=='__main__':
    warnings.filterwarnings('ignore',category=DeprecationWarning)
    main()
