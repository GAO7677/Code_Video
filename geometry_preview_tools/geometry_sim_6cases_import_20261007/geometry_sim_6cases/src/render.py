"""Deterministic CPU orthographic renderer for the integrated physics states.
No generative-image model is used. Mesh positions follow the solver directly.
The renderer uses an orthographic per-pixel depth buffer; it is not photorealistic.
"""
from __future__ import annotations
import math
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from physics import rotation, SphereWorld3
from scenes import TARGET, OBSTACLE, PLATFORM, SUPPORT, FLOOR


@lru_cache(maxsize=24)
def font(size:int,bold=False):
    candidates=[
        os.environ.get('GEOM_DEMO_FONT',''),
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
        'C:/Windows/Fonts/msyh.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for fn in candidates:
        if fn and Path(fn).exists():
            try:return ImageFont.truetype(fn,size)
            except OSError:pass
    return ImageFont.load_default(size=size)


def shade(color,n):
    light=np.array([-.4,-.6,1.]);light/=np.linalg.norm(light)
    n=np.array(n,dtype=float);n/=max(1e-12,float(np.linalg.norm(n)))
    k=.66+.32*max(0.,float(n@light))
    return tuple(int(np.clip(c*k,0,255)) for c in color)


def box_mesh(center,half,color):
    c=np.array(center);h=np.array(half)
    vs=np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                 [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]])*h+c
    ids=[[0,3,2,1],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return [(vs[ii],color) for ii in ids]


def prism_side(vertices,depth,color):
    """CCW polygon in x-z, extruded in y."""
    v=np.array(vertices);n=len(v);d=depth/2
    front=np.column_stack([v[:,0],np.full(n,-d),v[:,1]])
    back=np.column_stack([v[:,0],np.full(n,d),v[:,1]])
    faces=[(front,color),(back[::-1],color)]
    for i in range(n):
        j=(i+1)%n
        faces.append((np.array([front[i],back[i],back[j],front[j]]),color))
    return faces


def prism_top(vertices,z0,height,color):
    v=np.array(vertices);n=len(v)
    bot=np.column_stack([v,np.full(n,z0)])
    top=np.column_stack([v,np.full(n,z0+height)])
    faces=[(bot[::-1],color),(top,color)]
    for i in range(n):
        j=(i+1)%n
        faces.append((np.array([bot[i],bot[j],top[j],top[i]]),color))
    return faces


def cylinder_mesh(center,r,height,color,n=36):
    x,y,z=center
    angles=np.arange(n)*2*np.pi/n
    v=np.column_stack([x+r*np.cos(angles),y+r*np.sin(angles)])
    return prism_top(v,z,height,color)


@lru_cache(maxsize=64)
def sphere_sprite(radius_px:int,color:tuple):
    # Analytic shaded sphere sprite; geometry comes from the numeric simulation.
    r=max(2,radius_px);d=2*r+3
    yy,xx=np.mgrid[0:d,0:d]
    x=(xx-(d-1)/2)/r;y=-(yy-(d-1)/2)/r
    rho=x*x+y*y;inside=rho<=1.
    z=np.sqrt(np.maximum(0.,1-rho))
    light=np.array([-.45,.62,.85]);light/=np.linalg.norm(light)
    lam=np.maximum(0.,x*light[0]+y*light[1]+z*light[2])
    half=np.array([light[0],light[1],light[2]+1.]);half/=np.linalg.norm(half)
    spec=np.maximum(0.,x*half[0]+y*half[1]+z*half[2])**32
    intensity=.46+.53*lam
    rgb=np.array(color)[None,None,:]*intensity[:,:,None]+105*spec[:,:,None]
    rgba=np.dstack([np.clip(rgb,0,255).astype(np.uint8),(inside*255).astype(np.uint8)])
    return Image.fromarray(rgba)


class Renderer:
    def __init__(self,scene,width=640,height=360,aa=2):
        self.scene=scene;self.ow=width;self.oh=height;self.aa=aa
        self.w=width*aa;self.h=height*aa
        c=scene.camera
        az=math.radians(c['azimuth']);el=math.radians(c['elevation'])
        self.toward=np.array([math.cos(el)*math.cos(az),math.cos(el)*math.sin(az),math.sin(el)])
        self.right=np.array([-math.sin(az),math.cos(az),0.])
        self.up=np.cross(self.toward,self.right)
        self.target=np.array(c['target'],dtype=float)
        self.scale=self.w/c['span']
        self.static_meshes=None

    def project(self,vs):
        pts=np.asarray(vs)-self.target
        return np.column_stack([self.w/2+(pts@self.right)*self.scale,
                                self.h*.52-(pts@self.up)*self.scale])

    def primitives(self):
        s=self.scene;meshes=[];spheres=[]
        if isinstance(s.world,SphereWorld3):
            for b in s.world.boxes:
                if b.name=='floor':continue
                meshes+=box_mesh(b.center,b.half,b.render.get('color',PLATFORM))
            spheres.append((s.world.p.copy(),s.world.r,TARGET))
        else:
            if s.plane=='planar_xy':
                meshes+=box_mesh([.5,0.,.68],[4.9,3.5,.16],PLATFORM)
            for b in s.world.bodies:
                rd=b.render
                if rd.get('hidden') or b.name=='floor':continue
                color=rd.get('color',PLATFORM)
                if b.shape=='circle':
                    if s.plane=='planar_xy':
                        if not b.dynamic:
                            meshes+=cylinder_mesh([b.p[0],b.p[1],.84],b.radius,rd.get('height',.4),color)
                        else:
                            spheres.append((np.array([b.p[0],b.p[1],.84+b.radius]),b.radius,color))
                    else:spheres.append((np.array([b.p[0],0.,b.p[1]]),b.radius,color))
                elif s.plane=='planar_xy':
                    meshes+=prism_top(b._wv,.84,rd.get('height',.35),color)
                else:
                    meshes+=prism_side(b._wv,rd.get('depth',1.1),color)
        for e in s.draw_extras:
            if e['kind']=='profile':
                p=np.array(e['profile']);bottom=e.get('bottom',0.)
                # Convexity isn't required for rendering. One contour, accurate visible shape.
                v=np.vstack([[p[0,0],bottom],[p[-1,0],bottom],p[::-1]])
                meshes+=prism_side(v,e['depth'],e['color'])
            elif e['kind']=='pivot':
                meshes+=cylinder_mesh(e['pos'],e['radius'],e['height'],(70,80,90),24)
                top=np.array(e['pos'])+np.array([0.,0.,e['height']])
                spheres.append((top,.07,(210,219,221)))
            elif e['kind']=='marker':
                meshes+=cylinder_mesh(e['pos'],e['radius'],.014,(55,67,72),20)
        return meshes,spheres

    def _face(self, rgb, zbuf, pts, color):
        pts=np.asarray(pts,dtype=float)
        normal=np.cross(pts[1]-pts[0],pts[2]-pts[0])
        denom=float(normal@self.toward)
        if denom<=1e-10:return
        q=self.project(pts)
        xmin=max(0,int(math.floor(q[:,0].min())))
        xmax=min(self.w-1,int(math.ceil(q[:,0].max())))
        ymin=max(0,int(math.floor(q[:,1].min())))
        ymax=min(self.h-1,int(math.ceil(q[:,1].max())))
        if xmin>xmax or ymin>ymax:return
        mask=Image.new('L',(xmax-xmin+1,ymax-ymin+1),0)
        ImageDraw.Draw(mask).polygon([tuple(p-[xmin,ymin]) for p in q],fill=255)
        mask=np.asarray(mask)>0
        xs=(np.arange(xmin,xmax+1)-self.w/2)/self.scale+float(self.target@self.right)
        ys=(self.h*.52-np.arange(ymin,ymax+1))/self.scale+float(self.target@self.up)
        depth=(float(normal@pts[0])-float(normal@self.right)*xs[None,:]
               -float(normal@self.up)*ys[:,None])/denom
        zslice=zbuf[ymin:ymax+1,xmin:xmax+1]
        keep=mask&(depth>zslice)
        rgb[ymin:ymax+1,xmin:xmax+1][keep]=shade(color,normal)
        zslice[keep]=depth[keep]

    def _sphere(self,rgb,zbuf,pos,r,color):
        q=self.project([pos])[0]
        rp=max(2,int(round(r*self.scale)))
        spr=np.asarray(sphere_sprite(rp,tuple(color)))
        hh,ww=spr.shape[:2]
        x=int(q[0]-ww/2);y=int(q[1]-hh/2)
        x0=max(0,x);y0=max(0,y);x1=min(self.w,x+ww);y1=min(self.h,y+hh)
        if x0>=x1 or y0>=y1:return
        yy,xx=np.mgrid[y0:y1,x0:x1]
        dx=(xx-q[0])/self.scale;dy=(yy-q[1])/self.scale
        rho=dx*dx+dy*dy
        depth=float(pos@self.toward)+np.sqrt(np.maximum(0.,r*r-rho))
        sprite=spr[y0-y:y1-y,x0-x:x1-x]
        zs=zbuf[y0:y1,x0:x1]
        keep=(sprite[:,:,3]>0)&(rho<=r*r)&(depth>zs)
        rgb[y0:y1,x0:x1][keep]=sprite[:,:,:3][keep]
        zs[keep]=depth[keep]

    def image(self):
        im=Image.new('RGB',(self.w,self.h),(224,230,229))
        draw=ImageDraw.Draw(im)
        for x in np.arange(-10,11,1):
            q=self.project([[x,-9,.003],[x,9,.003]])
            draw.line([tuple(p) for p in q],fill=(211,220,219),width=max(1,self.aa))
        for y in np.arange(-9,10,1):
            q=self.project([[-10,y,.003],[10,y,.003]])
            draw.line([tuple(p) for p in q],fill=(211,220,219),width=max(1,self.aa))
        rgb=np.asarray(im).copy()
        # Initialize depth with the z=0 ground plane. Background grid is on it.
        xs=(np.arange(self.w)-self.w/2)/self.scale+float(self.target@self.right)
        ys=(self.h*.52-np.arange(self.h))/self.scale+float(self.target@self.up)
        zbuf=np.broadcast_to((-self.up[2]*ys[:,None]-self.right[2]*xs[None,:])/self.toward[2],
                             (self.h,self.w)).copy()
        meshes,spheres=self.primitives()
        for pts,color in meshes:self._face(rgb,zbuf,pts,color)
        for pos,r,color in spheres:self._sphere(rgb,zbuf,pos,r,color)
        im=Image.fromarray(rgb)
        if self.aa>1:im=im.resize((self.ow,self.oh),Image.Resampling.LANCZOS)
        return im


def framed_image(rgb,scene,t,frame,context=8,outcome=None,total=89/30):
    """Review UI is outside the scene. Per-case clean RGB is also exported."""
    w,h=rgb.size
    canvas=Image.new('RGB',(w,h+70),(249,250,250))
    canvas.paste(rgb,(0,38));d=ImageDraw.Draw(canvas)
    d.text((12,8),f'{scene.variable} = {scene.value:g} {scene.units}',font=font(17,True),fill=(31,52,61))
    phase='CONTEXT' if frame<context else 'FUTURE'
    d.text((w-188,11),f'{phase}   t={t:.2f}s',font=font(13),fill=(67,96,105))
    if outcome:
        d.text((12,h+43),'整段结果：'+outcome,font=font(15),fill=(42,69,76))
    else:
        d.text((12,h+43),'数值仿真原型 / 非高精度GT',font=font(13),fill=(93,113,119))
    x=int((w-24)*min(t/max(total,1e-12),1.))
    d.rectangle((12,h+66,w-12,h+68),fill=(218,226,228))
    d.rectangle((12,h+66,12+x,h+68),fill=(71,116,129))
    return canvas
