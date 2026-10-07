"""Six geometry families. The bridge-support and hinge-position scenes are excluded. Only 'value' changes within each family.
Dimensions in metres, time in seconds, mass in kg. Physics definitions, NOT
outcome-conditioned animations. Target events are evaluated after simulation.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np
from physics import World2, SphereWorld3, Body2, StaticBox3, box2, circle2

TARGET=(222,91,48)
OBSTACLE=(53,121,143)
PLATFORM=(173,190,191)
SUPPORT=(96,116,128)
SECOND=(214,158,65)
FLOOR=(219,225,224)

@dataclass
class Scene:
    key:str
    title:str
    variable:str
    value:float
    units:str
    world:object
    plane:str
    target:str='ball'
    draw_extras:list=field(default_factory=list)
    camera:dict=field(default_factory=dict)
    notes:list=field(default_factory=list)
    description:str=''

    def metadata(self):
        return {'family':self.key,'title':self.title,'control':{
            'name':self.variable,'value':self.value,'units':self.units},
            'model':self.plane,'camera':self.camera,'notes':self.notes,
            'description':self.description}


def vis(color, depth=.9, **kw):return dict(color=color,depth=depth,**kw)

def side_ground(w):
    w.add(box2('floor',(0.,-.13),(8.,.13),friction=.5,restitution=.05,
               render=vis(FLOOR,3.3)))

def add_side_box(w,name,x0,x1,z0,z1,color=PLATFORM,**kwargs):
    return w.add(box2(name,((x0+x1)/2,(z0+z1)/2),((x1-x0)/2,(z1-z0)/2),
                      render=vis(color,1.1),**kwargs))


def build_cylinder(value):
    w=World2((0.,0.))
    r=.18
    w.add(circle2('ball',(-1.35,0.),r,mass=1.,v=np.array([1.7,0.]),
        restitution=.8,friction=.06,render=vis(TARGET,kind='sphere')))
    w.add(circle2('cylinder',(.35,value),.28,mass=0.,restitution=.8,
        friction=.06,render=vis(OBSTACLE,height=.50,kind='cylinder')))
    return Scene('01_cylinder_offset','圆柱障碍横向偏移','cylinder_y',value,'m',w,'planar_xy',
        camera={'target':[.3,-.40,.75],'azimuth':-68.,'elevation':48.,'span':8.0},
        description='固定目标运动和圆柱尺寸，只改变圆柱横向位置。',
        notes=['水平运动按二维刚体接触求解；不模拟竖直运动和台面滚动阻力。'])


def build_barrier(value):
    w=World2((0.,0.))
    w.add(circle2('ball',(-1.35,0.),.18,mass=1.,v=np.array([1.7,0.]),
        restitution=.8,friction=.06,render=vis(TARGET,kind='sphere')))
    angle=math.radians(45.)
    anchor=np.array([-.20,-.75]);u=np.array([math.cos(angle),math.sin(angle)])
    w.add(box2('barrier',anchor+value/2*u,(value/2,.055),angle=angle,
        restitution=.8,friction=.06,render=vis(OBSTACLE,height=.43)))
    return Scene('02_barrier_length','有限挡板长度','barrier_length',value,'m',w,'planar_xy',
        camera={'target':[.7,.60,.8],'azimuth':-68.,'elevation':48.,'span':7.7},
        description='固定挡板端点与方向，只延长另一个端点。',
        draw_extras=[{'kind':'marker','pos':[anchor[0],anchor[1],.86],'radius':.06}],
        notes=['水平二维接触模型；固定端点用深色小圆点标记。'])


def build_rails(value):
    w=SphereWorld3((-1.25,0.,1.15),.20,(1.55,0.,0.),friction=.20)
    w.add(StaticBox3('floor',[.5,0.,-.1],[6.,3.,.1],render=vis(FLOOR)))
    w.add(StaticBox3('entry',[-1.25,0.,.475],[1.25,.80,.475],
        friction=.2,render=vis(PLATFORM)))
    for sign,name in [(-1,'rail_front'),(1,'rail_back')]:
        w.add(StaticBox3(name,[2.25,sign*(value/2+.09),.87],[2.25,.09,.08],
            friction=.20,render=vis(OBSTACLE)))
    return Scene('03_rail_gap','双轨净间距','rail_gap',value,'m',w,'sphere_boxes_3d',
        camera={'target':[.9,0.,.68],'azimuth':-66.,'elevation':35.,'span':6.7},
        description='固定小球与入口，只改变两条承重轨道之间的净间距。',
        notes=['球与静态盒进行三维接触求解，包含球自旋及接触摩擦。',
               '轨道在原型中视为刚性固定结构，不模拟轨道弯曲。'])


def build_hump(value):
    w=World2((0.,-9.81))
    side_ground(w)
    h0=.70;L=1.80
    add_side_box(w,'entry',-3.,0.,0.,h0,friction=.45,restitution=.01)
    add_side_box(w,'exit',L,6.,0.,h0,friction=.45,restitution=.01)
    n=48
    xs=np.linspace(0,L,n+1)
    zs=h0+value*np.sin(np.pi*xs/L)**2
    for i in range(n):
        # CCW convex trapezoid; all are static parts of the same solid track.
        v=np.array([[xs[i],0.],[xs[i+1],0.],[xs[i+1],zs[i+1]],[xs[i],zs[i]]])
        w.add(Body2(f'track_{i:02d}',np.zeros(2),'poly',vertices=v,
            friction=.45,restitution=.01,render={'hidden':True}))
    ball=w.add(circle2('ball',(-1.25,h0+.16),.16,mass=1.,v=np.array([2.2,0.]),
        omega=-2.2/.16,friction=.45,restitution=.01,render=vis(TARGET,kind='sphere')))
    profile=np.column_stack([xs,zs])
    return Scene('04_hump_height','坡峰高度','hump_height',value,'m',w,'planar_xz',
        camera={'target':[.95,0.,.70],'azimuth':-80.,'elevation':22.,'span':8.5},
        draw_extras=[{'kind':'profile','profile':profile.tolist(),'bottom':0.,'depth':1.1,'color':OBSTACLE}],
        description='固定入口运动、入口/出口高度与跨度，只改变平滑坡峰高度。',
        notes=['竖直平面刚体接触；采用实心球转动惯量，不能表现横向偏离。',
               '曲面用48段凸几何离散；高度变化联动坡度与曲率。'])


def build_hole(value):
    w=SphereWorld3((-1.25,0.,1.13),.18,(1.55,0.,0.),friction=.20)
    w.add(StaticBox3('floor',[.5,0.,-.1],[6.,3.,.1],render=vis(FLOOR)))
    # Four disjoint solid boxes make the hole. The empty area has no collider.
    xlo,xhi=.10,1.05
    ylo,yhi=value-.34,value+.34
    def bx(name,x0,x1,y0,y1):
        w.add(StaticBox3(name,[(x0+x1)/2,(y0+y1)/2,.475],
            [(x1-x0)/2,(y1-y0)/2,.475],friction=.20,render=vis(PLATFORM)))
    bx('left',-2.5,xlo,-1.35,1.35);bx('right',xhi,4.0,-1.35,1.35)
    bx('front',xlo,xhi,-1.35,ylo);bx('back',xlo,xhi,yhi,1.35)
    return Scene('05_hole_offset','台面孔洞横向位置','hole_y',value,'m',w,'sphere_boxes_3d',
        camera={'target':[.7,0.,.65],'azimuth':-66.,'elevation':48.,'span':6.7},
        description='固定孔洞尺寸与小球运动，只横向平移缺失的支撑区域。',
        notes=['球与静态盒的三维接触；台面是厚实基座，孔洞贯通至底板。'])


def build_edge(value):
    w=World2((0.,-9.81),iterations=14)
    side_ground(w)
    add_side_box(w,'table',-2.7,value,0.,.95,friction=.45,restitution=.03)
    w.add(box2('block',(.45,1.12),(.16,.17),mass=.5,
        friction=.40,restitution=.22,render=vis(SECOND,.38)))
    w.add(circle2('ball',(-1.35,1.12),.17,mass=.8,v=np.array([2.,0.]),omega=-2/.17,
        friction=.15,restitution=.22,render=vis(TARGET,kind='sphere')))
    return Scene('06_table_edge','被撞方块与桌缘','edge_x',value,'m',w,'planar_xz',
        camera={'target':[.15,0.,.65],'azimuth':-80.,'elevation':24.,'span':5.8},
        description='固定相同撞击，只改变方块前方还剩多少台面支撑。',
        notes=['竖直平面接触；方块允许平移和绕平面法向转动。',
               '所有变体起始时方块均完整放在桌面上。'])



FAMILIES=[
    (build_cylinder,[.65,0.,.30]),
    (build_barrier,[.65,.88,1.75]),
    (build_rails,[.14,.30,.56]),
    (build_hump,[.10,.23,.52]),
    (build_hole,[0.,.40,.85]),
    (build_edge,[.65,1.05,1.90]),
]


def summarize(scene:Scene,states:list[dict],times:list[float])->dict:
    """Post-hoc observation only. These rules are never used by the solver."""
    w=scene.world
    pos=np.array([s[scene.target]['position'] for s in states])
    vel=np.array([s[scene.target]['velocity'] for s in states])
    final=pos[-1]; key=scene.key
    result={'target_final_position':final.tolist(),'max_penetration_m':w.max_penetration,
            'contacts':w.events,'physics_steps_duration_s':w.t,
            'description':'Outcome labels are post-hoc coarse diagnostics, not a validated evaluator.'}
    if key.startswith('01'):
        hit=any('cylinder' in k for k in w.events)
        outcome='未接触，继续前进' if not hit else ('正碰后反弹' if abs(scene.value)<1e-5 else '偏心碰撞，改变方向')
        result['hit_obstacle']=hit
    elif key.startswith('02'):
        hit=any('barrier' in k for k in w.events)
        outcome='从端点外侧通过' if not hit else ('端点附近接触' if scene.value<1.3 else '撞击挡板后转向')
        result['hit_obstacle']=hit
    elif key.startswith('03'):
        low=float(pos[:,2].min())<.65
        outcome='从双轨之间落下' if low else '双轨持续承托'
        result['fell_below_rails']=low
    elif key.startswith('04'):
        passed=bool(np.any(pos[:,0]>1.95))
        reversed_=bool(np.any((pos[:,0]>0)&(vel[:,0]<-.15)))
        outcome='越过坡峰' if passed else ('爬升后回落' if reversed_ else '仍在爬升/接近转折')
        result.update(passed_hump=passed,reversed_on_hump=reversed_)
    elif key.startswith('05'):
        low=float(pos[:,2].min())<.65
        outcome='失去支撑，落入孔洞' if low else ('孔沿接触后继续' if abs(float(final[1]))>.03 else '避开孔洞，继续前进')
        result['fell_into_hole']=low
    elif key.startswith('06'):
        bp=np.array([s['block']['position'] for s in states])
        ba=np.array([s['block']['angle'] for s in states])
        fallen=bool(np.any(bp[:,1]<.80))
        outcome='被撞方块翻落台面' if fallen else '方块仍留在台面'
        result.update(block_fell=fallen,block_final_position=bp[-1].tolist(),block_max_abs_angle_rad=float(np.max(np.abs(ba))))
    else:
        raise ValueError(f'Unknown family: {key}')
    result['outcome']=outcome
    return result
