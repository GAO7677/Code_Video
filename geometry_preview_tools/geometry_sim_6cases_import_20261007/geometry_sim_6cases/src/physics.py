"""Small, explicit physics solvers used ONLY for geometry-demo prototyping.

Planar solver: circles/convex polygons, semi-implicit Euler, sequential normal
and Coulomb-friction impulses, penetration correction, optional ideal hinge.
3-D solver: a sphere and static axis-aligned boxes, contact impulses and spin.
This is not Bullet/Box2D and is not a validated benchmark-ground-truth engine.
All output positions are integrated; outcome labels do not control motion.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math
import numpy as np

Array = np.ndarray


def cross2(a: Array, b: Array) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def perp(a: Array) -> Array:
    return np.array([-a[1], a[0]], dtype=float)


def rotation(a: float) -> Array:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def unit(a: Array) -> Array:
    n = float(np.linalg.norm(a))
    return a / n if n > 1e-12 else np.array([1., 0.])


@dataclass
class Body2:
    name: str
    p: Array
    shape: str
    mass: float = 0.
    radius: float = 0.
    vertices: Array | None = None
    inertia: float | None = None
    v: Array = field(default_factory=lambda: np.zeros(2))
    angle: float = 0.
    omega: float = 0.
    friction: float = .25
    restitution: float = .15
    pinned: bool = False
    render: dict = field(default_factory=dict)
    enabled: bool = True
    gravity_scale: float = 1.
    damping: float = 0.

    def __post_init__(self):
        self.p = np.array(self.p, dtype=float)
        self.v = np.array(self.v, dtype=float)
        if self.vertices is not None:
            self.vertices = np.array(self.vertices, dtype=float)
        if self.inertia is None:
            if self.shape == 'circle':
                self.inertia = .4 * self.mass * self.radius ** 2
            elif self.mass > 0:
                # Polygon about the declared reference point; vertices CCW.
                cr = np.cross(self.vertices, np.roll(self.vertices, -1, axis=0))
                vs = self.vertices
                vn = np.roll(vs, -1, axis=0)
                self.inertia = self.mass * float(np.sum(cr * (
                    np.sum(vs*vs,axis=1) + np.sum(vs*vn,axis=1) + np.sum(vn*vn,axis=1)
                ))) / (6 * float(np.sum(cr)))
            else:
                self.inertia = 0.
        self.im = 1. / self.mass if self.mass > 0 and not self.pinned else 0.
        self.ii = 1. / self.inertia if self.mass > 0 and self.inertia > 0 else 0.
        self._wv = None
        self.refresh()

    @property
    def dynamic(self) -> bool:
        return self.im > 0 or self.ii > 0

    def refresh(self):
        if self.shape == 'poly':
            self._wv = self.vertices @ rotation(self.angle).T + self.p
            self.lo = self._wv.min(axis=0)
            self.hi = self._wv.max(axis=0)
            e = np.roll(self._wv, -1, axis=0) - self._wv
            self._normals = np.column_stack([e[:,1], -e[:,0]])
            self._normals /= np.linalg.norm(self._normals, axis=1)[:,None]
        else:
            self.lo = self.p-self.radius
            self.hi = self.p+self.radius

    def velocity_at(self, q: Array) -> Array:
        return self.v + self.omega*perp(q-self.p)

    def impulse(self, j: Array, q: Array):
        self.v += self.im*j
        self.omega += self.ii*cross2(q-self.p, j)

    def state(self) -> dict:
        return {'position': self.p.tolist(), 'velocity': self.v.tolist(),
                'angle': float(self.angle), 'angular_velocity': float(self.omega)}


def box2(name, p, half, mass=0., angle=0., offset=(0.,0.), **kwargs) -> Body2:
    x,y = half
    v = np.array([[-x,-y],[x,-y],[x,y],[-x,y]],dtype=float) + np.array(offset)
    return Body2(name=name, p=p, shape='poly', vertices=v,
                 mass=mass, angle=angle, **kwargs)


def circle2(name, p, radius, mass=1., **kwargs) -> Body2:
    return Body2(name=name, p=p, shape='circle', radius=radius, mass=mass, **kwargs)


@dataclass
class Contact:
    a: Body2
    b: Body2
    q: Array
    n: Array  # A -> B
    penetration: float
    normal_impulse: float = 0.
    tangent_impulse: float = 0.
    bounce: float = 0.
    def prepare(self):
        vn = float((self.b.velocity_at(self.q)-self.a.velocity_at(self.q))@self.n)
        e = min(self.a.restitution,self.b.restitution)
        self.bounce = -e*vn if vn < -.35 else 0.


def circle_poly(c: Body2, b: Body2) -> list[Contact]:
    v = b._wv
    distances = np.sum(b._normals*(c.p-v),axis=1)
    max_i = int(np.argmax(distances))
    if distances[max_i] > c.radius:
        return []
    if distances[max_i] <= 0:
        n = b._normals[max_i]
        q = c.p - distances[max_i]*n
        return [Contact(c,b,q,-n,float(c.radius-distances[max_i]))]
    edges = np.roll(v,-1,axis=0)-v
    t = np.sum((c.p-v)*edges,axis=1)/np.sum(edges*edges,axis=1)
    q_all = v + np.clip(t,0.,1.)[:,None]*edges
    delta = c.p-q_all
    d2 = np.sum(delta*delta,axis=1)
    i = int(np.argmin(d2))
    if d2[i] > c.radius*c.radius:
        return []
    d = math.sqrt(max(float(d2[i]),1e-20))
    return [Contact(c,b,q_all[i],-delta[i]/d,c.radius-d)]


def clip_poly(poly: list[Array], a: Array, b: Array) -> list[Array]:
    if not poly:
        return []
    out=[]
    edge=b-a
    prev=poly[-1]
    sp=cross2(edge,prev-a)
    for cur in poly:
        sc=cross2(edge,cur-a)
        if sc >= -1e-9:
            if sp < -1e-9:
                out.append(prev+(cur-prev)*(sp/(sp-sc)))
            out.append(cur)
        elif sp >= -1e-9:
            out.append(prev+(cur-prev)*(sp/(sp-sc)))
        prev,sp=cur,sc
    return out


def poly_poly(a: Body2,b: Body2) -> list[Contact]:
    av,bv=a._wv,b._wv
    axes=np.concatenate([a._normals,b._normals])
    pa=av@axes.T
    pb=bv@axes.T
    amin,amax=pa.min(axis=0),pa.max(axis=0)
    bmin,bmax=pb.min(axis=0),pb.max(axis=0)
    overlap=np.minimum(amax,bmax)-np.maximum(amin,bmin)
    if np.any(overlap < 0):
        return []
    i=int(np.argmin(overlap)); dep=float(overlap[i]); n=axes[i].copy()
    if float((bv.mean(axis=0)-av.mean(axis=0))@n)<0: n=-n
    intersection=list(av)
    for j in range(len(bv)):
        intersection=clip_poly(intersection,bv[j],bv[(j+1)%len(bv)])
        if not intersection: return []
    pts=np.array(intersection)
    tangent=perp(n); projections=pts@tangent
    ids=[int(np.argmin(projections)),int(np.argmax(projections))]
    # Contacts on the middle of the overlapping normal interval.
    plane=.5*(float(np.max(av@n))+float(np.min(bv@n)))
    qs=[pts[k]+(plane-float(pts[k]@n))*n for k in ids]
    if np.linalg.norm(qs[0]-qs[1])<1e-6: qs=qs[:1]
    return [Contact(a,b,q,n,dep) for q in qs]


def collide(a: Body2,b: Body2) -> list[Contact]:
    if not a.enabled or not b.enabled: return []
    if np.any(a.hi<b.lo) or np.any(b.hi<a.lo): return []
    if a.shape=='circle' and b.shape=='circle':
        d=b.p-a.p; dist=float(np.linalg.norm(d)); r=a.radius+b.radius
        if dist>=r: return []
        n=d/dist if dist>1e-10 else np.array([1.,0.])
        q=a.p+n*(a.radius-.5*(r-dist))
        return [Contact(a,b,q,n,r-dist)]
    if a.shape=='circle': return circle_poly(a,b)
    if b.shape=='circle': return circle_poly(b,a)
    return poly_poly(a,b)


class World2:
    def __init__(self, gravity=(0.,-9.81), iterations=10):
        self.gravity=np.array(gravity,dtype=float)
        self.bodies:list[Body2]=[]
        self.iterations=iterations
        self.t=0.
        self.events={}
        self.max_penetration=0.
        self.max_penetration_by_pair={}

    def add(self,b:Body2)->Body2:
        self.bodies.append(b); return b

    def step(self,dt:float):
        dyn=[b for b in self.bodies if b.dynamic and b.enabled]
        for b in dyn:
            if b.im:
                b.v += self.gravity*b.gravity_scale*dt
                b.v *= math.exp(-b.damping*dt)
                b.p += b.v*dt
            b.angle += b.omega*dt
            b.refresh()
        contacts=[]
        for i,a in enumerate(self.bodies):
            for b in self.bodies[i+1:]:
                if not(a.dynamic or b.dynamic): continue
                contacts.extend(collide(a,b))
        for c in contacts:
            c.prepare()
            key='|'.join(sorted((c.a.name,c.b.name)))
            self.max_penetration=max(self.max_penetration,c.penetration)
            self.max_penetration_by_pair[key]=max(self.max_penetration_by_pair.get(key,0.),c.penetration)
        # Sequential impulses. No prescribed collision outcome or trajectory.
        for _ in range(self.iterations):
            for c in contacts:
                a,b,n,q=c.a,c.b,c.n,c.q
                ra,rb=q-a.p,q-b.p
                ca,cb=cross2(ra,n),cross2(rb,n)
                kn=a.im+b.im+a.ii*ca*ca+b.ii*cb*cb
                if kn<1e-12:continue
                rel=b.velocity_at(q)-a.velocity_at(q)
                j=-(float(rel@n)-c.bounce)/kn
                old=c.normal_impulse
                c.normal_impulse=max(0.,old+j)
                j=c.normal_impulse-old
                a.impulse(-j*n,q);b.impulse(j*n,q)
                tangent=perp(n)
                ca,cb=cross2(ra,tangent),cross2(rb,tangent)
                kt=a.im+b.im+a.ii*ca*ca+b.ii*cb*cb
                if kt<1e-12:continue
                rel=b.velocity_at(q)-a.velocity_at(q)
                jt=-float(rel@tangent)/kt
                mu=math.sqrt(a.friction*b.friction)
                old=c.tangent_impulse
                c.tangent_impulse=float(np.clip(old+jt,-mu*c.normal_impulse,mu*c.normal_impulse))
                jt=c.tangent_impulse-old
                a.impulse(-jt*tangent,q);b.impulse(jt*tangent,q)
        self.t += dt
        for c in contacts:
            key='|'.join(sorted((c.a.name,c.b.name)))
            if c.normal_impulse>1e-7:
                if key not in self.events:
                    self.events[key]={'first_time_s':self.t,'max_normal_impulse':0.,'contact_steps':0}
                event=self.events[key]
                event['max_normal_impulse']=max(event['max_normal_impulse'],c.normal_impulse)
                event['contact_steps']+=1
            # Positional correction is deliberately conservative at high substep rate.
            correction=max(c.penetration-.00015,0.)*.32
            if correction <=0: continue
            a,b,n,q=c.a,c.b,c.n,c.q
            ca,cb=cross2(q-a.p,n),cross2(q-b.p,n)
            den=a.im+b.im+a.ii*ca*ca+b.ii*cb*cb
            if den<1e-12:continue
            lam=correction/den
            a.p-=a.im*lam*n;b.p+=b.im*lam*n
            a.angle-=a.ii*lam*ca;b.angle+=b.ii*lam*cb
        for b in dyn:b.refresh()

    def states(self):
        return {b.name:b.state() for b in self.bodies if b.dynamic}


@dataclass
class StaticBox3:
    name:str
    center:Array
    half:Array
    friction:float=.25
    restitution:float=.1
    render:dict=field(default_factory=dict)
    def __post_init__(self):
        self.center=np.array(self.center,dtype=float)
        self.half=np.array(self.half,dtype=float)


class SphereWorld3:
    """True 3-D sphere/axis-aligned-box contacts, static environment only."""
    def __init__(self,p,r,v,mass=1.,friction=.25,restitution=.1):
        self.p=np.array(p,dtype=float);self.v=np.array(v,dtype=float)
        self.r=r;self.mass=mass;self.omega=np.array([0.,float(v[0])/r,0.])
        self.inertia=.4*mass*r*r
        self.friction=friction;self.restitution=restitution
        self.boxes:list[StaticBox3]=[]
        self.t=0.;self.events={};self.max_penetration=0.
        self.max_penetration_by_pair={}

    def add(self,b):self.boxes.append(b);return b

    def step(self,dt):
        self.v[2]-=9.81*dt
        self.p+=self.v*dt
        for _ in range(3):
            for b in self.boxes:
                lo,hi=b.center-b.half,b.center+b.half
                nearest=np.clip(self.p,lo,hi)
                delta=self.p-nearest;d2=float(delta@delta)
                if d2>=self.r*self.r:continue
                if d2>1e-20:
                    d=math.sqrt(d2);n=delta/d;penetration=self.r-d
                else:
                    distances=np.concatenate([self.p-lo,hi-self.p])
                    i=int(np.argmin(distances));n=np.zeros(3)
                    n[i%3]=-1. if i<3 else 1.
                    penetration=self.r+float(distances[i])
                self.max_penetration=max(self.max_penetration,penetration)
                key='ball|'+b.name
                self.max_penetration_by_pair[key]=max(self.max_penetration_by_pair.get(key,0.),penetration)
                rvec=-n*self.r
                contact_v=self.v+np.cross(self.omega,rvec)
                vn=float(contact_v@n)
                j=0.
                if vn<0:
                    e=min(self.restitution,b.restitution) if vn<-.35 else 0.
                    j=-(1.+e)*vn*self.mass
                    self.v+=j*n/self.mass
                    contact_v=self.v+np.cross(self.omega,rvec)
                    tangent=contact_v-float(contact_v@n)*n
                    speed=float(np.linalg.norm(tangent))
                    if speed>1e-12:
                        tangent/=speed
                        jt=min(speed/(1/self.mass+self.r*self.r/self.inertia),math.sqrt(self.friction*b.friction)*j)
                        jtvec=-jt*tangent
                        self.v+=jtvec/self.mass
                        self.omega+=np.cross(rvec,jtvec)/self.inertia
                    if key not in self.events:
                        self.events[key]={'first_time_s':self.t+dt,'max_normal_impulse':0.,'contact_steps':0}
                    self.events[key]['max_normal_impulse']=max(self.events[key]['max_normal_impulse'],j)
                    self.events[key]['contact_steps']+=1
                self.p += max(0.,penetration-.00015)*.7*n
        self.t+=dt

    def states(self):
        return {'ball':{'position':self.p.tolist(),'velocity':self.v.tolist(),'angular_velocity':self.omega.tolist()}}
