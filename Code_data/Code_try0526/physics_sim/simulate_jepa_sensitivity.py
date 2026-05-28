#!/usr/bin/env python3
"""JEPA 敏感性实验 — 固定外观，系统改变运动参数"""

import os, json
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np; np.infty = np.inf
import pybullet as p, pybullet_data, pyrender, trimesh, cv2
from pyrender.constants import RenderFlags
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = DATA_DIR / "videos" / "jepa_sensitivity"
IMG_W, IMG_H = 1280, 720
FPS = 60
SIM_DURATION = 2.5
SIM_STEPS = int(SIM_DURATION * 240)
RECORD_EVERY = 4

CAM_EYE = np.array([0.05, -2.2, 1.2])
CAM_TARGET = np.array([0.05, 0.3, 0.45])
CAM_UP = np.array([0, 0, 1])

FIXED_APPEARANCE = {
    "ball_color": [0.85, 0.42, 0.18],
    "floor_color": [0.45, 0.40, 0.36],
    "wall_color": [0.48, 0.50, 0.55],
    "bg_color": [0.18, 0.16, 0.12],
    "ambient": [0.03, 0.03, 0.04],
    "light_color": [1.0, 0.95, 0.90],
    "light_intensity": 70.0,
    "light_pos": [-1.5, -1.5, 2.5],
}

# ─── Scenarios: each varies ONE parameter from baseline ─────────
BASELINE = {"e": 0.7, "mu": 0.5, "mass": 1.0, "block_mass": 1.5,
            "vx": 3.5, "vz": 1.8, "ball_x": -1.0, "gravity": 9.81,
            "block_x": 0.3}

def sc(label, desc, **kw):
    p = BASELINE.copy(); p.update(kw)
    return {"name": label, "desc": desc, **p}

SCENARIOS = [
    # ── Velocity sweep ──
    sc("vel_005",  "初速 0.5 m/s（极慢飘移）", vx=0.5, vz=0.3),
    sc("vel_015",  "初速 1.5 m/s（慢速）",      vx=1.5, vz=0.8),
    sc("vel_035",  "初速 3.5 m/s（基准）",      vx=3.5, vz=1.8),
    sc("vel_070",  "初速 7.0 m/s（快速）",      vx=7.0, vz=2.5),
    sc("vel_140",  "初速 14.0 m/s（极快速）",   vx=14.0, vz=4.0),
    # ── Mass sweep ──
    sc("mass_001", "球 0.01kg（乒乓球）",        mass=0.01),
    sc("mass_005", "球 0.05kg",                  mass=0.05),
    sc("mass_010", "球 0.1kg",                   mass=0.1),
    sc("mass_100", "球 1.0kg（基准）",           mass=1.0),
    sc("mass_500", "球 5.0kg（保龄球）",         mass=5.0),
    sc("mass_2000","球 20.0kg（铁球）",          mass=20.0),
    sc("mass_9999","球 100.0kg（巨型球）",       mass=100.0),
    # ── Gravity sweep ──
    sc("grav_050", "重力 4.9 m/s²（月球）",      gravity=4.9),
    sc("grav_098", "重力 9.81 m/s²（地球）",     gravity=9.81),
    sc("grav_200", "重力 19.6 m/s²（超重）",     gravity=19.6),
    # ── No collision ──
    sc("nomiss",   "球从上方飞过（不碰撞）",     ball_x=-1.0, block_x=3.0),
    # ── Reverse direction ──
    sc("rev_035",  "球从右侧撞来（反向）",       ball_x=2.0, vx=-3.5, block_x=-0.3),
    # ── Block mass sweep ──
    sc("blk_005",  "木块 0.5kg（轻块）",         block_mass=0.5),
    sc("blk_500",  "木块 5.0kg（重块）",         block_mass=5.0),
    sc("blk_2000", "木块 20.0kg（固定块）",      block_mass=20.0),
]


def _look_at(eye, target, up):
    eye=np.array(eye,dtype=float); target=np.array(target,dtype=float); up=np.array(up,dtype=float)
    z=eye-target; z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.array([[x[0],y[0],z[0],eye[0]],[x[1],y[1],z[1],eye[1]],[x[2],y[2],z[2],eye[2]],[0,0,0,1]])
def _tr(x,y,z): return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]],dtype=float)
def _pb_pose(pos, quat):
    mat=np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    pose=np.eye(4,dtype=float); pose[:3,:3]=mat; pose[:3,3]=pos; return pose


class FixedRenderer:
    def __init__(self):
        a = FIXED_APPEARANCE
        self.scene = pyrender.Scene(bg_color=a["bg_color"], ambient_light=a["ambient"])
        f = trimesh.creation.box(extents=[20,10,0.04])
        f.visual.vertex_colors = np.tile(a["floor_color"], (len(f.vertices),1)).astype(np.float32)
        self.scene.add(pyrender.Mesh.from_trimesh(f, smooth=False), pose=_tr(0,2,-0.04))
        w = trimesh.creation.box(extents=[20,0.02,3])
        w.visual.vertex_colors = np.tile(a["wall_color"], (len(w.vertices),1)).astype(np.float32)
        self.scene.add(pyrender.Mesh.from_trimesh(w, smooth=False), pose=_tr(0,5,1.5))
        spot = pyrender.SpotLight(color=a["light_color"], intensity=a["light_intensity"],
                                   innerConeAngle=0.4, outerConeAngle=1.0)
        self.scene.add(spot, pose=_look_at(a["light_pos"], [0.3,0.5,0], [0,0,1]))
        cam = pyrender.PerspectiveCamera(yfov=np.radians(55), aspectRatio=IMG_W/IMG_H)
        self.scene.add(cam, pose=_look_at(CAM_EYE, CAM_TARGET, CAM_UP))
        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
        self.ball_node = None; self.block_node = None

    def set_ball(self, pos, quat):
        if self.ball_node: self.scene.remove_node(self.ball_node)
        s = trimesh.creation.icosphere(subdivisions=3, radius=0.18)
        s.visual.vertex_colors = np.tile(FIXED_APPEARANCE["ball_color"], (len(s.vertices),1)).astype(np.float32)
        self.ball_node = self.scene.add(pyrender.Mesh.from_trimesh(s, smooth=True), pose=_pb_pose(pos,quat))

    def set_block(self, pos, quat):
        if self.block_node: self.scene.remove_node(self.block_node)
        b = trimesh.creation.box(extents=[0.5,0.4,0.6])
        b.visual.vertex_colors = np.tile([0.48,0.32,0.18], (len(b.vertices),1)).astype(np.float32)
        self.block_node = self.scene.add(pyrender.Mesh.from_trimesh(b, smooth=False), pose=_pb_pose(pos,quat))

    def render(self): return self.renderer.render(self.scene, flags=RenderFlags.SHADOWS_SPOT)[0]
    def cleanup(self): self.renderer.delete()


def run_one(sc):
    p.setGravity(0,0,-sc["gravity"])
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100, numSubSteps=1)

    ball = p.createMultiBody(sc["mass"], p.createCollisionShape(p.GEOM_SPHERE, radius=0.18),
                              basePosition=(sc["ball_x"],0,0.2))
    block = p.createMultiBody(sc["block_mass"], p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25,0.2,0.3]),
                               basePosition=(sc["block_x"],0,0.3))
    p.changeDynamics(ball,-1,restitution=sc["e"],lateralFriction=sc["mu"],
                     spinningFriction=0.003,linearDamping=0.03,angularDamping=0.03)
    p.changeDynamics(block,-1,restitution=sc["e"],lateralFriction=sc["mu"],
                     spinningFriction=0.008,linearDamping=0.06,angularDamping=0.06,
                     activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    p.resetBaseVelocity(ball, linearVelocity=[sc["vx"],0,sc["vz"]])
    for _ in range(10): p.stepSimulation()

    r = FixedRenderer()
    frames = []
    for step in range(SIM_STEPS):
        p.stepSimulation()
        if step % RECORD_EVERY != 0: continue
        bp,bq = p.getBasePositionAndOrientation(ball)
        kp,kq = p.getBasePositionAndOrientation(block)
        r.set_ball(bp,bq); r.set_block(kp,kq)
        color = r.render()
        frames.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))

    vpath = VIDEO_DIR / f"{sc['name']}.mp4"
    out = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W,IMG_H))
    for f in frames: out.write(f)
    out.release()

    # Metadata JSON
    meta = {
        "video": str(vpath), "scenario": sc["name"], "description": sc["desc"],
        "parameters": {"restitution": sc["e"], "friction": sc["mu"],
                       "ball_mass_kg": sc["mass"], "block_mass_kg": sc["block_mass"],
                       "velocity_ms": [sc["vx"],0,sc["vz"]], "gravity": sc["gravity"]},
        "caption": "Ball colliding with a wooden block",
        "experiment": "jepa_sensitivity",
    }
    json.dump(meta, open(VIDEO_DIR / f"{sc['name']}.json","w"), indent=2, ensure_ascii=False)
    r.cleanup()
    p.removeBody(ball); p.removeBody(block)
    return vpath


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    for sc in SCENARIOS:
        print(f"[{sc['name']}] {sc['desc']}")
        run_one(sc)

    p.disconnect()
    print(f"\nDone: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
