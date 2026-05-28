#!/usr/bin/env python3
"""外观变体重渲染 — 相同物理轨迹，不同外观，测试 PDI 外观敏感性"""

import os, json, sys
os.environ["PYOPENGL_PLATFORM"] = "egl"
import numpy as np; np.infty = np.inf
import pybullet as p, pybullet_data
import pyrender, trimesh, cv2
from pyrender.constants import RenderFlags
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = DATA_DIR / "videos" / "ball_block_appearance"
SRC_DIR = DATA_DIR / "videos" / "ball_block"
IMG_W, IMG_H = 1280, 720
FPS = 60

CAM_EYE = np.array([0.05, -2.2, 1.2])
CAM_TARGET = np.array([0.05, 0.3, 0.45])
CAM_UP = np.array([0, 0, 1])

# Physics params (same as original)
SCENARIOS = [
    ("e03_mu05_m1",   0.3, 0.5, 1.0, "e=0.3 plastic"),
    ("e05_mu05_m1",   0.5, 0.5, 1.0, "e=0.5 medium"),
    ("e07_mu05_m1",   0.7, 0.5, 1.0, "e=0.7 bouncy"),
    ("e09_mu05_m1",   0.9, 0.5, 1.0, "e=0.9 superball"),
    ("e07_mu01_m1",   0.7, 0.1, 1.0, "e=0.7 low-fric"),
    ("e07_mu10_m1",   0.7, 1.0, 1.0, "e=0.7 high-fric"),
    ("e07_mu05_m01",  0.7, 0.5, 0.1, "e=0.7 light-ball"),
    ("e07_mu05_m5",   0.7, 0.5, 5.0, "e=0.7 heavy-ball"),
]

# Appearance variants
APPEARANCES = {
    "v1_default": {
        "ball_color": [0.85, 0.42, 0.18],  # orange basketball
        "floor_color": [0.45, 0.40, 0.36],
        "wall_color": [0.48, 0.50, 0.55],
        "bg_color": [0.18, 0.16, 0.12],
        "ambient": [0.03, 0.03, 0.04],
        "light_color": [1.0, 0.95, 0.90],
        "light_intensity": 70.0,
        "light_pos": [-1.5, -1.5, 2.5],
    },
    "v2_dark_blue": {
        "ball_color": [0.15, 0.25, 0.65],
        "floor_color": [0.28, 0.26, 0.24],
        "wall_color": [0.22, 0.24, 0.30],
        "bg_color": [0.05, 0.05, 0.10],
        "ambient": [0.02, 0.02, 0.04],
        "light_color": [0.6, 0.7, 1.0],
        "light_intensity": 80.0,
        "light_pos": [-1.5, -1.5, 2.5],
    },
    "v3_warm_bright": {
        "ball_color": [0.25, 0.70, 0.30],
        "floor_color": [0.60, 0.50, 0.24],
        "wall_color": [0.65, 0.58, 0.45],
        "bg_color": [0.35, 0.30, 0.22],
        "ambient": [0.06, 0.05, 0.03],
        "light_color": [1.0, 0.85, 0.60],
        "light_intensity": 55.0,
        "light_pos": [-1.5, -1.5, 2.5],
    },
}


def _look_at(eye, target, up):
    eye=np.array(eye,dtype=float); target=np.array(target,dtype=float); up=np.array(up,dtype=float)
    z=eye-target; z/=np.linalg.norm(z); x=np.cross(up,z); x/=np.linalg.norm(x); y=np.cross(z,x)
    return np.array([[x[0],y[0],z[0],eye[0]],[x[1],y[1],z[1],eye[1]],[x[2],y[2],z[2],eye[2]],[0,0,0,1]])

def _tr(x,y,z): return np.array([[1,0,0,x],[0,1,0,y],[0,0,1,z],[0,0,0,1]],dtype=float)
def _pb_pose(pos, quat):
    mat=np.array(p.getMatrixFromQuaternion(quat)).reshape(3,3)
    pose=np.eye(4,dtype=float); pose[:3,:3]=mat; pose[:3,3]=pos; return pose


class VariantRenderer:
    def __init__(self, app):
        a = app
        self.scene = pyrender.Scene(bg_color=a["bg_color"], ambient_light=a["ambient"])
        # Floor
        f = trimesh.creation.box(extents=[20,10,0.04])
        f.visual.vertex_colors = np.tile(a["floor_color"], (len(f.vertices),1)).astype(np.float32)
        self.scene.add(pyrender.Mesh.from_trimesh(f, smooth=False), pose=_tr(0,2,-0.04))
        # Wall
        w = trimesh.creation.box(extents=[20,0.02,3])
        w.visual.vertex_colors = np.tile(a["wall_color"], (len(w.vertices),1)).astype(np.float32)
        self.scene.add(pyrender.Mesh.from_trimesh(w, smooth=False), pose=_tr(0,5,1.5))
        # Light
        spot = pyrender.SpotLight(color=a["light_color"], intensity=a["light_intensity"],
                                   innerConeAngle=0.4, outerConeAngle=1.0)
        self.scene.add(spot, pose=_look_at(a["light_pos"], [0.3,0.5,0], [0,0,1]))
        # Camera
        cam = pyrender.PerspectiveCamera(yfov=np.radians(55), aspectRatio=IMG_W/IMG_H)
        self.scene.add(cam, pose=_look_at(CAM_EYE, CAM_TARGET, CAM_UP))
        self.renderer = pyrender.OffscreenRenderer(IMG_W, IMG_H)
        self.ball_node = None
        self.block_node = None

    def set_ball(self, pos, quat, ball_color):
        if self.ball_node: self.scene.remove_node(self.ball_node)
        s = trimesh.creation.icosphere(subdivisions=3, radius=0.18)
        s.visual.vertex_colors = np.tile(ball_color, (len(s.vertices),1)).astype(np.float32)
        self.ball_node = self.scene.add(pyrender.Mesh.from_trimesh(s, smooth=True), pose=_pb_pose(pos,quat))

    def set_block(self, pos, quat):
        if self.block_node: self.scene.remove_node(self.block_node)
        b = trimesh.creation.box(extents=[0.5,0.4,0.6])
        b.visual.vertex_colors = np.tile([0.48,0.32,0.18], (len(b.vertices),1)).astype(np.float32)
        self.block_node = self.scene.add(pyrender.Mesh.from_trimesh(b, smooth=False), pose=_pb_pose(pos,quat))

    def render(self): return self.renderer.render(self.scene, flags=RenderFlags.SHADOWS_SPOT)[0]
    def cleanup(self): self.renderer.delete()


def run_one(sc_name, e, mu, mass, label):
    """Run physics once, record trajectory, render with all variants."""
    p.setGravity(0,0,-9.81)
    p.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=100, numSubSteps=1)

    col_b = p.createCollisionShape(p.GEOM_SPHERE, radius=0.18)
    ball = p.createMultiBody(mass, col_b, basePosition=(-1,0,0.2))
    col_k = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25,0.2,0.3])
    block = p.createMultiBody(1.5, col_k, basePosition=(0.3,0,0.3))
    p.changeDynamics(ball,-1,restitution=e,lateralFriction=mu,spinningFriction=0.003,linearDamping=0.03,angularDamping=0.03)
    p.changeDynamics(block,-1,restitution=e,lateralFriction=mu,spinningFriction=0.008,linearDamping=0.06,angularDamping=0.06,activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING)
    p.resetBaseVelocity(ball, linearVelocity=[3.5,0,1.8])
    for _ in range(10): p.stepSimulation()

    # Record trajectory
    traj = []
    for step in range(int(2.5*240)):
        p.stepSimulation()
        if step % 4 != 0: continue
        bp,bq = p.getBasePositionAndOrientation(ball)
        kp,kq = p.getBasePositionAndOrientation(block)
        traj.append((bp,bq,kp,kq))

    # Render with each variant
    results = {}
    for vname, app in APPEARANCES.items():
        r = VariantRenderer(app)
        frames = []
        for bp,bq,kp,kq in traj:
            r.set_ball(bp,bq,app["ball_color"])
            r.set_block(kp,kq)
            color = r.render()
            frames.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        # Write video
        vpath = VIDEO_DIR / f"{sc_name}_{vname}.mp4"
        out = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_W,IMG_H))
        for f in frames: out.write(f)
        out.release()
        results[vname] = str(vpath)
        r.cleanup()

    p.removeBody(ball); p.removeBody(block)
    return results


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    all_meta = {}
    for sc_name, e, mu, mass, label in SCENARIOS:
        print(f"[{sc_name}] {label}")
        paths = run_one(sc_name, e, mu, mass, label)
        # Write JSON
        for vname, vpath in paths.items():
            meta = {
                "video": vpath,
                "scenario": sc_name,
                "appearance_variant": vname,
                "parameters": {"restitution": e, "lateral_friction": mu, "ball_mass_kg": mass},
                "caption": "Ball colliding with a wooden block",
            }
            jpath = VIDEO_DIR / f"{sc_name}_{vname}.json"
            json.dump(meta, open(jpath, "w"), indent=2, ensure_ascii=False)
            print(f"  {vname}: {jpath.name}")

    p.disconnect()
    print(f"\nDone: {VIDEO_DIR}")


if __name__ == "__main__":
    main()
