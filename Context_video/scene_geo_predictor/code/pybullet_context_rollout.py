"""Roll a few observed context states forward in PyBullet and build a local view.

This is a small, CPU-only bridge diagnostic.  It deliberately uses the saved
physics-only pilot states as the context-state source, estimates the RGB7
velocity from positions 6 and 7, and starts a fresh PyBullet world at RGB7.
It does not render RGB or feed future labels into the rollout.  The saved
physics future is used only for the post-rollout comparison plot/metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent
DATA_DEFAULT = Path("/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5")
OUTPUT_DEFAULT = Path("/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_pybullet_context_preview_v1")
FPS = 30
SIM_HZ = 240
SUBSTEPS = SIM_HZ // FPS
FRAME_COUNT = 41
BALL_RADIUS_M = 0.11

# One strongly interactive representative from each pilot family.  The pilot
# manifest is checked below, so these are never silently replaced by a missing
# or differently shaped sample.
DEFAULT_CASES = (
    "phase1_aperture_g03_v0720",
    "phase1_deflector_g03_v86000",
    "phase1_support_edge_g03_v0620",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def load_engine():
    # Reuse the exact generator/replay dependencies already used by the pilot.
    sys.path.insert(0, str(PROJECT))
    import prepare_physvideo_phase1_pilot as pilot

    generator, _ = pilot.load_engine()
    return pilot, generator


def spec_from_key(key: str, manifest: dict) -> dict:
    for row in manifest["records"]:
        if row["key"] == key:
            return row
    raise KeyError(f"case is not in pilot manifest: {key}")


def case_from_record(pilot, generator, record: dict):
    family = record["family"]
    group = int(record["group_id"].rsplit("g", 1)[1])
    value = float(record["geometry_value"])
    family_index = {"aperture": 0, "deflector": 1, "support_edge": 2}[family]
    seed = 2026092100 + family_index * 100 + group
    case = pilot.make_case(generator, family, group, value, seed)
    if case.case_id != record["key"]:
        raise ValueError(f"reconstructed case id mismatch: {case.case_id} != {record['key']}")
    return case, seed


def rolling_angular_velocity(linear_velocity: np.ndarray, radius: float) -> np.ndarray:
    """Approximate rolling spin for a horizontal sphere without angular labels."""
    vx, vy, _ = [float(v) for v in linear_velocity]
    return np.asarray([vy / radius, -vx / radius, 0.0], dtype=np.float32)


def _collision_shape_for_client(p, obj, client_id: int) -> int:
    """Create pilot primitive collision shapes on an explicit Bullet client.

    The original legacy helper uses PyBullet's implicit default client.  The
    pilot bridge can be called while another DIRECT client exists, so the
    small pilot shape vocabulary is created explicitly on this rollout's
    client instead of silently writing to client 0.
    """
    size = obj.size
    shape = str(obj.shape)
    if shape in {"sphere", "ellipsoid"}:
        radius = float(size["radius"] if "radius" in size else max(size["rx"], size["ry"], size["rz"]))
        return int(p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id))
    if shape in {"box", "rounded_box", "wedge"}:
        return int(
            p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[float(size["hx"]), float(size["hy"]), float(size["hz"])],
                physicsClientId=client_id,
            )
        )
    raise ValueError(f"explicit-client pilot shape adapter does not support {shape!r}")


def _step_v2v_simulation_for_client(p, body_ids: dict[str, int], blueprint, client_id: int):
    """Run the original pilot step semantics on an explicit client.

    Pilot families take the ordinary ``stepSimulation`` branch.  The puck
    correction is copied here for completeness because the legacy generator's
    implementation predates explicit client IDs.
    """
    if blueprint.family_key != "SCENE_PUCK_BARRIER":
        p.stepSimulation(physicsClientId=client_id)
        return None

    puck_id = body_ids.get("puck")
    barrier_id = body_ids.get("puck_barrier")
    if puck_id is None or barrier_id is None:
        p.stepSimulation(physicsClientId=client_id)
        return None

    previous_velocity = np.asarray(
        p.getBaseVelocity(puck_id, physicsClientId=client_id)[0], dtype=np.float64
    )
    p.stepSimulation(physicsClientId=client_id)
    contacts = p.getContactPoints(puck_id, barrier_id, physicsClientId=client_id)
    if not contacts:
        return None
    contact = max(
        contacts,
        key=lambda point: math.hypot(float(point[7][0]), float(point[7][1])),
    )
    normal = np.asarray(contact[7], dtype=np.float64)
    normal[2] = 0.0
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1e-8:
        return None
    normal /= normal_length
    current_velocity = np.asarray(
        p.getBaseVelocity(puck_id, physicsClientId=client_id)[0], dtype=np.float64
    )
    incoming_normal_speed = float(np.dot(previous_velocity, normal))
    solved_normal_speed = float(np.dot(current_velocity, normal))
    objects = {obj.name: obj for obj in blueprint.objects}
    effective_restitution = min(
        float(objects["puck"].restitution),
        float(objects["puck_barrier"].restitution),
    )
    if incoming_normal_speed >= -0.08 or solved_normal_speed >= 0.08:
        return None
    desired_normal_speed = -effective_restitution * incoming_normal_speed
    correction = desired_normal_speed - solved_normal_speed
    current_velocity += correction * normal
    angular_velocity = p.getBaseVelocity(puck_id, physicsClientId=client_id)[1]
    p.resetBaseVelocity(
        puck_id,
        linearVelocity=current_velocity.tolist(),
        angularVelocity=angular_velocity,
        physicsClientId=client_id,
    )
    return {
        "incoming_normal_speed_mps": round(incoming_normal_speed, 6),
        "solver_normal_speed_mps": round(solved_normal_speed, 6),
        "outgoing_normal_speed_mps": round(desired_normal_speed, 6),
        "effective_restitution": round(effective_restitution, 6),
        "normal_xy": [round(float(value), 6) for value in normal[:2]],
        "corrected_velocity_xy_mps": [round(float(value), 6) for value in current_velocity[:2]],
    }


def rollout_from_rgb7(pilot, generator, case, record: dict, sample_dir: Path):
    """Initialize static geometry plus the RGB7 state, then simulate 41 frames."""
    import pybullet as p

    raw_path = sample_dir / "raw" / "states_xyzw.npz"
    with np.load(raw_path, allow_pickle=False) as raw:
        positions = raw["positions"].astype(np.float32)
        linear_velocities = raw["linear_velocities"].astype(np.float32)
        quats = raw["quats"].astype(np.float32)
        names = raw["object_names"].astype(str).tolist()
    dynamic = [i for i, name in enumerate(names) if name == "pilot_ball"]
    if dynamic != [0]:
        raise ValueError(f"expected pilot_ball at object index 0: {record['key']}")
    dynamic_index = dynamic[0]
    # This is the information a context-state bridge supplies: p0..p7,
    # then a finite-difference velocity at RGB7.  Do not use saved future data
    # for initialization.
    p7 = positions[7, dynamic_index].copy()
    v7 = (positions[7, dynamic_index] - positions[6, dynamic_index]) * FPS
    q7 = quats[7, dynamic_index].copy()
    omega7 = rolling_angular_velocity(v7, BALL_RADIUS_M)
    target = positions[8:49, dynamic_index].copy()

    g = generator
    b = case.blueprint
    objects = [o for o in b.objects if not o.metadata.get("visual_only")]
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")
    try:
        p.setAdditionalSearchPath(g.pybullet_data.getDataPath(), physicsClientId=client)
        p.resetSimulation(physicsClientId=client)
        p.setGravity(0.0, 0.0, -g.EARTH_GRAVITY, physicsClientId=client)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1.0 / SIM_HZ,
            numSolverIterations=g.legacy.PHYSICS_SOLVER_ITERATIONS,
            numSubSteps=int(b.metadata.get("physics_sub_steps", SUBSTEPS)),
            contactERP=g.legacy.PHYSICS_CONTACT_ERP,
            erp=g.legacy.PHYSICS_CONTACT_ERP,
            physicsClientId=client,
        )
        plane = p.loadURDF("plane.urdf", physicsClientId=client)
        surface = g.build_surface_catalog()[b.surface_key]
        floor_mu = float(np.clip(
            b.metadata.get("floor_friction", surface.floor_friction_range.midpoint()),
            0.01,
            1.2,
        ))
        # Keep the original replay convention for the floor; it is recorded
        # explicitly so a future protocol can decide whether to change it.
        p.changeDynamics(
            plane,
            -1,
            lateralFriction=floor_mu,
            restitution=0.02,
            activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            physicsClientId=client,
        )
        ids: dict[str, int] = {}
        for obj in objects:
            if obj.dynamic:
                base_position = p7
                base_orientation = q7
            else:
                base_position = obj.position
                base_orientation = g.legacy._quat_from_euler_deg(list(obj.orientation_euler_deg))
            body = p.createMultiBody(
                baseMass=float(obj.mass) if obj.dynamic else 0.0,
                baseCollisionShapeIndex=_collision_shape_for_client(p, obj, client),
                basePosition=list(base_position),
                baseOrientation=list(base_orientation),
                physicsClientId=client,
            )
            kwargs = dict(
                restitution=float(obj.restitution),
                lateralFriction=float(obj.friction),
                rollingFriction=float(obj.metadata.get("rolling_friction", 0)),
                spinningFriction=float(obj.metadata.get("spinning_friction", 0)),
                linearDamping=float(obj.linear_damping),
                angularDamping=float(obj.angular_damping),
                activationState=p.ACTIVATION_STATE_DISABLE_SLEEPING,
            )
            if "ccd_swept_sphere_radius_m" in obj.metadata:
                kwargs.update(
                    ccdSweptSphereRadius=float(obj.metadata["ccd_swept_sphere_radius_m"]),
                    contactProcessingThreshold=0.0,
                )
            kwargs["physicsClientId"] = client
            p.changeDynamics(body, -1, **kwargs)
            if obj.dynamic:
                p.resetBaseVelocity(
                    body,
                    linearVelocity=list(v7),
                    angularVelocity=list(omega7),
                    physicsClientId=client,
                )
            ids[obj.name] = body
        for descriptor in b.metadata.get("constraints", []):
            raise ValueError("explicit-client pilot bridge does not support blueprint constraints")
        for left, right in b.metadata.get("disable_collision_pairs", []):
            p.setCollisionFilterPair(ids[str(left)], ids[str(right)], -1, -1, 0, physicsClientId=client)

        dynamic_body = ids["pilot_ball"]
        p.performCollisionDetection(physicsClientId=client)
        initial_nonfloor_contacts = [
            int(point[2])
            for point in p.getContactPoints(bodyA=dynamic_body, physicsClientId=client)
            if int(point[2]) != int(plane)
        ]
        rollout_positions, rollout_velocities, contact_frames = [], [], []
        contact_body_names = {}
        for name, body_id in ids.items():
            contact_body_names[int(body_id)] = name
        contact_body_names[int(plane)] = "floor"
        corrections = 0
        for frame in range(FRAME_COUNT):
            for _ in range(SUBSTEPS):
                corrections += bool(_step_v2v_simulation_for_client(p, ids, b, client) is not None)
            state = p.getBasePositionAndOrientation(dynamic_body, physicsClientId=client)
            velocity = p.getBaseVelocity(dynamic_body, physicsClientId=client)
            rollout_positions.append(state[0])
            rollout_velocities.append(velocity[0])
            nonfloor = [
                contact_body_names.get(int(point[2]), str(int(point[2])))
                for point in p.getContactPoints(bodyA=dynamic_body, physicsClientId=client)
                if int(point[2]) != int(plane)
            ]
            contact_frames.append(nonfloor)

        rollout = np.asarray(rollout_positions, dtype=np.float32)
        rollout_velocity = np.asarray(rollout_velocities, dtype=np.float32)
    finally:
        p.disconnect(client)

    error = np.linalg.norm(rollout - target, axis=-1)
    target_delta = np.linalg.norm(target - target[0], axis=-1)
    rollout_delta = np.linalg.norm(rollout - rollout[0], axis=-1)
    first_contact = next((i for i, names_at_frame in enumerate(contact_frames) if names_at_frame), None)
    return {
        "key": record["key"],
        "family": record["family"],
        "group_id": record["group_id"],
        "geometry_value": record["geometry_value"],
        "geometry_units": record["geometry_units"],
        "interaction": record["interaction"],
        "observed": positions[:8, dynamic_index].tolist(),
        "target": target.tolist(),
        "bullet": rollout.tolist(),
        "bullet_velocity": rollout_velocity.tolist(),
        "contact_frames": contact_frames,
        "first_nonfloor_contact_future_index": first_contact,
        "initial_nonfloor_contacts": [contact_body_names.get(i, str(i)) for i in initial_nonfloor_contacts],
        "metrics": {
            "ADE_m": float(error.mean()),
            "FDE_m": float(error[-1]),
            "max_error_m": float(error.max()),
            "target_displacement_end_m": float(target_delta[-1]),
            "bullet_displacement_end_m": float(rollout_delta[-1]),
            "velocity_init_error_mps": float(np.linalg.norm(v7 - linear_velocities[7, dynamic_index])),
            "rollout_corrections": int(corrections),
            "frames_with_nonfloor_contact": int(sum(bool(x) for x in contact_frames)),
        },
        "protocol": {
            "state_source": "saved context positions p0..p7; v7 finite difference p7-p6",
            "future_used_for_initialization": False,
            "appearance_used": False,
            "physics_engine": "PyBullet DIRECT via original replay helpers",
            "sim_hz": SIM_HZ,
            "fps": FPS,
            "substeps_per_output_frame": SUBSTEPS,
            "angular_velocity_policy": "rolling_from_v7; sphere-only approximation",
            "floor_restitution": 0.02,
        },
    }


def build_html(results: dict) -> str:
    payload = json.dumps(results, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Context → Bullet | phase-one rollout</title>
<style>
:root{--ink:#e9f1f5;--muted:#91a5b0;--bg:#09121b;--panel:#101f2b;--panel2:#0d1a25;--line:#223a48;--cyan:#7ce7ef;--orange:#ffb35c;--violet:#c9a4ff;--red:#ff7085;--green:#9ee493}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(1000px 520px at 12% -8%,#163243 0%,transparent 58%),var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans","Segoe UI",sans-serif;letter-spacing:.01em}
.wrap{max-width:1440px;margin:0 auto;padding:28px 30px 42px}.eyebrow{font:11px "IBM Plex Mono",ui-monospace,monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);margin-bottom:10px}.hero{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;border-bottom:1px solid var(--line);padding-bottom:23px}.hero h1{font-size:clamp(28px,4vw,51px);line-height:1.03;letter-spacing:-.055em;margin:0;font-weight:650}.hero p{max-width:620px;color:var(--muted);margin:10px 0 0}.stamp{font:12px "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);white-space:nowrap}.layout{display:grid;grid-template-columns:245px 1fr;gap:18px;margin-top:19px}.rail,.panel{background:linear-gradient(145deg,rgba(20,40,53,.96),rgba(10,24,34,.96));border:1px solid var(--line);box-shadow:0 18px 48px rgba(0,0,0,.2)}.rail{padding:16px}.rail h2,.panel h2{font-size:12px;text-transform:uppercase;letter-spacing:.14em;margin:0;color:var(--muted);font-weight:600}.case-btn{display:block;width:100%;text-align:left;background:transparent;border:0;border-bottom:1px solid var(--line);color:var(--ink);padding:15px 4px 14px;cursor:pointer}.case-btn:first-of-type{margin-top:5px}.case-btn .family{display:block;font:10px "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.13em}.case-btn .name{display:block;margin-top:3px;font-weight:600}.case-btn .value{display:block;color:var(--cyan);font:12px "IBM Plex Mono",ui-monospace,monospace;margin-top:5px}.case-btn.active{padding-left:11px;border-left:3px solid var(--orange);background:linear-gradient(90deg,rgba(255,179,92,.12),transparent)}.note{color:var(--muted);font-size:12px;margin-top:18px}.main{min-width:0}.metrics{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px;margin-bottom:13px}.metric{background:var(--panel2);border:1px solid var(--line);padding:12px 13px}.metric .label{display:block;font:10px "IBM Plex Mono",ui-monospace,monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}.metric .num{display:block;font:20px "IBM Plex Mono",ui-monospace,monospace;margin-top:5px;color:var(--orange)}.metric.good .num{color:var(--green)}.plots{display:grid;grid-template-columns:1fr 1fr;gap:13px}.panel{padding:14px}.panel h2{margin-bottom:10px}.plot{width:100%;height:330px;display:block;background:rgba(2,9,14,.5);border:1px solid #1d3440}.plot text{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:10px;fill:#78909c}.grid{stroke:#1d3440;stroke-width:1}.axis{stroke:#63808e;stroke-width:1}.context{fill:none;stroke:var(--cyan);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.gt{fill:none;stroke:#eff6f8;stroke-width:1.8;stroke-dasharray:4 5;opacity:.8}.bullet{fill:none;stroke:var(--orange);stroke-width:3;stroke-linecap:round;stroke-linejoin:round}.selected{stroke:var(--violet);stroke-width:2;fill:#0b151e}.contact{stroke:var(--red);stroke-width:2;stroke-dasharray:3 4}.legend{display:flex;gap:17px;flex-wrap:wrap;color:var(--muted);font:11px "IBM Plex Mono",ui-monospace,monospace;margin-top:8px}.legend i{display:inline-block;width:22px;height:3px;vertical-align:middle;margin-right:5px}.legend .c{background:var(--cyan)}.legend .g{background:#eff6f8}.legend .b{background:var(--orange)}.legend .s{width:9px;height:9px;border:2px solid var(--violet);background:transparent;border-radius:50%}.timeline{margin-top:13px;padding:15px 16px}.timeline-top{display:flex;justify-content:space-between;align-items:center;gap:12px}.timeline-top strong{font-size:18px;letter-spacing:-.03em}.timeline-top code{font:12px "IBM Plex Mono",ui-monospace,monospace;color:var(--cyan)}input[type=range]{width:100%;accent-color:var(--orange);margin:13px 0 3px}.ticks{display:flex;justify-content:space-between;color:var(--muted);font:10px "IBM Plex Mono",ui-monospace,monospace}.provenance{margin-top:13px;color:var(--muted);font-size:12px;border-left:2px solid var(--cyan);padding:9px 12px;background:rgba(124,231,239,.04)}.provenance b{color:var(--ink);font-weight:600}.footer{margin-top:18px;color:var(--muted);font:11px "IBM Plex Mono",ui-monospace,monospace}.footer a{color:var(--cyan)}
@media(max-width:980px){.layout{grid-template-columns:1fr}.rail{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}.rail h2,.note{grid-column:1/-1}.case-btn{border-bottom:0}.metrics{grid-template-columns:repeat(3,1fr)}}@media(max-width:640px){.wrap{padding:20px 14px}.hero{display:block}.stamp{display:block;margin-top:14px}.plots{grid-template-columns:1fr}.metrics{grid-template-columns:repeat(2,1fr)}.rail{grid-template-columns:1fr}.plot{height:270px}}
</style></head><body><div class="wrap">
<header class="hero"><div><div class="eyebrow">physVideo / phase-one diagnostic</div><h1>Context <span style="color:var(--cyan)">→</span> Bullet</h1><p>Start a fresh PyBullet world at RGB7, roll the observed state forward for 41 frames, and compare the coarse physics path with the saved replay. Appearance is intentionally absent.</p></div><div class="stamp">CPU · DIRECT · 240 Hz → 30 Hz</div></header>
<div class="layout"><aside class="rail"><h2>Representative cases</h2><div id="cases"></div><div class="note">The context velocity is estimated from <code>p7 − p6</code>. The saved future is comparison-only.</div></aside><main class="main"><section class="metrics" id="metrics"></section><div class="plots"><section class="panel"><h2>Top-down · x / y</h2><svg class="plot" id="xy" viewBox="0 0 800 330" preserveAspectRatio="none"></svg><div class="legend"><span><i class="c"></i>observed RGB0–7</span><span><i class="g"></i>saved replay</span><span><i class="b"></i>PyBullet rollout</span><span><i class="s"></i>selected frame</span></div></section><section class="panel"><h2>Side view · x / z</h2><svg class="plot" id="xz" viewBox="0 0 800 330" preserveAspectRatio="none"></svg><div class="legend"><span><i class="c"></i>observed</span><span><i class="g"></i>saved replay</span><span><i class="b"></i>PyBullet</span><span style="color:var(--red)">— non-floor contact</span></div></section></div><section class="panel timeline"><div class="timeline-top"><strong id="frame-title">RGB8</strong><code id="frame-detail"></code></div><input id="slider" type="range" min="0" max="40" value="0" step="1"><div class="ticks"><span>RGB7 / t+0</span><span>t+0.67 s</span><span>t+1.33 s</span></div></section><div class="provenance" id="provenance"></div><div class="footer">Structured-state pilot · <a href="https://github.com/bulletphysics/bullet3" target="_blank" rel="noreferrer">PyBullet</a> DIRECT · no RGB rendered · no future input to initialization</div></main></div></div>
<script>
const DATA=__DATA__;
let selectedCase=0;
const casesEl=document.getElementById('cases'), metricsEl=document.getElementById('metrics'), slider=document.getElementById('slider');
function esc(v){return String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function fmt(v,d=3){return Number(v).toFixed(d);}
function drawPlot(svgId, c, axes, selected){
  const svg=document.getElementById(svgId); const W=800,H=330,pad={l:42,r:16,t:16,b:31};
  const obs=c.observed, gt=c.target, pb=c.bullet;
  const all=obs.concat(gt,pb); const xs=all.map(p=>p[axes[0]]), ys=all.map(p=>p[axes[1]]);
  let xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys); const dx=Math.max(xmax-xmin,.08),dy=Math.max(ymax-ymin,.08); xmin-=dx*.08;xmax+=dx*.08;ymin-=dy*.12;ymax+=dy*.12;
  const X=x=>(pad.l+(x-xmin)/(xmax-xmin)*(W-pad.l-pad.r)),Y=y=>(H-pad.b-(y-ymin)/(ymax-ymin)*(H-pad.t-pad.b));
  const pts=a=>a.map(p=>`${X(p[axes[0]]).toFixed(1)},${Y(p[axes[1]]).toFixed(1)}`).join(' ');
  const obsFull=obs, gtFull=[obs[7]].concat(gt), pbFull=[obs[7]].concat(pb);
  let h=''; for(let i=0;i<5;i++){const gy=pad.t+i*(H-pad.t-pad.b)/4; const gx=pad.l+i*(W-pad.l-pad.r)/4; h+=`<line class="grid" x1="${pad.l}" y1="${gy}" x2="${W-pad.r}" y2="${gy}"/><line class="grid" x1="${gx}" y1="${pad.t}" x2="${gx}" y2="${H-pad.b}"/>`;}
  h+=`<line class="axis" x1="${pad.l}" y1="${H-pad.b}" x2="${W-pad.r}" y2="${H-pad.b}"/><line class="axis" x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${H-pad.b}"/>`;
  h+=`<polyline class="gt" points="${pts(gtFull)}"/><polyline class="bullet" points="${pts(pbFull)}"/><polyline class="context" points="${pts(obsFull)}"/>`;
  const si=selected+1, s=pbFull[si], t=gtFull[si]; h+=`<circle class="selected" cx="${X(s[axes[0]])}" cy="${Y(s[axes[1]])}" r="6"/><circle class="gt" cx="${X(t[axes[0]])}" cy="${Y(t[axes[1]])}" r="4"/>`;
  const cf=c.contact_frames[selected]; if(cf&&cf.length){h+=`<line class="contact" x1="${X(s[axes[0]])-9}" y1="${pad.t}" x2="${X(s[axes[0]])-9}" y2="${H-pad.b}"/>`;}
  h+=`<text x="${W-pad.r-42}" y="${H-8}">${axes[0]===0?'x':'x'} / m</text><text x="8" y="${pad.t+9}">${axes[1]===1?'y':'z'} / m</text>`; svg.innerHTML=h;
}
function render(){
 const c=DATA.cases[selectedCase], m=c.metrics; slider.value=slider.value; const vals=[['ADE',fmt(m.ADE_m*1000,1)+' mm',false],['FDE',fmt(m.FDE_m*1000,1)+' mm',false],['max error',fmt(m.max_error_m*1000,1)+' mm',false],['v7 init error',fmt(m.velocity_init_error_mps,3)+' m/s',true],['non-floor contacts',String(m.frames_with_nonfloor_contact),true]];
 metricsEl.innerHTML=vals.map(x=>`<div class="metric ${x[2]?'good':''}"><span class="label">${x[0]}</span><span class="num">${x[1]}</span></div>`).join('');
 document.getElementById('frame-title').textContent=`RGB${8+Number(slider.value)}`; const t=(Number(slider.value)+1)/30; const contact=c.contact_frames[Number(slider.value)]||[]; document.getElementById('frame-detail').textContent=`t + ${fmt(t,3)} s · ${contact.length?('contact: '+contact.join(', ')):'free / floor-only'}`;
 drawPlot('xy',c,[0,1],Number(slider.value)); drawPlot('xz',c,[0,2],Number(slider.value));
 document.getElementById('provenance').innerHTML=`<b>${esc(c.family)}</b> · ${esc(c.key)} · geometry ${fmt(c.geometry_value,3)} ${esc(c.geometry_units)}<br>State: ${esc(c.protocol.state_source)} · saved future used only for comparison · initial non-floor contact: ${c.initial_nonfloor_contacts.length?esc(c.initial_nonfloor_contacts.join(', ')):'none'}`;
}
DATA.cases.forEach((c,i)=>{const b=document.createElement('button');b.className='case-btn';b.innerHTML=`<span class="family">${esc(c.family)}</span><span class="name">${esc(c.key)}</span><span class="value">${fmt(c.geometry_value,3)} ${esc(c.geometry_units)}</span>`;b.onclick=()=>{selectedCase=i;document.querySelectorAll('.case-btn').forEach(x=>x.classList.remove('active'));b.classList.add('active');slider.value=0;render();};casesEl.appendChild(b);if(i===0)b.classList.add('active');});
slider.addEventListener('input',render);render();
</script></body></html>'''.replace("__DATA__", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    manifest = load_json(data_root / "pilot_manifest.json")
    if manifest.get("status") != "EXECUTED":
        raise ValueError("pilot manifest is not EXECUTED")
    pilot, generator = load_engine()
    output.mkdir(parents=True)
    results = {
        "schema": "pybullet_context_rollout_preview_v1",
        "status": "EXECUTED",
        "appearance": "not_used",
        "data_root": str(data_root),
        "cases": [],
    }
    for key in args.cases:
        record = spec_from_key(key, manifest)
        sample_dir = data_root / "samples" / key
        if not sample_dir.is_dir():
            raise FileNotFoundError(sample_dir)
        case, seed = case_from_record(pilot, generator, record)
        result = rollout_from_rgb7(pilot, generator, case, record, sample_dir)
        result["simulation_seed"] = seed
        result["sample_dir"] = str(sample_dir)
        results["cases"].append(result)
        np.savez_compressed(
            output / f"{key}_rollout.npz",
            observed=np.asarray(result["observed"], dtype=np.float32),
            target=np.asarray(result["target"], dtype=np.float32),
            bullet=np.asarray(result["bullet"], dtype=np.float32),
            bullet_velocity=np.asarray(result["bullet_velocity"], dtype=np.float32),
        )
        print(json.dumps({"case": key, **result["metrics"]}, sort_keys=True), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    (output / "index.html").write_text(build_html(results), encoding="utf-8")
    (output / "run_command.txt").write_text(
        f"/data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B {Path(__file__).resolve()} "
        f"--data-root {data_root} --output {output} --cases {' '.join(args.cases)}\n"
    )
    print(json.dumps({"status": results["status"], "output": str(output), "cases": args.cases}, indent=2))


if __name__ == "__main__":
    main()
