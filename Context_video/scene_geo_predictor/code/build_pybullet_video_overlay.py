"""Render pilot scenes and overlay observed/replay/PyBullet trajectories.

The phase-one pilot intentionally contains no RGB video.  This bridge therefore
creates a canonical render from the same blueprint and overlays the saved
physics replay plus the context-only PyBullet rollout produced by
``pybullet_context_rollout.py``.  The output is a small, local diagnostic, not
an appearance or model-quality benchmark.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np


PROJECT = Path(__file__).resolve().parent
ENGINE_ROOT = Path("/home/gaoya/Code_Video/Dataset_physv_v2v_0819")
DATA_DEFAULT = Path(
    "/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_phase1_v5"
)
ROLLOUT_DEFAULT = Path(
    "/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_pybullet_context_preview_v2"
)
OUTPUT_DEFAULT = Path(
    "/data/gaoya/agent-data/outputs/physvideo_next_experiment_20260921_pybullet_context_video_overlay_v1"
)
FPS = 30
OBSERVED_FRAMES = 8
FUTURE_FRAMES = 41
TOTAL_FRAMES = OBSERVED_FRAMES + FUTURE_FRAMES

DEFAULT_CASES = (
    "phase1_aperture_g03_v0280",
    "phase1_aperture_g03_v0720",
    "phase1_deflector_g03_v86000",
    "phase1_support_edge_g03_v0620",
)

# BGR colors for OpenCV.  These deliberately match the dark diagnostic UI.
CYAN = (239, 231, 124)
WHITE = (244, 246, 239)
ORANGE = (92, 179, 255)
VIOLET = (201, 164, 255)
RED = (85, 112, 255)
INK = (238, 241, 236)
MUTED = (154, 174, 184)
PANEL = (18, 33, 44)


def load_engine():
    """Import the repository renderer as a package so its relative imports work."""
    sys.path[:0] = [str(PROJECT), str(ENGINE_ROOT)]
    import prepare_physvideo_phase1_pilot as pilot

    generator, _ = pilot.load_engine()
    from scripts import render_sim_0705 as renderer

    return pilot, generator, renderer


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_case(pilot, generator, record: dict):
    family = str(record["family"])
    group = int(str(record["group_id"]).rsplit("g", 1)[1])
    value = float(record["geometry_value"])
    family_index = {"aperture": 0, "deflector": 1, "support_edge": 2}[family]
    seed = 2026092100 + family_index * 100 + group
    case = pilot.make_case(generator, family, group, value, seed)
    if case.case_id != record["key"]:
        raise ValueError(f"reconstructed case mismatch: {case.case_id} != {record['key']}")
    return case, seed


def renderable_blueprint(blueprint):
    """Give the legacy renderer catalog keys without changing scene physics.

    The phase-one protocol deliberately names its unified sphere and pilot
    families independently of the older renderer catalog.  The renderer only
    uses these keys for prompt/material descriptions; shape, pose, mass and
    collision parameters remain untouched.
    """
    shape_fallback = {
        "sphere": "ball",
        "puck": "flat_puck",
        "capsule": "capsule_can",
        "cylinder": "upright_cylinder",
        "rounded_box": "barrier_box",
        "box": "barrier_box",
    }
    objects = tuple(
        replace(obj, family_key=shape_fallback.get(obj.shape, "barrier_box"))
        for obj in blueprint.objects
    )
    # F1 is a catalogued one-dynamic-object family and is sufficient for the
    # renderer's caption bundle; it does not enter the PyBullet replay.
    return replace(blueprint, family_key="F1", objects=objects)


def project_points(points: np.ndarray, camera, width: int, height: int) -> list[tuple[int, int] | None]:
    """Project world XYZ points with the same look-at convention as pyrender."""
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    up_world = np.asarray(camera.up, dtype=np.float64)
    back = eye - target
    back /= max(float(np.linalg.norm(back)), 1e-12)
    right = np.cross(up_world, back)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(back, right)
    forward = -back
    f = float(height) / (2.0 * np.tan(np.radians(float(camera.yfov_deg)) / 2.0))
    cx = 0.5 * (float(width) - 1.0)
    cy = 0.5 * (float(height) - 1.0)
    projected: list[tuple[int, int] | None] = []
    for point in np.asarray(points, dtype=np.float64):
        delta = point - eye
        depth = float(np.dot(delta, forward))
        if depth <= 1e-6:
            projected.append(None)
            continue
        x = cx + f * float(np.dot(delta, right)) / depth
        y = cy - f * float(np.dot(delta, up)) / depth
        projected.append((int(round(x)), int(round(y))))
    return projected


def draw_polyline(frame: np.ndarray, points: list[tuple[int, int] | None], color, thickness: int, dashed: bool = False) -> None:
    for index in range(1, len(points)):
        left, right = points[index - 1], points[index]
        if left is None or right is None:
            continue
        if dashed and index % 2 == 0:
            continue
        cv2.line(frame, left, right, color, thickness, cv2.LINE_AA)


def draw_marker(frame: np.ndarray, point: tuple[int, int] | None, color, radius: int = 5, ring: bool = False) -> None:
    if point is None:
        return
    if ring:
        cv2.circle(frame, point, radius + 3, RED, 2, cv2.LINE_AA)
    cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)
    cv2.circle(frame, point, radius + 1, (9, 17, 24), 1, cv2.LINE_AA)


def add_label(frame: np.ndarray, text: str, origin: tuple[int, int], color=INK, scale: float = 0.48) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 12, 17), 3, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def overlay_video(
    source_video: Path,
    output_video: Path,
    case: dict,
    camera,
    width: int,
    height: int,
) -> dict:
    observed = np.asarray(case["observed"], dtype=np.float32)
    target = np.asarray(case["target"], dtype=np.float32)
    bullet = np.asarray(case["bullet"], dtype=np.float32)
    replay = np.concatenate([observed, target], axis=0)
    bullet_full = np.concatenate([observed, bullet], axis=0)
    replay_px = project_points(replay, camera, width, height)
    bullet_px = project_points(bullet_full, camera, width, height)
    observed_px = replay_px[:OBSERVED_FRAMES]

    contact_frames = case.get("contact_frames", [])
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open canonical render: {source_video}")
    frames: list[np.ndarray] = []
    try:
        for frame_index in range(TOTAL_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"canonical render ended at frame {frame_index}: {source_video}")
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            overlay = frame.copy()

            # A translucent top strip keeps the provenance readable without
            # hiding the object or the geometry that the video is meant to show.
            cv2.rectangle(overlay, (0, 0), (width, 58), PANEL, -1)
            frame = cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0)

            observed_end = min(frame_index + 1, OBSERVED_FRAMES)
            draw_polyline(frame, observed_px[:observed_end], CYAN, 4)
            if frame_index >= OBSERVED_FRAMES - 1:
                draw_polyline(frame, replay_px[OBSERVED_FRAMES - 1 : frame_index + 1], WHITE, 2, dashed=True)
                draw_polyline(frame, bullet_px[OBSERVED_FRAMES - 1 : frame_index + 1], ORANGE, 4)
            if frame_index < OBSERVED_FRAMES:
                draw_marker(frame, observed_px[frame_index], CYAN, 6)
            else:
                draw_marker(frame, replay_px[frame_index], WHITE, 5)
                draw_marker(frame, bullet_px[frame_index], ORANGE, 6)
                names = contact_frames[frame_index - OBSERVED_FRAMES] if frame_index - OBSERVED_FRAMES < len(contact_frames) else []
                if names:
                    draw_marker(frame, bullet_px[frame_index], RED, 8, ring=True)

            phase = "OBSERVED CONTEXT | RGB0-7" if frame_index < OBSERVED_FRAMES else "ROLLOUT | RGB7 -> future"
            add_label(frame, f"{case['key']}  |  {phase}", (16, 23), INK, 0.42)
            add_label(frame, "canonical physics render | no original RGB", (16, 45), MUTED, 0.40)
            add_label(frame, f"frame {frame_index:02d}/{TOTAL_FRAMES - 1:02d}  t={frame_index / FPS:0.2f}s", (width - 185, 45), CYAN, 0.36)
            add_label(frame, "cyan context", (16, height - 36), CYAN, 0.40)
            add_label(frame, "white replay", (125, height - 36), WHITE, 0.40)
            add_label(frame, "orange context->Bullet", (215, height - 36), ORANGE, 0.40)
            if frame_index >= OBSERVED_FRAMES and contact_frames[frame_index - OBSERVED_FRAMES]:
                add_label(frame, "red ring: contact", (width - 132, height - 36), RED, 0.38)
            frames.append(frame)
    finally:
        capture.release()

    output_video.parent.mkdir(parents=True, exist_ok=True)
    # Use the project's H.264 writer so browser playback works consistently.
    from scripts import generate_sim_preview_gallery as legacy

    legacy._write_video_h264(output_video, frames)
    return {"source_video": str(source_video), "overlay_video": str(output_video), "frames": len(frames), "fps": FPS}


def build_html(results: dict) -> str:
    payload = json.dumps(results, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Context → Bullet · video overlay</title>
<style>
:root{{--bg:#08121b;--panel:#10202b;--panel2:#0d1a24;--line:#24404d;--ink:#edf4f5;--muted:#91a7b2;--cyan:#7ce7ef;--orange:#ffb35c;--violet:#c9a4ff;--red:#ff7085;--green:#9ee493}}
*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(900px 520px at 8% -10%,#173848 0%,transparent 62%),var(--bg);color:var(--ink);font:14px/1.5 "IBM Plex Sans","Segoe UI",sans-serif}}.wrap{{max-width:1440px;margin:auto;padding:28px 30px 42px}}.eyebrow{{font:11px "IBM Plex Mono",monospace;letter-spacing:.16em;text-transform:uppercase;color:var(--cyan);margin-bottom:10px}}.hero{{display:flex;justify-content:space-between;align-items:flex-end;gap:18px;border-bottom:1px solid var(--line);padding-bottom:22px}}h1{{font-size:clamp(28px,4vw,50px);line-height:1.03;letter-spacing:-.055em;margin:0;font-weight:650}}.hero p{{max-width:700px;color:var(--muted);margin:10px 0 0}}.stamp{{font:12px "IBM Plex Mono",monospace;color:var(--muted);white-space:nowrap}}.layout{{display:grid;grid-template-columns:245px 1fr;gap:18px;margin-top:19px}}.rail,.panel{{background:linear-gradient(145deg,rgba(20,40,53,.97),rgba(10,24,34,.97));border:1px solid var(--line);box-shadow:0 18px 48px rgba(0,0,0,.22)}}.rail{{padding:16px}}.rail h2,.panel h2{{font-size:12px;text-transform:uppercase;letter-spacing:.14em;color:var(--muted);margin:0;font-weight:600}}.case-btn{{display:block;width:100%;text-align:left;background:transparent;border:0;border-bottom:1px solid var(--line);color:var(--ink);padding:14px 4px;cursor:pointer}}.case-btn .family{{display:block;font:10px "IBM Plex Mono",monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.13em}}.case-btn .name{{display:block;margin-top:3px;font-weight:600}}.case-btn .value{{display:block;color:var(--cyan);font:12px "IBM Plex Mono",monospace;margin-top:4px}}.case-btn.active{{padding-left:11px;border-left:3px solid var(--orange);background:linear-gradient(90deg,rgba(255,179,92,.12),transparent)}}.note{{color:var(--muted);font-size:12px;margin-top:18px}}.main{{min-width:0}}.metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:13px}}.metric{{background:var(--panel2);border:1px solid var(--line);padding:12px 13px}}.metric .label{{display:block;font:10px "IBM Plex Mono",monospace;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}}.metric .num{{display:block;font:20px "IBM Plex Mono",monospace;margin-top:5px;color:var(--orange)}}.video-panel{{padding:14px}}.video-head{{display:flex;justify-content:space-between;gap:10px;align-items:baseline;margin-bottom:10px}}video{{width:100%;display:block;background:#03090e;border:1px solid #1d3440;aspect-ratio:16/9;object-fit:contain}}.caption{{color:var(--muted);font:11px "IBM Plex Mono",monospace;margin-top:9px}}.links{{display:flex;gap:14px;flex-wrap:wrap;margin-top:8px}}a{{color:var(--cyan)}}.provenance{{margin-top:13px;color:var(--muted);font-size:12px;border-left:2px solid var(--cyan);padding:9px 12px;background:rgba(124,231,239,.04)}}.provenance b{{color:var(--ink)}}.footer{{margin-top:18px;color:var(--muted);font:11px "IBM Plex Mono",monospace}}@media(max-width:980px){{.layout{{grid-template-columns:1fr}}.rail{{display:grid;grid-template-columns:repeat(3,1fr);gap:5px}}.rail h2,.note{{grid-column:1/-1}}.case-btn{{border-bottom:0}}}}@media(max-width:640px){{.wrap{{padding:20px 14px}}.hero{{display:block}}.stamp{{display:block;margin-top:14px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.rail{{grid-template-columns:1fr}}}}
</style></head><body><div class="wrap"><header class="hero"><div><div class="eyebrow">physVideo / video bridge</div><h1>Context <span style="color:var(--cyan)">→</span> Bullet <span style="color:var(--orange)">/ overlay</span></h1><p>Canonical scene render with three traces: the observed RGB0–7 context, the saved original Bullet replay, and a fresh PyBullet rollout initialized only from RGB7.</p></div><div class="stamp">CPU · DIRECT · H.264 · 30 FPS</div></header><div class="layout"><aside class="rail"><h2>Cases</h2><div id="cases"></div><div class="note">The pilot has <b>no original RGB/MP4</b>. The bottom trace and metrics are diagnostic overlays, not a camera-model prediction.</div></aside><main class="main"><section class="metrics" id="metrics"></section><section class="panel video-panel"><div class="video-head"><h2 id="video-title">Overlay video</h2><span id="video-phase" class="stamp"></span></div><video id="video" controls muted playsinline></video><div id="caption" class="caption"></div><div id="links" class="links"></div></section><div class="provenance"><b>Reading the overlay:</b> cyan is the 8-frame observed history; white is the saved future; orange is the context-only PyBullet rollout; a red ring marks a non-floor contact in that rollout. The render underneath is generated from the same physics blueprint and is explicitly labeled canonical.</div><div class="footer">Structured-state pilot · <a href="results.json">results.json</a> · <a href="../physvideo_next_experiment_20260921_pybullet_context_preview_v2/index.html">trajectory-only view</a></div></main></div></div><script>
const DATA={payload};let selected=0;
const fmt=(x,unit='m')=>x==null?'—':(unit==='mm'?(x*1000).toFixed(1)+' mm':x.toFixed(3)+' m');
function renderCases(){{const el=document.getElementById('cases');el.innerHTML='';DATA.cases.forEach((c,i)=>{{const b=document.createElement('button');b.className='case-btn'+(i===selected?' active':'');b.innerHTML='<span class="family">'+c.family+' · '+c.group_id+'</span><span class="name">'+c.key+'</span><span class="value">'+c.geometry_value+' '+c.geometry_units+'</span>';b.onclick=()=>{{selected=i;render()}};el.appendChild(b)}})}}
function render(){{const c=DATA.cases[selected],m=c.metrics;document.getElementById('metrics').innerHTML=[['ADE',fmt(m.ADE_m,'mm')],['FDE',fmt(m.FDE_m,'mm')],['max error',fmt(m.max_error_m,'mm')],['non-floor contacts',m.frames_with_nonfloor_contact+' frames']].map(x=>'<div class="metric"><span class="label">'+x[0]+'</span><span class="num">'+x[1]+'</span></div>').join('');document.getElementById('video-title').textContent=c.key+' · overlay';document.getElementById('video-phase').textContent=c.family+' · '+c.geometry_value+' '+c.geometry_units;const v=document.getElementById('video');v.src=c.overlay_video;v.load();document.getElementById('caption').textContent='Saved replay displacement '+m.target_displacement_end_m.toFixed(3)+' m · PyBullet displacement '+m.bullet_displacement_end_m.toFixed(3)+' m · RGB7 velocity init error '+m.velocity_init_error_mps.toFixed(3)+' m/s';document.getElementById('links').innerHTML='<a href="'+c.overlay_video+'" download>download overlay</a><a href="'+c.canonical_video+'" download>download canonical</a>';renderCases()}}render();
</script></body></html>'''


def run(args: argparse.Namespace) -> dict:
    data_root = args.data_root
    rollout_root = args.rollout_root
    output_root = args.output_root
    manifest = load_json(data_root / "pilot_manifest.json")
    rollout = load_json(rollout_root / "results.json")
    rollout_by_key = {row["key"]: row for row in rollout["cases"]}
    records = {row["key"]: row for row in manifest["records"]}
    missing = [key for key in args.cases if key not in records or key not in rollout_by_key]
    if missing:
        raise KeyError(f"cases missing from manifest/rollout: {missing}")

    pilot, generator, renderer = load_engine()
    all_cases = []
    for key in args.cases:
        record = records[key]
        rollout_case = rollout_by_key[key]
        case, seed = reconstruct_case(pilot, generator, record)
        canonical_root = output_root / "canonical" / key
        source_video = canonical_root / "videos" / f"{key}.mp4"
        if not source_video.exists():
            canonical = renderer.render_blueprint_case(
                blueprint=renderable_blueprint(case.blueprint),
                seed=seed,
                output_root=canonical_root,
                width=args.width,
                height=args.height,
                scene_style=args.scene_style,
                preserve_states=False,
            )
            source_video = Path(canonical["video"])
        overlay_video = output_root / "videos" / f"{key}_overlay.mp4"
        video_meta = overlay_video_fn(
            source_video=source_video,
            output_video=overlay_video,
            case=rollout_case,
            camera=case.blueprint.camera,
            width=args.width,
            height=args.height,
        )
        all_cases.append(
            {
                "key": key,
                "family": rollout_case["family"],
                "group_id": rollout_case["group_id"],
                "geometry_value": rollout_case["geometry_value"],
                "geometry_units": rollout_case["geometry_units"],
                "metrics": rollout_case["metrics"],
                "contact_frames": rollout_case["contact_frames"],
                "canonical_video": str(source_video.relative_to(output_root)),
                "overlay_video": str(overlay_video.relative_to(output_root)),
                "video": video_meta,
            }
        )

    results = {
        "schema": "physvideo_context_bullet_video_overlay_v1",
        "status": "EXECUTED",
        "note": "The pilot has rgb_generated=false; videos are canonical renders from the same blueprint, with trajectory overlays.",
        "output_root": str(output_root),
        "data_root": str(data_root),
        "rollout_root": str(rollout_root),
        "fps": FPS,
        "resolution": [args.width, args.height],
        "observed_frames": OBSERVED_FRAMES,
        "future_frames": FUTURE_FRAMES,
        "scene_style": args.scene_style,
        "cases": all_cases,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "index.html").write_text(build_html(results), encoding="utf-8")
    return results


# Keep the call site readable while avoiding a name collision with the CLI
# function.  This alias also makes it straightforward to monkey-patch in a
# notebook when inspecting a single case.
overlay_video_fn = overlay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_DEFAULT)
    parser.add_argument("--rollout-root", type=Path, default=ROLLOUT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--scene-style", default="indoor_realistic", choices=("indoor_realistic", "indoor_natural", "simple"))
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "output_root": result.get("output_root", ""), "cases": [c["key"] for c in result["cases"]]}, indent=2))
