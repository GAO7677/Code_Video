from pathlib import Path
import json
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
OUTPUT_DIR = ROOT / "tdw_axis_export_compare"
OUTPUT_VIDEO = OUTPUT_DIR / "axis_wrong_replay_compare.mp4"
OUTPUT_POSTER = OUTPUT_DIR / "axis_wrong_replay_compare_poster.png"
OUTPUT_JSON = OUTPUT_DIR / "axis_wrong_replay_compare.json"

FPS = 24
NUM_FRAMES = 192
CANVAS_W = 2560
CANVAS_H = 1040
PANEL_W = 1180
PANEL_H = 720
MARGIN_X = 70
MARGIN_Y = 92
GAP_X = 60
FOOTER_TOP = 850


def load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size=size)
    return ImageFont.load_default()


def normalize(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-8:
        return v
    return v / n


def make_camera(cam_pos, look_at, up_hint):
    cam_pos = np.asarray(cam_pos, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    up_hint = np.asarray(up_hint, dtype=np.float64)
    forward = normalize(look_at - cam_pos)
    right = normalize(np.cross(forward, up_hint))
    up = normalize(np.cross(right, forward))
    return cam_pos, right, up, forward


def project(points, camera, panel_x, panel_y):
    cam_pos, cam_right, cam_up, cam_forward = camera
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    rel = pts - cam_pos[None, :]
    x_cam = np.sum(rel * cam_right[None, :], axis=1)
    y_cam = np.sum(rel * cam_up[None, :], axis=1)
    z_cam = np.sum(rel * cam_forward[None, :], axis=1)
    safe_z = np.where(z_cam > 0.05, z_cam, np.nan)
    focal = 700.0
    cx = panel_x + PANEL_W / 2.0
    cy = panel_y + PANEL_H / 2.0 + 40.0
    u = cx + focal * (x_cam / safe_z)
    v = cy - focal * (y_cam / safe_z)
    return np.stack([u, v], axis=-1), z_cam


def draw_line_3d(draw, p0, p1, camera, panel_x, panel_y, fill, width=3):
    uv, z = project([p0, p1], camera, panel_x, panel_y)
    if np.all(np.isfinite(uv)) and np.all(z > 0.05):
        draw.line((float(uv[0, 0]), float(uv[0, 1]), float(uv[1, 0]), float(uv[1, 1])), fill=fill, width=width)


def draw_grid(draw, mode, camera, panel_x, panel_y):
    grid_color = (206, 200, 190)
    axis_x = (187, 77, 77)
    axis_y = (80, 144, 92)
    axis_z = (68, 114, 179)
    if mode == "yup":
        for x in np.linspace(-1.5, 1.5, 7):
            draw_line_3d(draw, [x, 0.0, -1.5], [x, 0.0, 1.5], camera, panel_x, panel_y, grid_color, 2)
        for z in np.linspace(-1.5, 1.5, 7):
            draw_line_3d(draw, [-1.5, 0.0, z], [1.5, 0.0, z], camera, panel_x, panel_y, grid_color, 2)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], camera, panel_x, panel_y, axis_x, 5)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], camera, panel_x, panel_y, axis_y, 5)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], camera, panel_x, panel_y, axis_z, 5)
    else:
        for x in np.linspace(-1.5, 1.5, 7):
            draw_line_3d(draw, [x, -1.5, 0.0], [x, 1.5, 0.0], camera, panel_x, panel_y, grid_color, 2)
        for y in np.linspace(-1.5, 1.5, 7):
            draw_line_3d(draw, [-1.5, y, 0.0], [1.5, y, 0.0], camera, panel_x, panel_y, grid_color, 2)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], camera, panel_x, panel_y, axis_x, 5)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], camera, panel_x, panel_y, axis_y, 5)
        draw_line_3d(draw, [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], camera, panel_x, panel_y, axis_z, 5)


def draw_labels(draw, mode, camera, panel_x, panel_y, font):
    labels = {
        "X": np.array([1.08, 0.0, 0.0], dtype=np.float64),
        "Y": np.array([0.0, 1.08, 0.0], dtype=np.float64),
        "Z": np.array([0.0, 0.0, 1.08], dtype=np.float64),
    }
    uv, z = project(np.stack(list(labels.values()), axis=0), camera, panel_x, panel_y)
    colors = {"X": (187, 77, 77), "Y": (80, 144, 92), "Z": (68, 114, 179)}
    for idx, key in enumerate(labels.keys()):
        if np.isfinite(uv[idx]).all() and z[idx] > 0.05:
            suffix = " up" if (mode == "yup" and key == "Y") or (mode == "zup" and key == "Z") else ""
            draw.text((float(uv[idx, 0]) + 4, float(uv[idx, 1]) - 10), key + suffix, font=font, fill=colors[key])


def draw_ball(draw, pos, radius, camera, panel_x, panel_y, fill):
    uv, z = project([pos], camera, panel_x, panel_y)
    if not np.isfinite(uv[0]).all() or z[0] <= 0.05:
        return
    scale = float(np.clip(220.0 / z[0], 16.0, 42.0))
    x = float(uv[0, 0])
    y = float(uv[0, 1])
    r = scale * float(radius)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=fill, outline=(30, 30, 30), width=2)


def draw_traj(draw, traj, camera, panel_x, panel_y, fill, width=4):
    if len(traj) < 2:
        return
    uv, z = project(np.asarray(traj, dtype=np.float64), camera, panel_x, panel_y)
    prev = None
    for idx in range(len(traj)):
        if not np.isfinite(uv[idx]).all() or z[idx] <= 0.05:
            prev = None
            continue
        curr = (float(uv[idx, 0]), float(uv[idx, 1]))
        if prev is not None:
            draw.line(prev + curr, fill=fill, width=width)
        prev = curr


def tuple_stream(frame_idx):
    t = frame_idx / FPS
    x = 0.18 * math.sin(0.55 * t)
    z = 0.42
    y = max(0.18, 1.55 - 0.5 * 9.81 * (min(t, 0.55) ** 2))
    return np.array([x, y, z], dtype=np.float64)


def build_frames():
    left_cam = make_camera([3.1, 2.2, -3.2], [0.0, 0.55, 0.3], [0.0, 1.0, 0.0])
    right_cam = make_camera([3.1, -3.2, 2.2], [0.0, 0.3, 0.55], [0.0, 0.0, 1.0])
    title_font = load_font(34)
    head_font = load_font(26)
    body_font = load_font(20)
    mono_font = load_font(24)

    left_panel_x = MARGIN_X
    right_panel_x = MARGIN_X + PANEL_W + GAP_X
    panel_y = MARGIN_Y

    traj = [tuple_stream(i) for i in range(NUM_FRAMES)]
    frames = []
    for idx in range(NUM_FRAMES):
        current = traj[idx]
        history = traj[: idx + 1]
        canvas = Image.new("RGB", (CANVAS_W, CANVAS_H), (244, 240, 232))
        draw = ImageDraw.Draw(canvas)

        draw.text((MARGIN_X, 24), "Wrong Replay Compare: same numeric tuple, different axis interpretation",
                  font=title_font, fill=(24, 22, 19))

        draw.rounded_rectangle((left_panel_x, panel_y, left_panel_x + PANEL_W, panel_y + PANEL_H),
                               radius=24, fill=(255, 255, 255), outline=(220, 214, 205), width=2)
        draw.rounded_rectangle((right_panel_x, panel_y, right_panel_x + PANEL_W, panel_y + PANEL_H),
                               radius=24, fill=(255, 255, 255), outline=(220, 214, 205), width=2)

        draw.text((left_panel_x + 28, panel_y + 20), "Correct replay: TDW tuples interpreted as Y-up",
                  font=head_font, fill=(133, 95, 60))
        draw.text((right_panel_x + 28, panel_y + 20), "Wrong replay: same tuples directly interpreted as Z-up",
                  font=head_font, fill=(70, 107, 93))

        draw_grid(draw, "yup", left_cam, left_panel_x, panel_y)
        draw_grid(draw, "zup", right_cam, right_panel_x, panel_y)
        draw_labels(draw, "yup", left_cam, left_panel_x, panel_y, body_font)
        draw_labels(draw, "zup", right_cam, right_panel_x, panel_y, body_font)

        draw_traj(draw, history, left_cam, left_panel_x, panel_y, (133, 95, 60), 5)
        draw_traj(draw, history, right_cam, right_panel_x, panel_y, (70, 107, 93), 5)
        draw_ball(draw, current, 0.12, left_cam, left_panel_x, panel_y, (225, 119, 85))
        draw_ball(draw, current, 0.12, right_cam, right_panel_x, panel_y, (94, 161, 140))

        draw.rounded_rectangle((MARGIN_X, FOOTER_TOP, CANVAS_W - MARGIN_X, CANVAS_H - 36),
                               radius=24, fill=(255, 255, 255), outline=(220, 214, 205), width=2)
        draw.text((MARGIN_X + 28, FOOTER_TOP + 22), f"Frame {idx:03d}   tuple stream = [x, y, z] = {np.round(current, 3).tolist()}",
                  font=mono_font, fill=(24, 22, 19))

        left_lines = [
            "Left panel uses TDW native convention:",
            "x = horizontal, y = vertical, z = depth",
            "Gravity acts on -Y, so the point falls onto the ground plane.",
        ]
        right_lines = [
            "Right panel makes the classic mistake:",
            "It reuses the same numeric tuple but treats z as the vertical axis.",
            "Result: the fall turns into sideways drift / hovering.",
        ]
        y0 = FOOTER_TOP + 76
        for line in left_lines:
            draw.text((MARGIN_X + 28, y0), line, font=body_font, fill=(60, 58, 54))
            y0 += 32
        y1 = FOOTER_TOP + 76
        for line in right_lines:
            draw.text((CANVAS_W // 2 + 24, y1), line, font=body_font, fill=(60, 58, 54))
            y1 += 32

        draw.text((MARGIN_X + 28, CANVAS_H - 82),
                  "This is why RGB can look unchanged while exported 3D truth becomes wrong if axis conversion is skipped.",
                  font=mono_font, fill=(24, 22, 19))
        frames.append(np.asarray(canvas))
    return frames


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = build_frames()
    imageio.mimwrite(str(OUTPUT_VIDEO), frames, fps=FPS, quality=8)
    imageio.imwrite(str(OUTPUT_POSTER), frames[0])
    payload = {
        "video": str(OUTPUT_VIDEO),
        "poster": str(OUTPUT_POSTER),
        "message": "Left is the correct Y-up replay. Right intentionally skips Y-up to Z-up conversion and therefore misinterprets the same tuple stream.",
        "tuple_semantics": {
            "correct_y_up": "x right, y up, z depth",
            "wrong_z_up": "x right, y depth, z up",
        },
    }
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT_VIDEO)
    print(OUTPUT_POSTER)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()
