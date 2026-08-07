#!/usr/bin/env python3
"""Build static frame-image contact sheets from an existing loss visualization run.

The source MP4s already contain the exact loss overlays. This utility decodes
them into image sheets and adds explicit VAE and V-JEPA temporal mappings:

* Wan VAE38: 49 frames -> 13 temporal latents (first frame + 4-frame groups).
* V-JEPA full-video input: 49 raw frames -> 50 model frames -> 25 tubelets.
  The final raw frame is duplicated once only when the tubelet size requires
  even temporal length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "xssc_vjepa_loss_heatmaps/pybullet_multiobject_step03463_compare"
)
DEFAULT_STEP_RUN = "step03463_lora_actual_frames"
DEFAULT_NO_STEP_RUN = "no_step03463_lora_actual_frames"
DEFAULT_PAGE = "comparison_step03463_pybullet_multiobject_all_frames_images.html"

FRAME_COUNT = 49
VAE_TEMPORAL_STRIDE = 4
VJEPA_TUBELET_SIZE = 2
SHEET_COLUMNS = 7
TILE_WIDTH = 224
TILE_IMAGE_HEIGHT = 128
TILE_STRIP_HEIGHT = 60
SHEET_GAP = 3
JPEG_QUALITY = 88


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_video(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.size == 0 or not np.isfinite(frame).all():
            raise RuntimeError(f"Invalid frame in video: {path}")
        frames.append(frame)
    capture.release()
    if len(frames) != FRAME_COUNT:
        raise RuntimeError(
            f"Expected {FRAME_COUNT} frames in {path}, got {len(frames)}"
        )
    return frames


def _vae_latent_span(frame_index: int, *, latent_time_steps: int) -> tuple[int, int, int]:
    if not 0 <= int(frame_index) < FRAME_COUNT:
        raise ValueError(frame_index)
    if int(frame_index) == 0:
        latent_index = 0
        return latent_index, 0, 0
    latent_index = 1 + (int(frame_index) - 1) // VAE_TEMPORAL_STRIDE
    start = 1 + VAE_TEMPORAL_STRIDE * (latent_index - 1)
    end = min(FRAME_COUNT - 1, VAE_TEMPORAL_STRIDE * latent_index)
    if latent_index >= int(latent_time_steps):
        raise RuntimeError(
            f"Frame {frame_index} maps outside VAE latent time {latent_time_steps}"
        )
    return latent_index, start, end


def _vae_spans(*, latent_time_steps: int) -> list[tuple[int, int]]:
    spans = [(0, 0)]
    for latent_index in range(1, int(latent_time_steps)):
        spans.append(
            (
                1 + VAE_TEMPORAL_STRIDE * (latent_index - 1),
                min(FRAME_COUNT - 1, VAE_TEMPORAL_STRIDE * latent_index),
            )
        )
    return spans


def _vjepa_model_frame_indices(record: dict[str, Any]) -> list[int]:
    model_indices = record.get("vjepa_model_frame_indices")
    if isinstance(model_indices, list) and model_indices:
        return [int(value) for value in model_indices]
    selected = [int(value) for value in record["vjepa_frame_indices"]]
    if str(record.get("vjepa_frame_sampling")) == "full" and len(selected) % VJEPA_TUBELET_SIZE:
        selected = selected + [selected[-1]]
    return selected


def _vjepa_mapping(record: dict[str, Any]) -> dict[int, dict[str, Any]]:
    selected = _vjepa_model_frame_indices(record)
    if len(selected) % VJEPA_TUBELET_SIZE:
        raise RuntimeError("V-JEPA selected frame count is not tubelet divisible")
    used = {int(value) for value in record["vjepa_loss_frame_indices"]}
    mapping: dict[int, dict[str, Any]] = {}
    for token_index in range(0, len(selected), VJEPA_TUBELET_SIZE):
        frames = selected[token_index : token_index + VJEPA_TUBELET_SIZE]
        for frame_index in frames:
            frame_key = int(frame_index)
            if not 0 <= frame_key < FRAME_COUNT:
                continue
            mapping[frame_key] = {
                "token_index": token_index // VJEPA_TUBELET_SIZE,
                "frames": frames,
                "used": frame_key in used,
            }
    return mapping


def _put_text(image: np.ndarray, text: str, x: int, y: int, *, color: tuple[int, int, int]) -> None:
    cv2.putText(
        image,
        text,
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_temporal_strip(
    frame: np.ndarray,
    *,
    frame_index: int,
    record: dict[str, Any],
    latent_time_steps: int,
    kind: str,
) -> np.ndarray:
    image = cv2.resize(
        frame,
        (TILE_WIDTH, TILE_IMAGE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    strip = np.full(
        (TILE_STRIP_HEIGHT, TILE_WIDTH, 3),
        (18, 24, 21),
        dtype=np.uint8,
    )
    vae_index, vae_start, vae_end = _vae_latent_span(
        frame_index,
        latent_time_steps=latent_time_steps,
    )
    vjepa = _vjepa_mapping(record)
    vjepa_info = vjepa.get(int(frame_index))
    if vjepa_info is None:
        vjepa_label = "VJ -- not sampled"
    else:
        status = "USED" if vjepa_info["used"] else "masked"
        pair = ",".join(str(value) for value in vjepa_info["frames"])
        vjepa_label = f"VJ t{vjepa_info['token_index']:02d} [{pair}] {status}"
    vae_label = f"VAE z{vae_index:02d}/{latent_time_steps - 1:02d} [{vae_start}-{vae_end}]"
    _put_text(strip, f"f{frame_index:02d} | {vae_label}", 4, 12, color=(188, 225, 202))
    _put_text(
        strip,
        vjepa_label,
        4,
        25,
        color=(139, 220, 231) if vjepa_info is not None else (140, 150, 144),
    )

    # The two bars show the temporal compression, not spatial loss intensity.
    bar_left = 4
    bar_right = TILE_WIDTH - 4
    bar_width = bar_right - bar_left
    vae_spans = _vae_spans(latent_time_steps=latent_time_steps)
    vae_segment = bar_width / float(latent_time_steps)
    for latent_index in range(latent_time_steps):
        x0 = int(round(bar_left + latent_index * vae_segment))
        x1 = int(round(bar_left + (latent_index + 1) * vae_segment)) - 1
        active = latent_index == vae_index
        color = (85, 185, 118) if active else (47, 72, 57)
        cv2.rectangle(strip, (x0, 31), (max(x0, x1), 36), color, -1)
    vjepa_tokens = int(record.get("vjepa_temporal_tokens", 0))
    if vjepa_tokens <= 0:
        vjepa_tokens = len(_vjepa_model_frame_indices(record)) // VJEPA_TUBELET_SIZE
    vjepa_segment = bar_width / float(vjepa_tokens)
    current_token = None if vjepa_info is None else int(vjepa_info["token_index"])
    for token_index in range(vjepa_tokens):
        x0 = int(round(bar_left + token_index * vjepa_segment))
        x1 = int(round(bar_left + (token_index + 1) * vjepa_segment)) - 1
        if token_index == current_token:
            color = (57, 186, 208) if bool(vjepa_info["used"]) else (70, 104, 110)
        else:
            color = (39, 65, 72)
        cv2.rectangle(strip, (x0, 41), (max(x0, x1), 46), color, -1)
    _put_text(
        strip,
        f"green: VAE {FRAME_COUNT}->{latent_time_steps} | cyan: VJ full-video ->"
        f"{vjepa_tokens} tubelets | {kind}",
        4,
        57,
        color=(175, 185, 178),
    )
    return np.concatenate([image, strip], axis=0)


def _contact_sheet(
    frames: list[np.ndarray],
    *,
    record: dict[str, Any],
    latent_time_steps: int,
    kind: str,
) -> np.ndarray:
    tile_height = TILE_IMAGE_HEIGHT + TILE_STRIP_HEIGHT
    rows = (len(frames) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = np.full(
        (
            rows * tile_height + max(rows - 1, 0) * SHEET_GAP,
            SHEET_COLUMNS * TILE_WIDTH + max(SHEET_COLUMNS - 1, 0) * SHEET_GAP,
            3,
        ),
        (7, 10, 8),
        dtype=np.uint8,
    )
    for frame_index, frame in enumerate(frames):
        tile = _draw_temporal_strip(
            frame,
            frame_index=frame_index,
            record=record,
            latent_time_steps=latent_time_steps,
            kind=kind,
        )
        row, column = divmod(frame_index, SHEET_COLUMNS)
        top = row * (tile_height + SHEET_GAP)
        left = column * (TILE_WIDTH + SHEET_GAP)
        sheet[top : top + tile_height, left : left + TILE_WIDTH] = tile
        cv2.rectangle(
            sheet,
            (left, top),
            (left + TILE_WIDTH - 1, top + tile_height - 1),
            (82, 101, 88),
            1,
        )
    return sheet


def _object_text(record: dict[str, Any]) -> str:
    return " | ".join(
        str(value.get("object_phrase") or value.get("object_noun") or "object")
        for value in record.get("selected_entity_slots", [])
    )


def _build_page(
    output_root: Path,
    *,
    pairs: list[dict[str, Any]],
    page_name: str,
) -> Path:
    payload = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V-JEPA Loss Frame Images</title>
<style>
:root { --bg:#101513; --line:#3a4b40; --text:#edf5ee; --muted:#a5b6aa;
  --lime:#b8e986; --cyan:#79c8d2; --amber:#f1b45c; }
* { box-sizing:border-box; }
body { margin:0; background:linear-gradient(135deg,#101513,#1a241f 56%,#101513);
  color:var(--text); font:14px/1.45 "IBM Plex Sans",Verdana,sans-serif; }
header { padding:18px 24px 15px; border-bottom:1px solid var(--line); background:#141c18; }
h1 { margin:0 0 5px; font:700 24px/1.1 Georgia,serif; }
main { max-width:1900px; margin:0 auto; padding:18px 24px 36px; }
.toolbar { display:flex; flex-wrap:wrap; align-items:end; gap:10px 16px;
  padding-bottom:14px; border-bottom:1px solid var(--line); }
label { display:grid; gap:4px; color:var(--muted); font-size:12px; }
select { min-height:36px; border:1px solid var(--line); border-radius:5px;
  background:#222e27; color:var(--text); padding:6px 10px; accent-color:var(--lime); }
.status { margin:14px 0; color:var(--muted); }
.case { border-top:1px solid var(--line); padding:16px 0 20px; }
.case-head { display:flex; flex-wrap:wrap; justify-content:space-between; gap:8px;
  align-items:baseline; margin-bottom:8px; }
.case-head b { font-size:16px; }
.objects { color:var(--muted); font-size:12px; margin-bottom:10px; }
.sheet-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; }
figure { margin:0; min-width:0; }
figcaption { min-height:38px; color:var(--muted); font-size:12px; }
figcaption b { color:var(--text); font-size:13px; }
img { display:block; width:100%; height:auto; border:1px solid var(--line);
  background:#050606; image-rendering:auto; }
.target img { border-color:var(--amber); }
.flow img { border-color:#d67e57; }
.vjepa img { border-color:var(--cyan); }
.legend { border-top:1px solid var(--line); margin-top:16px; padding-top:14px;
  color:var(--muted); }
.legend b { color:var(--text); }
code { color:#d3edac; }
@media(max-width:1250px) { .sheet-grid { grid-template-columns:1fr; } }
@media(max-width:680px) { main,header { padding-left:12px; padding-right:12px; } }
</style>
</head>
<body>
<header>
  <h1>V-JEPA Loss Frame Images</h1>
    <div class="muted">Static 49-frame contact sheets | Wan VAE38 and full-video V-JEPA temporal compression shown per frame</div>
</header>
<main>
  <div class="toolbar">
    <label>Model condition
      <select id="condition"><option value="step">step03463_lora</option><option value="no_step">no_step03463_lora</option></select>
    </label>
    <label>Timestep weight
      <select id="weight"><option value="low_weight">Low</option><option value="mid_weight">Mid</option><option value="high_weight">High</option></select>
    </label>
  </div>
  <div class="status" id="status"></div>
  <div id="gallery"></div>
  <div class="legend">
    <b>Reading each frame:</b>
    <code>VAE z03 [9-12]</code> means raw frames 9 through 12 share one Wan VAE temporal latent.
    <code>VJ t03 [13,16] USED</code> means the tubelet frames contribute to feature loss.
    In full-video mode the final raw frame is duplicated once only for tubelet alignment.
    Gray or missing V-JEPA tokens are not feature-loss frames. The green bar is the
    49-to-13 VAE mapping; the cyan bar is the full-video V-JEPA mapping.
  </div>
</main>
<script>
const DATA=__PAYLOAD__;
const condition=document.getElementById("condition");
const weight=document.getElementById("weight");
const gallery=document.getElementById("gallery");
const status=document.getElementById("status");
function recordFor(pair){return pair[condition.value];}
function render(){
  const records=DATA.map(recordFor), label=weight.value;
    status.textContent="Showing static frame images for "+condition.value+" | "+label+
    " | every case contains all 49 raw target frames and full-video V-JEPA overlays.";
  gallery.innerHTML=records.map(record=>{
    const sheets=record.image_sheets[label];
    return '<section class="case"><div class="case-head"><b>case '+record.case_position+
      ' | '+record.case_label+' | seed '+record.case_seed+'</b><span class="muted">'+
      record.selected_object_count+' objects | VAE '+record.vae_latent_time_steps+
      ' temporal latents | V-JEPA '+record.vjepa_temporal_tokens+
      ' tubelets | future-loss frames '+record.vjepa_loss_frame_indices.length+
      '</span></div><div class="objects">'+record.object_text+'</div><div class="sheet-grid">'+
      '<figure class="target"><figcaption><b>Loss-input target</b><br>all 49 frames</figcaption><img loading="lazy" src="'+record.target_image+'"></figure>'+ 
      '<figure class="flow"><figcaption><b>Flow loss</b><br>latent v-MSE projected to frames</figcaption><img loading="lazy" src="'+sheets.flow+'"></figure>'+ 
      '<figure class="vjepa"><figcaption><b>V-JEPA feature loss</b><br>full-video input and tubelets</figcaption><img loading="lazy" src="'+sheets.vjepa_feature+'"></figure>'+ 
      '</div></section>';
  }).join("");
}
condition.onchange=render; weight.onchange=render; render();
</script>
</body>
</html>
"""
    page_path = output_root / page_name
    page_path.write_text(page.replace("__PAYLOAD__", payload), encoding="utf-8")
    return page_path


def build_gallery(
    output_root: Path,
    *,
    step_run: str,
    no_step_run: str,
    page_name: str,
) -> Path:
    run_records: dict[str, dict[int, dict[str, Any]]] = {}
    for run in (step_run, no_step_run):
        index = json.loads((output_root / run / "index.json").read_text(encoding="utf-8"))
        run_records[run] = {
            int(record["case_position"]): record for record in index["records"]
        }
    if set(run_records[step_run]) != set(run_records[no_step_run]):
        raise RuntimeError("Step/no-step case positions do not match")

    assets_root = output_root / "frame_image_gallery_assets"
    pairs: list[dict[str, Any]] = []
    for position in sorted(run_records[step_run]):
        pair: dict[str, Any] = {}
        for run, condition in ((step_run, "step"), (no_step_run, "no_step")):
            record = run_records[run][position]
            if condition == "no_step":
                step_record = run_records[step_run][position]
                for field in (
                    "case_id",
                    "case_label",
                    "case_seed",
                    "vjepa_frame_indices",
                    "vjepa_loss_frame_indices",
                ):
                    if record[field] != step_record[field]:
                        raise RuntimeError(f"Case {position} differs in {field}")
            output_case = assets_root / run / f"case_{position:02d}"
            output_case.mkdir(parents=True, exist_ok=True)
            latent_time_steps = 1 + (FRAME_COUNT - 1) // VAE_TEMPORAL_STRIDE
            if int(record.get("vjepa_temporal_tokens", 0)) == 0:
                record["vjepa_temporal_tokens"] = len(record["vjepa_frame_indices"]) // VJEPA_TUBELET_SIZE
            target_frames = _read_video(output_root / run / record["target_video"])
            target_image = output_case / "target.jpg"
            cv2.imwrite(
                str(target_image),
                _contact_sheet(
                    target_frames,
                    record=record,
                    latent_time_steps=latent_time_steps,
                    kind="target",
                ),
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
            )
            sheets_by_weight: dict[str, dict[str, str]] = {}
            for variant in record["variants"]:
                variant_dir = output_case / str(variant["label"])
                variant_dir.mkdir(parents=True, exist_ok=True)
                sheets: dict[str, str] = {}
                for kind in ("flow", "vjepa_feature"):
                    frames = _read_video(
                        output_root / run / variant["videos"][kind]
                    )
                    image_path = variant_dir / f"{kind}.jpg"
                    cv2.imwrite(
                        str(image_path),
                        _contact_sheet(
                            frames,
                            record=record,
                            latent_time_steps=latent_time_steps,
                            kind=kind,
                        ),
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                    )
                    sheets[kind] = image_path.relative_to(output_root).as_posix()
                sheets_by_weight[str(variant["label"])] = sheets
            pair[condition] = {
                "case_position": int(record["case_position"]),
                "case_id": str(record["case_id"]),
                "case_label": str(record["case_label"]),
                "case_seed": int(record["case_seed"]),
                "selected_object_count": int(record["selected_object_count"]),
                "object_text": _object_text(record),
                "vjepa_frame_indices": [int(value) for value in record["vjepa_frame_indices"]],
                "vjepa_loss_frame_indices": [int(value) for value in record["vjepa_loss_frame_indices"]],
                "vae_latent_time_steps": int(latent_time_steps),
                "vjepa_temporal_tokens": int(record["vjepa_temporal_tokens"]),
                "target_image": target_image.relative_to(output_root).as_posix(),
                "image_sheets": sheets_by_weight,
            }
        pairs.append(pair)
    _json_dump(output_root / "frame_image_gallery_index.json", pairs)
    return _build_page(output_root, pairs=pairs, page_name=page_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--step-run", default=DEFAULT_STEP_RUN)
    parser.add_argument("--no-step-run", default=DEFAULT_NO_STEP_RUN)
    parser.add_argument("--page-name", default=DEFAULT_PAGE)
    args = parser.parse_args()
    page = build_gallery(
        args.output_root.resolve(),
        step_run=args.step_run,
        no_step_run=args.no_step_run,
        page_name=args.page_name,
    )
    print(f"Image gallery page: {page}", flush=True)


if __name__ == "__main__":
    main()
