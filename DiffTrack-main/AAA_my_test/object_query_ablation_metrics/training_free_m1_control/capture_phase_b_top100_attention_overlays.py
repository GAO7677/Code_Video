#!/usr/bin/env python3
"""Capture and render Phase-B Top100 Object-A attention overlays.

The protocol is intentionally narrow and frozen to
``0613pybullet_sample_001460_w002 / seed 90094``.  Five deterministic replays
are compared: Baseline, Sparse-8-point alpha={0.1,0.25}, and frozen SAM2
full-mask alpha={0.1,0.25}.  All five rows are *observed* with the same frozen
Baseline SAM2 full-mask query set, so changing a row does not change the
measurement itself.

For each latent time t and denoising window W we stream

  mean_{s in W, cfg, (layer,head) in Top100, q in R_t} A_s[q, K_t]

where softmax is still normalized over all 13x22x40 keys.  Only the displayed
same-time K_t slice is selected after softmax.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any

import cv2
import imageio.v3 as iio
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for import_root in (REPO_ROOT, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    _run_pipe_once,
    build_wan_ti2v_pipeline,
    save_video_np,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES  # noqa: E402
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_direct_scaling_phase_bd import (  # noqa: E402
    M1DirectScalingAblator,
    SAM2FullMaskM1DirectScalingAblator,
    TIME_SCOPE_TO_MASK,
    load_full_mask_partition,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    AttentionMatrixAblator,
    generation_inputs,
    object_queries,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (  # noqa: E402
    selected_head_entries,
    sha256_file,
    tracks_root,
    validate_head_ranking,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


CASE = "0613pybullet_sample_001460_w002"
SEED = 90094
GRID = (13, 22, 40)
FRAME_TOKEN_COUNT = GRID[1] * GRID[2]
SEQUENCE = int(np.prod(GRID))
ANCHOR_FRAMES = tuple(range(0, 49, 4))
WINDOWS: dict[str, tuple[int, ...]] = {
    "all40": tuple(range(40)),
    "first10": tuple(range(10)),
    "first20": tuple(range(20)),
    "last20": tuple(range(20, 40)),
}
WINDOW_LABELS = {
    "all40": "40 Steps Mean · S000-S039",
    "first10": "First10 · S000-S009",
    "first20": "First20 · S000-S019",
    "last20": "Last20 · S020-S039",
}
DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/"
    "test5_20case_5seed/test5_phase_bd_manifest.json"
)
DEFAULT_RANKING = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/head_scopes_latest3350_with_random100.json"
)
DEFAULT_TRACKS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/"
    "test5_20case_5seed/frozen_baseline_tracks"
)
SPARSE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/"
    "test5_20case_5seed"
)
FULL_ROOT = SPARSE_ROOT.with_name("test5_20case_5seed_sam2_full_mask")
FULL_MASK_ROOT = FULL_ROOT / "baseline_sam2_full_masks"
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_direct_enhancement_v2/"
    "seed90094_top100_attention_overlays"
)


@dataclass(frozen=True)
class Variant:
    id: str
    label: str
    token_source: str
    alpha: float | None
    video: Path


def _phase_b_video(root: Path, alpha_tag: str) -> Path:
    return (
        root
        / "phase_b"
        / CASE
        / f"seed_{SEED:05d}"
        / (
            "single_object__object_A__m1_all_time__top100"
            f"__alpha_{alpha_tag}__denoise_00_39"
        )
        / "generated.mp4"
    )


def variants(sample: dict[str, Any]) -> tuple[Variant, ...]:
    return (
        Variant(
            "baseline",
            "Baseline · No intervention",
            "baseline",
            None,
            Path(str(sample["baseline_video"])),
        ),
        Variant(
            "sparse_a0p1",
            "Sparse 8-point · alpha=0.1",
            "sparse_points",
            0.1,
            _phase_b_video(SPARSE_ROOT, "0p1"),
        ),
        Variant(
            "full_a0p1",
            "SAM2 Full-mask · alpha=0.1",
            "sam2_full_mask",
            0.1,
            _phase_b_video(FULL_ROOT, "0p1"),
        ),
        Variant(
            "sparse_a0p25",
            "Sparse 8-point · alpha=0.25",
            "sparse_points",
            0.25,
            _phase_b_video(SPARSE_ROOT, "0p25"),
        ),
        Variant(
            "full_a0p25",
            "SAM2 Full-mask · alpha=0.25",
            "sam2_full_mask",
            0.25,
            _phase_b_video(FULL_ROOT, "0p25"),
        ),
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


class SameTimeTop100Capture:
    """Streaming exact-softmax capture with no full attention tensor retained."""

    def __init__(self, entries: list[dict], query_rows_by_time: list[list[int]]) -> None:
        self.by_block: dict[int, list[int]] = {}
        for entry in entries:
            self.by_block.setdefault(int(entry["block"]), []).append(int(entry["head"]))
        for block in self.by_block:
            self.by_block[block] = sorted(set(self.by_block[block]))
        if sum(map(len, self.by_block.values())) != 100:
            raise RuntimeError("capture requires exactly 100 unique Top100 heads")
        if len(query_rows_by_time) != GRID[0] or any(not rows for rows in query_rows_by_time):
            raise RuntimeError("every latent frame must contain full-mask Object-A queries")
        self.query_rows_by_time = [tuple(map(int, rows)) for rows in query_rows_by_time]
        self.sums = {
            name: np.zeros(GRID, dtype=np.float64) for name in WINDOWS
        }
        self.counts = {
            name: np.zeros(GRID[0], dtype=np.int64) for name in WINDOWS
        }
        self.calls_by_step_cfg: dict[tuple[int, int], int] = {}

    @torch.no_grad()
    def record(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        block: int,
        step: int,
        cfg_call: int,
    ) -> None:
        heads = self.by_block.get(int(block), ())
        if not heads:
            return
        if step not in range(40) or cfg_call not in (0, 1):
            raise RuntimeError(f"invalid capture coordinate step={step}, cfg={cfg_call}")
        if q.ndim != 3 or k.ndim != 3 or q.shape[1] != SEQUENCE or k.shape[1] != SEQUENCE:
            raise RuntimeError(f"expected Q/K sequence {SEQUENCE}, got {q.shape}/{k.shape}")
        num_heads = int(q.shape[-1] // 128)
        if q.shape[-1] != num_heads * 128 or k.shape[-1] != q.shape[-1]:
            raise RuntimeError("Q/K channel width is not 128-dim head aligned")
        qh = q.reshape(q.shape[0], SEQUENCE, num_heads, 128).permute(0, 2, 1, 3)
        kh = k.reshape(k.shape[0], SEQUENCE, num_heads, 128).permute(0, 2, 1, 3)
        selected_q = qh[:, heads]
        selected_k = kh[:, heads].transpose(-1, -2)
        scale = 1.0 / math.sqrt(128)
        active_windows = [name for name, steps in WINDOWS.items() if step in steps]
        for time_index, row_values in enumerate(self.query_rows_by_time):
            rows = torch.as_tensor(row_values, device=q.device, dtype=torch.long)
            logits = torch.matmul(selected_q[:, :, rows], selected_k).float().mul(scale)
            probabilities = torch.softmax(logits, dim=-1)
            start = time_index * FRAME_TOKEN_COUNT
            same_time = probabilities[..., start : start + FRAME_TOKEN_COUNT]
            # One map per physical head; query tokens and CFG batch are averaged.
            head_maps = same_time.mean(dim=(0, 2)).reshape(
                len(heads), GRID[1], GRID[2]
            )
            map_sum = head_maps.sum(dim=0).detach().cpu().numpy().astype(np.float64)
            for window in active_windows:
                self.sums[window][time_index] += map_sum
                self.counts[window][time_index] += len(heads)
            del logits, probabilities, same_time, head_maps
        key = (int(step), int(cfg_call))
        self.calls_by_step_cfg[key] = self.calls_by_step_cfg.get(key, 0) + len(heads)

    def finalize(self) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        expected_coordinates = {(step, cfg) for step in range(40) for cfg in (0, 1)}
        if set(self.calls_by_step_cfg) != expected_coordinates:
            missing = sorted(expected_coordinates - set(self.calls_by_step_cfg))
            raise RuntimeError(f"capture is missing denoising/CFG coordinates: {missing[:8]}")
        bad = {
            f"s{step:02d}_cfg{cfg}": count
            for (step, cfg), count in self.calls_by_step_cfg.items()
            if count != 100
        }
        if bad:
            raise RuntimeError(f"expected 100 captured heads per step/CFG: {bad}")
        maps: dict[str, np.ndarray] = {}
        for window, steps in WINDOWS.items():
            expected = len(steps) * 2 * 100
            if not np.all(self.counts[window] == expected):
                raise RuntimeError(
                    f"{window}: counts={self.counts[window].tolist()}, expected={expected}"
                )
            maps[window] = (
                self.sums[window] / self.counts[window][:, None, None]
            ).astype(np.float32)
        audit = {
            "physical_top100_heads": 100,
            "denoising_steps": 40,
            "cfg_branches": 2,
            "head_events_all40": 8000,
            "query_token_counts_by_latent": [len(rows) for rows in self.query_rows_by_time],
            "window_head_events_per_latent": {
                name: int(self.counts[name][0]) for name in WINDOWS
            },
        }
        return maps, audit


class PassiveCaptureAblator(AttentionMatrixAblator):
    """Install model-step bookkeeping and capture Q/K without intervention."""

    def __init__(self, *args, capture: SameTimeTop100Capture, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.capture = capture

    def _attention(self, q, k, v, original, block: int):
        if self.active and self.by_block.get(block):
            self.capture.record(q, k, block, self.current_step, self.current_cfg_call)
        return original(q, k, v)

    def audit(self) -> dict[str, Any]:
        if sorted(self.model_call_counts) != list(range(40)):
            raise RuntimeError(f"expected steps 0..39, got {sorted(self.model_call_counts)}")
        if any(count != 2 for count in self.model_call_counts.values()):
            raise RuntimeError(f"expected two CFG calls per step: {self.model_call_counts}")
        return {"model_call_counts": self.model_call_counts, "intervention": "none"}


def attach_capture(ablator, capture: SameTimeTop100Capture) -> None:
    """Decorate an intervention ablator without changing its math or install order."""
    original_attention = ablator._attention

    def decorated(self, q, k, v, original, block: int):
        if self.active and self.by_block.get(block):
            capture.record(q, k, block, self.current_step, self.current_cfg_call)
        return original_attention(q, k, v, original, block)

    ablator._attention = MethodType(decorated, ablator)


def _full_mask_rows_by_time(partition) -> list[list[int]]:
    rows: list[list[int]] = []
    for time_index in range(GRID[0]):
        values = {
            int(value)
            for signature_rows in partition.signature_rows_by_time.values()
            for value in signature_rows[time_index]
        }
        rows.append(sorted(values))
    return rows


def _build_intervention(
    *,
    pipe,
    variant: Variant,
    entries: list[dict],
    points: np.ndarray,
    region_slices: dict[str, slice],
    tracks: np.ndarray,
    anchors: np.ndarray,
    partition,
    capture: SameTimeTop100Capture,
):
    common = dict(
        pipe=pipe,
        entries=entries,
        query_points=points,
        region_slices=region_slices,
        pixel_hw=(704, 1280),
        target_scope="single_object",
        mask_mode=TIME_SCOPE_TO_MASK["all_time"],
        region="object_A",
    )
    if variant.alpha is None:
        return PassiveCaptureAblator(**common, capture=capture)
    klass = (
        SAM2FullMaskM1DirectScalingAblator
        if variant.token_source == "sam2_full_mask"
        else M1DirectScalingAblator
    )
    extra = {"partition": partition} if variant.token_source == "sam2_full_mask" else {}
    ablator = klass(
        **common,
        tracks=tracks,
        anchor_frames=anchors,
        record_dose=True,
        alpha=float(variant.alpha),
        time_scope="all_time",
        denoise_start=0,
        denoise_end=39,
        **extra,
    )
    attach_capture(ablator, capture)
    return ablator


def _video_uint8(video: np.ndarray) -> np.ndarray:
    values = np.asarray(video)
    if values.dtype == np.uint8:
        return values
    if np.issubdtype(values.dtype, np.floating):
        if float(values.max()) <= 1.5:
            values = values * 255.0
        return np.clip(np.rint(values), 0, 255).astype(np.uint8)
    return np.clip(values, 0, 255).astype(np.uint8)


def _decoded_video(path: Path) -> np.ndarray:
    frames = iio.imread(path)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"unexpected video shape for {path}: {frames.shape}")
    return frames.astype(np.uint8)


def _replay_audit(replay: Path, source: Path) -> dict[str, Any]:
    replay_frames = _decoded_video(replay)
    source_frames = _decoded_video(source)
    count = min(len(replay_frames), len(source_frames))
    if replay_frames.shape[1:] != source_frames.shape[1:] or count != 49:
        raise RuntimeError(
            f"replay/source video mismatch: {replay_frames.shape} vs {source_frames.shape}"
        )
    delta = replay_frames[:count].astype(np.float32) - source_frames[:count].astype(np.float32)
    mae = float(np.abs(delta).mean())
    max_abs = int(np.abs(delta).max())
    return {
        "source_video": str(source),
        "source_sha256": sha256_file(source),
        "replay_video": str(replay),
        "replay_sha256": sha256_file(replay),
        "sha256_equal": sha256_file(replay) == sha256_file(source),
        "decoded_frame_mae_0_255": mae,
        "decoded_frame_max_abs_0_255": max_abs,
        "frame_count": count,
    }


def _draw_query_cells(frame: np.ndarray, rows: list[int], time_index: int) -> None:
    cell_h = frame.shape[0] / GRID[1]
    cell_w = frame.shape[1] / GRID[2]
    offset = time_index * FRAME_TOKEN_COUNT
    for row in rows:
        local = row - offset
        y, x = divmod(local, GRID[2])
        x0, x1 = int(round(x * cell_w)), int(round((x + 1) * cell_w)) - 1
        y0, y1 = int(round(y * cell_h)), int(round((y + 1) * cell_h)) - 1
        cv2.rectangle(frame, (x0, y0), (x1, y1), (245, 245, 245), 1, cv2.LINE_AA)


def _overlay_frame(
    rgb: np.ndarray,
    heat: np.ndarray,
    *,
    scale: float,
    title: str,
    rows: list[int],
    time_index: int,
) -> np.ndarray:
    normalized = np.clip(heat / max(scale, 1e-12), 0.0, 1.0)
    display = np.sqrt(normalized)
    resized = cv2.resize(
        display.astype(np.float32),
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    color = cv2.applyColorMap(np.rint(resized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    alpha = (0.72 * resized)[..., None]
    blended = np.clip(rgb.astype(np.float32) * (1.0 - alpha) + color * alpha, 0, 255).astype(np.uint8)
    _draw_query_cells(blended, rows, time_index)
    cv2.rectangle(blended, (0, 0), (blended.shape[1], 44), (9, 18, 23), -1)
    cv2.putText(
        blended,
        title,
        (12, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 248, 244),
        1,
        cv2.LINE_AA,
    )
    return blended


def render_overlays(output_root: Path, variant_rows: tuple[Variant, ...], rows_by_time: list[list[int]]) -> dict:
    captures: dict[str, dict[str, np.ndarray]] = {}
    for variant in variant_rows:
        path = output_root / "captures" / variant.id / "top100_same_time_maps.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as arrays:
            captures[variant.id] = {window: arrays[window].astype(np.float32) for window in WINDOWS}
    if not captures:
        raise RuntimeError("no completed capture maps are available to render")

    scales: dict[str, float] = {}
    for window in WINDOWS:
        values = np.concatenate([maps[window].reshape(-1) for maps in captures.values()])
        positive = values[values > 0]
        if not len(positive):
            raise RuntimeError(f"{window}: attention maps are identically zero")
        scales[window] = float(np.quantile(positive, 0.995))

    records = []
    for variant in variant_rows:
        if variant.id not in captures:
            records.append({"variant_id": variant.id, "ready": False})
            continue
        capture_root = output_root / "captures" / variant.id
        capture_manifest = json.loads(
            (capture_root / "manifest.json").read_text(encoding="utf-8")
        )
        replay_video = capture_root / "replay.mp4"
        if not replay_video.is_file():
            raise FileNotFoundError(f"capture replay is missing: {replay_video}")
        frames = _decoded_video(replay_video)
        images: dict[str, list[str]] = {}
        for window in WINDOWS:
            directory = output_root / "overlays" / window / variant.id
            directory.mkdir(parents=True, exist_ok=True)
            names = []
            for time_index, pixel_frame in enumerate(ANCHOR_FRAMES):
                name = f"latent_{time_index:02d}__frame_{pixel_frame:02d}.jpg"
                title = (
                    f"L{time_index:02d} / F{pixel_frame:02d}  "
                    f"|R_t|={len(rows_by_time[time_index])}  "
                    f"mean={captures[variant.id][window][time_index].mean():.3e}"
                )
                overlay = _overlay_frame(
                    frames[pixel_frame],
                    captures[variant.id][window][time_index],
                    scale=scales[window],
                    title=title,
                    rows=rows_by_time[time_index],
                    time_index=time_index,
                )
                target = directory / name
                iio.imwrite(target, overlay, quality=90)
                names.append(name)
            images[window] = names
        records.append(
            {
                "variant_id": variant.id,
                "label": variant.label,
                "token_source": variant.token_source,
                "alpha": variant.alpha,
                "source_video": str(variant.video),
                "source_video_sha256": sha256_file(variant.video),
                "display_video": str(replay_video),
                "display_video_sha256": sha256_file(replay_video),
                "replay_audit": capture_manifest.get("replay_audit", {}),
                "ready": True,
                "images": images,
            }
        )
    payload = {
        "protocol": "phase_b_seed90094_top100_same_time_attention_overlay_v1",
        "case": CASE,
        "seed": SEED,
        "default_window": "all40",
        "windows": [
            {"id": name, "label": WINDOW_LABELS[name], "steps": list(steps)}
            for name, steps in WINDOWS.items()
        ],
        "shared_color_scale_q995": scales,
        "color_scale_scope": "shared across all five variants and all 13 latent frames within each denoising window",
        "anchor_pixel_frames": list(ANCHOR_FRAMES),
        "latent_grid": list(GRID),
        "query_token_counts_by_latent": [len(rows) for rows in rows_by_time],
        "formula": (
            "H_W,t(x,y)=mean_{s in W,cfg,(l,h) in latest3350 Top100,q in frozen "
            "Baseline SAM2 R_t} softmax(QK^T/sqrt(128))[q,K_t(x,y)]"
        ),
        "softmax_key_domain": "all 13x22x40 keys; K_t is sliced only after softmax",
        "query_definition": "same frozen Baseline SAM2 object_A full-mask latent tokens for all five rows",
        "cfg_aggregation": "mean conditional and unconditional",
        "head_aggregation": "equal mean of 100 physical layer-heads",
        "query_aggregation": "equal mean of all full-mask object_A query tokens in each latent frame",
        "records": records,
    }
    _atomic_json(output_root / "overlay_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--sam2-full-mask-root", type=Path, default=FULL_MASK_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--only", choices=("all", "render", "baseline", "sparse_a0p1", "full_a0p1", "sparse_a0p25", "full_a0p25"), default="all")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    sample = next(
        (
            row
            for row in manifest.get("samples", [])
            if str(row.get("case")) == CASE and int(row.get("seed", -1)) == SEED
        ),
        None,
    )
    if sample is None:
        raise KeyError(f"manifest has no {CASE}/seed {SEED}")
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest, ranking, allow_tagged_snapshot_change=True
    )
    entries = selected_head_entries(
        ranking_entries, "top100", dict(ranking.get("head_scopes") or {})
    )
    if len(entries) != 100:
        raise RuntimeError(f"expected latest3350 Top100, got {len(entries)} heads")
    rows = variants(sample)
    for variant in rows:
        if not variant.video.is_file():
            raise FileNotFoundError(f"missing source video: {variant.video}")

    partition, full_mask_path = load_full_mask_partition(
        args.sam2_full_mask_root, CASE, SEED, Path(str(sample["baseline_video"]))
    )
    rows_by_time = _full_mask_rows_by_time(partition)
    args.output_root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": "phase_b_seed90094_top100_same_time_attention_overlay_v1",
        "case": CASE,
        "seed": SEED,
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": sha256_file(args.manifest_path),
        "head_ranking_path": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "full_mask_cache": str(full_mask_path),
        "full_mask_cache_sha256": sha256_file(full_mask_path),
        "query_token_counts_by_latent": [len(values) for values in rows_by_time],
        "variants": [variant.__dict__ | {"video": str(variant.video)} for variant in rows],
    }
    _atomic_json(args.output_root / "protocol.json", protocol)

    if args.only == "render":
        render_overlays(args.output_root, rows, rows_by_time)
        print(args.output_root / "overlay_manifest.json")
        return

    case_lookup = {case.key: case for case in CASES}
    _, cache_dir, payload, wan_args, image = generation_inputs(sample, case_lookup, SEED)
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    points, query_regions = object_queries(cache)
    region_slices = {region.region_name: point_slice for region, point_slice in query_regions}
    track_path = tracks_root(args.tracks_root, CASE, SEED) / "tracks.npz"
    if not track_path.is_file():
        raise FileNotFoundError(f"frozen tracks missing: {track_path}")
    with np.load(track_path, allow_pickle=False) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)
    if tuple(anchors.tolist()) != ANCHOR_FRAMES:
        raise RuntimeError(f"unexpected latent anchors: {anchors.tolist()}")

    selected_rows = rows if args.only == "all" else tuple(row for row in rows if row.id == args.only)
    pending = [
        row
        for row in selected_rows
        if args.overwrite
        or not (args.output_root / "captures" / row.id / "complete.json").is_file()
    ]
    if pending:
        wan_args.cfg_scale = 5.0
        wan_args.sampling_steps = 40
        pipe_wrapper = build_wan_ti2v_pipeline(wan_args)
        for index, variant in enumerate(pending, start=1):
            output = args.output_root / "captures" / variant.id
            output.mkdir(parents=True, exist_ok=True)
            (output / "complete.json").unlink(missing_ok=True)
            (output / "error.txt").unlink(missing_ok=True)
            print(f"[{index}/{len(pending)}] capture {variant.id}", flush=True)
            try:
                capture = SameTimeTop100Capture(entries, rows_by_time)
                ablator = _build_intervention(
                    pipe=pipe_wrapper.pipe,
                    variant=variant,
                    entries=entries,
                    points=points,
                    region_slices=region_slices,
                    tracks=tracks,
                    anchors=anchors,
                    partition=partition,
                    capture=capture,
                )
                ablator.install()
                try:
                    video = _run_pipe_once(
                        pipe=pipe_wrapper,
                        prompt=str(payload["input_caption"]),
                        negative_prompt=str(wan_args.negative_prompt),
                        seed=SEED,
                        input_image=image,
                        height=704,
                        width=1280,
                        num_frames=49,
                        cfg_scale=5.0,
                        num_inference_steps=40,
                        sample_shift=5.0,
                        sample_solver="unipc",
                        offload_model=False,
                    )
                finally:
                    ablator.remove()
                intervention_audit = ablator.audit()
                maps, capture_audit = capture.finalize()
                replay = output / "replay.mp4"
                temporary = output / "replay.tmp.mp4"
                save_video_np(_video_uint8(video), temporary, fps=30)
                temporary.replace(replay)
                replay_audit = _replay_audit(replay, variant.video)
                _atomic_npz(
                    output / "top100_same_time_maps.npz",
                    **maps,
                    query_token_counts_by_latent=np.asarray(
                        [len(values) for values in rows_by_time], dtype=np.int32
                    ),
                    anchor_pixel_frames=np.asarray(ANCHOR_FRAMES, dtype=np.int32),
                )
                result = {
                    "variant_id": variant.id,
                    "label": variant.label,
                    "token_source": variant.token_source,
                    "alpha": variant.alpha,
                    "source_video": str(variant.video),
                    "intervention_audit": intervention_audit,
                    "capture_audit": capture_audit,
                    "replay_audit": replay_audit,
                }
                _atomic_json(output / "manifest.json", result)
                _atomic_json(
                    output / "complete.json",
                    {
                        "variant_id": variant.id,
                        "capture_head_events": capture_audit["head_events_all40"],
                        "decoded_replay_mae": replay_audit["decoded_frame_mae_0_255"],
                    },
                )
                del video, ablator, capture
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                raise
        del pipe_wrapper
        gc.collect()
        torch.cuda.empty_cache()

    render_overlays(args.output_root, rows, rows_by_time)
    print(args.output_root / "overlay_manifest.json")


if __name__ == "__main__":
    main()
