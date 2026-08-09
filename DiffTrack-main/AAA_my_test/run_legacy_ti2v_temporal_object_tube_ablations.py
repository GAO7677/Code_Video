#!/usr/bin/env python3
"""Run Top100 attention ablations on frozen all-latent object-token tubes."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
for path in (ROOT, CODE_ROOT, COTRACKER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import build_wan_ti2v_pipeline  # noqa: E402
from AAA_my_test.run_legacy_ti2v_firstlatent_pck_worker import (  # noqa: E402
    load_cotracker,
    object_queries,
    run_cotracker,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    MASK_BLOCKS,
    MATRIX_MASKS,
    AttentionMatrixAblator,
    build_args,
    generate_video,
    generation_inputs,
    sample_inputs,
    variant_id,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402
from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import save_video_np  # noqa: E402


DEFAULT_CASE = "0613pybullet_sample_001460_w002"
DEFAULT_MANIFEST = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/cases.json"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_MANIFEST.parent / "attention_matrix_ablations_temporal_tube_v1"
TOP_N = 100
MASK_MODES = MATRIX_MASKS + ("literal_kv_zero",)
PROTOCOL = "attention_matrix_ablation_temporal_object_tube_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_tasks(sample: dict) -> list[dict]:
    regions = [
        str(row["region_name"])
        for row in sample["regions"]
        if row.get("region_type") == "object"
    ]
    targets = [("single_object", region) for region in regions]
    targets.append(("all_objects", None))
    return [
        {
            "case": str(sample["case"]),
            "seed": int(sample["seed"]),
            "target_scope": target_scope,
            "mask_mode": mask_mode,
            "region": region,
            "top_n": TOP_N,
        }
        for target_scope, region in targets
        for mask_mode in MASK_MODES
    ]


def task_root(task: dict, output_root: Path) -> Path:
    return (
        output_root
        / str(task["case"])
        / f"seed_{int(task['seed']):05d}"
        / variant_id(
            str(task["target_scope"]),
            str(task["mask_mode"]),
            int(task["top_n"]),
            task.get("region"),
        )
    )


def tracks_root(output_root: Path, case: str, seed: int) -> Path:
    return output_root / case / f"seed_{seed:05d}" / "frozen_baseline_tracks"


def atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def prepare_tracks(
    sample: dict,
    case_lookup: dict,
    output_root: Path,
    device: str,
    overwrite: bool,
) -> Path:
    case, seed = str(sample["case"]), int(sample["seed"])
    output = tracks_root(output_root, case, seed)
    track_path = output / "tracks.npz"
    manifest_path = output / "manifest.json"
    if track_path.is_file() and manifest_path.is_file() and not overwrite:
        return track_path

    output.mkdir(parents=True, exist_ok=True)
    json_path, cache_dir = sample_inputs(sample, case_lookup)
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    if int(cache.metadata.get("query_context_frame", -1)) != 0:
        raise RuntimeError(f"{case}: expected query frame 0 cache")
    query_points, query_regions = object_queries(cache)
    baseline_video = Path(str(sample["baseline_video"]))
    frames = iio.imread(baseline_video)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise RuntimeError(f"unexpected baseline video shape: {frames.shape}")

    model = load_cotracker(device)
    try:
        tracks, visibility = run_cotracker(model, frames, query_points, device)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
    latent_frames = 13
    anchors = np.arange(latent_frames, dtype=np.int64) * 4
    if int(anchors[-1]) >= len(frames):
        anchors = np.rint(np.linspace(0, len(frames) - 1, latent_frames)).astype(np.int64)
    if not np.isfinite(tracks[anchors]).all():
        raise RuntimeError("CoTracker returned non-finite coordinates at a latent anchor")

    starts = np.asarray([part.start for _, part in query_regions], dtype=np.int32)
    ends = np.asarray([part.stop for _, part in query_regions], dtype=np.int32)
    names = np.asarray([region.region_name for region, _ in query_regions])
    atomic_npz(
        track_path,
        tracks=tracks.astype(np.float32),
        visibility=visibility.astype(np.bool_),
        anchor_pixel_frames=anchors,
        query_points=query_points.astype(np.float32),
        region_names=names,
        point_starts=starts,
        point_ends=ends,
        source_video=np.asarray(str(baseline_video)),
        source_json=np.asarray(str(json_path)),
        pixel_height=np.int32(frames.shape[1]),
        pixel_width=np.int32(frames.shape[2]),
        seed=np.int32(seed),
    )
    metadata = {
        "case": case,
        "seed": seed,
        "source_video": str(baseline_video),
        "source_json": str(json_path),
        "query_cache_dir": str(cache_dir),
        "tracker": "CoTracker3 offline scaled checkpoint",
        "query_pixel_frame": 0,
        "latent_anchor_pixel_frames": anchors.tolist(),
        "point_count": int(len(query_points)),
        "regions": [
            {
                "region_name": str(name),
                "point_start": int(start),
                "point_end": int(end),
                "anchor_visibility_rate": float(visibility[anchors, start:end].mean()),
            }
            for name, start, end in zip(names.tolist(), starts.tolist(), ends.tolist())
        ],
        "selection_policy": (
            "use each finite CoTracker-predicted coordinate at all 13 latent anchors; "
            "visibility is retained for audit but does not delete an anchor token"
        ),
        "frozen_before_intervention": True,
    }
    manifest_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return track_path


class TemporalObjectTubeAblator(AttentionMatrixAblator):
    """Use tracked object tokens at every latent time as the R partition."""

    def __init__(self, *args, tracks: np.ndarray, anchor_frames: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.tracks = torch.as_tensor(tracks, dtype=torch.float32)
        self.anchor_frames = torch.as_tensor(anchor_frames, dtype=torch.long)
        if self.tracks.ndim != 3 or self.tracks.shape[-1] != 2:
            raise ValueError(f"expected tracks [frames, points, 2], got {self.tracks.shape}")
        if self.tracks.shape[1] != self.query_points.shape[0]:
            raise ValueError("track point count does not match cached object query points")
        self.query_token_indices_by_latent_frame: list[list[int]] | None = None

    def _rows(self, device: torch.device) -> torch.Tensor | None:
        if self.current_grid is None:
            raise RuntimeError("attention grid is unavailable")
        time, height, width = self.current_grid
        if time != len(self.anchor_frames):
            raise RuntimeError(
                f"attention latent time {time} != tracked anchors {len(self.anchor_frames)}"
            )
        tracks = self.tracks[self.anchor_frames].to(device)
        if self.target_scope == "single_object":
            tracks = tracks[:, self.region_slices[str(self.region)]]
        x = torch.floor(tracks[..., 0] * width / self.pixel_width).long().clamp(0, width - 1)
        y = torch.floor(tracks[..., 1] * height / self.pixel_height).long().clamp(0, height - 1)
        spatial = y * width + x
        offsets = torch.arange(time, device=device, dtype=torch.long)[:, None] * height * width
        tokens_by_time = spatial + offsets
        rows = torch.unique(tokens_by_time.flatten(), sorted=True)
        values = [int(value) for value in rows.detach().cpu().tolist()]
        by_time = [
            [int(value) for value in torch.unique(row, sorted=True).detach().cpu().tolist()]
            for row in tokens_by_time
        ]
        if self.query_token_indices is None:
            self.query_token_indices = values
            self.query_token_indices_by_latent_frame = by_time
        elif self.query_token_indices != values:
            raise RuntimeError("temporal object tube token mapping changed during generation")
        return rows

    def audit(self) -> dict:
        result = super().audit()
        result["query_token_indices_by_latent_frame"] = self.query_token_indices_by_latent_frame
        result["latent_frame_token_counts"] = [
            len(row) for row in (self.query_token_indices_by_latent_frame or [])
        ]
        return result


def process(
    pipe,
    manifest: dict,
    sample: dict,
    case_lookup: dict,
    task: dict,
    output_root: Path,
    track_path: Path,
    overwrite: bool,
) -> None:
    output = task_root(task, output_root)
    complete_path = output / "complete.json"
    ready = all(
        (output / name).is_file()
        for name in ("complete.json", "manifest.json", "generated.mp4")
    )
    if ready and not overwrite:
        print(f"skip {output.relative_to(output_root)}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)

    json_path, cache_dir, payload, args, image = generation_inputs(
        sample, case_lookup, int(task["seed"])
    )
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    points, query_regions = object_queries(cache)
    region_slices = {region.region_name: point_slice for region, point_slice in query_regions}
    with np.load(track_path) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)
    entries = manifest["entries"][:TOP_N]
    ablator = TemporalObjectTubeAblator(
        pipe.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        str(task["target_scope"]),
        str(task["mask_mode"]),
        task.get("region"),
        tracks=tracks,
        anchor_frames=anchors,
    )
    ablator.install()
    try:
        video = generate_video(pipe, payload, args, image, int(task["seed"]))
    finally:
        ablator.remove()
    audit = ablator.audit()
    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")

    metadata = {
        **task,
        "variant_id": variant_id(
            str(task["target_scope"]),
            str(task["mask_mode"]),
            int(task["top_n"]),
            task.get("region"),
        ),
        "protocol": PROTOCOL,
        "attention_definition": "A=softmax(QK^T/sqrt(d)); Y=A@V",
        "selected_token_definition": (
            "union of sparse object points tracked on the seed-matched baseline and "
            "mapped at all 13 latent anchors F00,F04,...,F48"
        ),
        "matrix_partition": {
            "R": "frozen all-latent object token tube",
            "C": "all tokens outside the selected temporal tube",
            "S": "A[R,R]",
            "I": "A[R,C] (C K/V -> R queries)",
            "O": "A[C,R] (R K/V -> C queries)",
        },
        "zeroed_matrix_blocks": list(MASK_BLOCKS[str(task["mask_mode"])]),
        "semantic_qkv_projection_intervention": str(task["mask_mode"]) == "literal_kv_zero",
        "post_mask_renormalization": False,
        "softmax_recomputed_after_k_intervention": str(task["mask_mode"]) == "literal_kv_zero",
        "implementation": (
            "literal selected temporal-tube K/V vectors set to zero before attention"
            if str(task["mask_mode"]) == "literal_kv_zero"
            else "post-softmax A@V block decomposition; column masks use exact V_R=0 equivalence"
        ),
        "trajectory_source": "CoTracker pseudo-GT on seed-matched no-intervention baseline",
        "trajectory_is_frozen_before_intervention": True,
        "trajectory_visibility_policy": (
            "all finite predicted anchor coordinates are used; visibility is audit-only"
        ),
        "latent_anchor_pixel_frames": anchors.tolist(),
        "tracks_npz": str(track_path),
        "input_json": str(json_path),
        "query_cache_dir": str(cache_dir),
        "selected_entries": entries,
        "denoising_steps": list(range(40)),
        "cfg_branches": ["conditional", "unconditional"],
        "audit": audit,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_path.write_text(
        json.dumps(
            {
                "case": task["case"],
                "seed": task["seed"],
                "variant_id": metadata["variant_id"],
                "protocol": PROTOCOL,
                "selected_temporal_tokens": len(audit["query_token_indices"]),
                "modified_head_events": audit["modified_head_events"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del video


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    sample = next(
        (
            row
            for row in manifest["samples"]
            if str(row["case"]) == args.case and int(row["seed"]) == args.seed
        ),
        None,
    )
    if sample is None:
        raise KeyError(f"case/seed not found in manifest: {args.case}/{args.seed}")
    if len(manifest.get("entries", [])) < TOP_N:
        raise RuntimeError("manifest does not contain the frozen Top100 ranking")
    tasks = build_tasks(sample)
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be in [0, {len(tasks)})")
        tasks = [tasks[args.task_index]]

    from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES

    case_lookup = {case.key: case for case in CASES}
    track_path = prepare_tracks(
        sample, case_lookup, args.output_root, str(args.device), bool(args.overwrite)
    )
    pipe = build_wan_ti2v_pipeline(build_args(int(sample["seed"])))
    for index, task in enumerate(tasks, start=1):
        output = task_root(task, args.output_root)
        print(f"[{index}/{len(tasks)}] start {output.relative_to(args.output_root)}", flush=True)
        try:
            process(
                pipe,
                manifest,
                sample,
                case_lookup,
                task,
                args.output_root,
                track_path,
                bool(args.overwrite),
            )
        except Exception:
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(traceback.format_exc(), flush=True)
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {output.relative_to(args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
