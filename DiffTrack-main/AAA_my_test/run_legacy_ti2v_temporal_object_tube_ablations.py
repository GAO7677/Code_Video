#!/usr/bin/env python3
"""Run ranked-head attention ablations on frozen all-latent object-token tubes."""

from __future__ import annotations

import argparse
import gc
import json
import re
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
DEFAULT_HEAD_RANKING = DEFAULT_MANIFEST.parent / "pck_head_scopes_s039_frozen134.json"
TOP_N = 100
HEAD_SCOPES = ("top100", "bottom100", "all720")
HEAD_SCOPE_COUNTS = {"top100": 100, "bottom100": 100, "all720": 720}
TEMPORAL_DIRECTIONAL_SPECS = {
    "self_future": {
        "id": "M1-future",
        "base_block": "S",
        "target_partition": "R",
        "source_partition": "R",
        "direction": "future",
    },
    "incoming_future": {
        "id": "M2-future",
        "base_block": "I",
        "target_partition": "R",
        "source_partition": "C",
        "direction": "future",
    },
    "outgoing_future": {
        "id": "M3-future",
        "base_block": "O",
        "target_partition": "C",
        "source_partition": "R",
        "direction": "future",
    },
    "self_same": {
        "id": "M1-same",
        "base_block": "S",
        "target_partition": "R",
        "source_partition": "R",
        "direction": "same",
    },
    "incoming_same": {
        "id": "M2-same",
        "base_block": "I",
        "target_partition": "R",
        "source_partition": "C",
        "direction": "same",
    },
    "outgoing_same": {
        "id": "M3-same",
        "base_block": "O",
        "target_partition": "C",
        "source_partition": "R",
        "direction": "same",
    },
    "self_past": {
        "id": "M1-past",
        "base_block": "S",
        "target_partition": "R",
        "source_partition": "R",
        "direction": "past",
    },
    "incoming_past": {
        "id": "M2-past",
        "base_block": "I",
        "target_partition": "R",
        "source_partition": "C",
        "direction": "past",
    },
    "outgoing_past": {
        "id": "M3-past",
        "base_block": "O",
        "target_partition": "C",
        "source_partition": "R",
        "direction": "past",
    },
}
TEMPORAL_DIRECTIONAL_MODES = tuple(TEMPORAL_DIRECTIONAL_SPECS)
MASK_MODES = MATRIX_MASKS + ("literal_kv_zero",) + TEMPORAL_DIRECTIONAL_MODES
PROTOCOL = "attention_matrix_ablation_temporal_object_tube_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="process every manifest sample; workers are sharded by case/seed sample",
    )
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument(
        "--ranking-tag",
        default="",
        help="optional output-ID suffix for a different frozen ranking snapshot",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--tracks-root",
        type=Path,
        default=None,
        help="optional root containing reusable frozen_baseline_tracks; defaults to output-root",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--mask-modes", nargs="+", choices=MASK_MODES, default=None)
    parser.add_argument(
        "--head-scopes",
        nargs="+",
        default=["top100"],
        help=(
            "head scopes declared by the ranking JSON; built-ins are top100, "
            "bottom100, all720 and custom pair scopes such as layer-matched Random100"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="optional seed filter applied before worker sharding",
    )
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-dose",
        action="store_true",
        help=(
            "record exact removed attention mass and A@V contribution/output norms "
            "for M1/M2/M3 in dose_metrics.npz"
        ),
    )
    return parser.parse_args()


def temporal_variant_id(task: dict) -> str:
    mask_mode = str(task["mask_mode"])
    head_scope = str(task.get("head_scope", "top100"))
    ranking_tag = str(task.get("ranking_tag") or "")
    if (
        head_scope == "top100"
        and not ranking_tag
        and mask_mode not in TEMPORAL_DIRECTIONAL_MODES
    ):
        return variant_id(
            str(task["target_scope"]),
            mask_mode,
            int(task["top_n"]),
            task.get("region"),
        )
    target = (
        str(task["region"])
        if str(task["target_scope"]) == "single_object"
        else "all_objects"
    )
    suffix = head_scope if not ranking_tag else f"{head_scope}_{ranking_tag}"
    return f"{task['target_scope']}__{target}__{mask_mode}__{suffix}"


def build_tasks(
    sample: dict,
    head_scopes: list[str] | tuple[str, ...] = ("top100",),
    ranking_tag: str = "",
    head_scope_counts: dict[str, int] | None = None,
) -> list[dict]:
    counts = HEAD_SCOPE_COUNTS if head_scope_counts is None else head_scope_counts
    regions = [
        str(row["region_name"])
        for row in sample["regions"]
        if row.get("region_type") == "object"
    ]
    targets = [("single_object", region) for region in regions]
    if len(regions) > 1:
        targets.append(("all_objects", None))
    return [
        {
            "case": str(sample["case"]),
            "seed": int(sample["seed"]),
            "target_scope": target_scope,
            "mask_mode": mask_mode,
            "region": region,
            "top_n": counts[head_scope],
            "head_scope": head_scope,
            "ranking_tag": ranking_tag,
        }
        for head_scope in head_scopes
        for target_scope, region in targets
        for mask_mode in MASK_MODES
    ]


def validate_head_ranking(
    manifest: dict,
    ranking: dict,
    allow_tagged_snapshot_change: bool = False,
) -> list[dict]:
    entries = list(ranking.get("entries") or [])
    if len(entries) != 720:
        raise RuntimeError(f"head ranking must contain 720 entries, got {len(entries)}")
    pairs = [(int(row["block"]), int(row["head"])) for row in entries]
    if len(set(pairs)) != 720 or set(pairs) != {
        (block, head) for block in range(30) for head in range(24)
    }:
        raise RuntimeError("head ranking is not a one-to-one 30 x 24 layer-head ranking")
    if entries[:TOP_N] != list(manifest.get("entries") or [])[:TOP_N]:
        if not allow_tagged_snapshot_change:
            raise RuntimeError(
                "head ranking Top100 does not match the frozen experiment manifest"
            )
        source_path = Path(str(ranking.get("source_manifest") or ""))
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                "tagged head ranking must reference a readable source_manifest"
            ) from exc
        if entries[:TOP_N] != list(source.get("entries") or [])[:TOP_N]:
            raise RuntimeError(
                "tagged head ranking Top100 does not match its source_manifest"
            )
    return entries


def selected_head_entries(
    ranking_entries: list[dict],
    head_scope: str,
    head_scope_definitions: dict[str, dict] | None = None,
) -> list[dict]:
    if head_scope == "top100":
        return ranking_entries[:100]
    if head_scope == "bottom100":
        return ranking_entries[-100:]
    if head_scope == "all720":
        return ranking_entries
    definition = (head_scope_definitions or {}).get(head_scope)
    if definition and definition.get("pairs") is not None:
        lookup = {
            (int(row["block"]), int(row["head"])): row for row in ranking_entries
        }
        pairs = [(int(pair[0]), int(pair[1])) for pair in definition["pairs"]]
        if len(pairs) != len(set(pairs)):
            raise RuntimeError(f"{head_scope}: duplicate physical heads")
        try:
            return [lookup[pair] for pair in pairs]
        except KeyError as exc:
            raise RuntimeError(f"{head_scope}: unknown layer-head pair {exc.args[0]}") from exc
    raise ValueError(f"unknown head scope: {head_scope}")


def head_scope_counts(ranking: dict) -> dict[str, int]:
    result = dict(HEAD_SCOPE_COUNTS)
    for name, definition in (ranking.get("head_scopes") or {}).items():
        if definition.get("pairs") is not None:
            result[str(name)] = len(definition["pairs"])
        elif "rank_start" in definition and "rank_end" in definition:
            result[str(name)] = int(definition["rank_end"]) - int(definition["rank_start"]) + 1
    return result


def task_root(task: dict, output_root: Path) -> Path:
    return (
        output_root
        / str(task["case"])
        / f"seed_{int(task['seed']):05d}"
        / temporal_variant_id(task)
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


def temporal_directional_groups(
    tokens_by_time: list[list[int]],
    frame_token_count: int,
    mask_mode: str,
    device: torch.device,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Return (target query rows, source V rows) for one strict time direction."""
    spec = TEMPORAL_DIRECTIONAL_SPECS[mask_mode]
    partition_rows: dict[str, list[torch.Tensor]] = {"R": [], "C": []}
    for time_index, values in enumerate(tokens_by_time):
        frame_start = time_index * frame_token_count
        frame_rows = torch.arange(
            frame_start, frame_start + frame_token_count, device=device, dtype=torch.long
        )
        r_rows = torch.as_tensor(values, device=device, dtype=torch.long)
        if r_rows.numel():
            keep = ~torch.isin(frame_rows, r_rows)
            c_rows = frame_rows[keep]
        else:
            c_rows = frame_rows
        partition_rows["R"].append(r_rows)
        partition_rows["C"].append(c_rows)

    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    time_count = len(tokens_by_time)
    for target_time in range(time_count):
        if spec["direction"] == "future":
            source_times = range(target_time)
        elif spec["direction"] == "past":
            source_times = range(target_time + 1, time_count)
        elif spec["direction"] == "same":
            source_times = (target_time,)
        else:
            raise ValueError(f"unsupported temporal direction: {spec['direction']}")
        source_parts = [
            partition_rows[str(spec["source_partition"])][source_time]
            for source_time in source_times
        ]
        target_rows = partition_rows[str(spec["target_partition"])][target_time]
        if not source_parts or not target_rows.numel():
            continue
        source_rows = torch.cat(source_parts)
        if source_rows.numel():
            groups.append((target_rows, source_rows))
    return groups


def apply_temporal_directional_ablation(
    output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    original,
    heads: tuple[int, ...] | list[int],
    num_heads: int,
    groups: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[int, int, int]:
    """Subtract exact post-softmax A@V contributions for directional token groups."""
    output_heads = AttentionMatrixAblator._head_view(output, num_heads)
    source_heads = AttentionMatrixAblator._head_view(v, num_heads)
    selected_v = torch.zeros_like(v)
    selected_heads = AttentionMatrixAblator._head_view(selected_v, num_heads)
    affected_rows = 0
    zeroed_entries_per_head = 0
    for target_rows, source_rows in groups:
        selected_heads.zero_()
        for head in heads:
            selected_heads[:, source_rows, head, :] = source_heads[:, source_rows, head, :]
        contribution = original(q[:, target_rows, :], k, selected_v)
        contribution_heads = AttentionMatrixAblator._head_view(contribution, num_heads)
        for head in heads:
            output_heads[:, target_rows, head, :] = (
                output_heads[:, target_rows, head, :] - contribution_heads[:, :, head, :]
            )
        affected_rows += int(target_rows.numel())
        zeroed_entries_per_head += int(target_rows.numel() * source_rows.numel())
    return len(groups), affected_rows, zeroed_entries_per_head


class TemporalObjectTubeAblator(AttentionMatrixAblator):
    """Use tracked object tokens at every latent time as the R partition."""

    def __init__(self, *args, tracks: np.ndarray, anchor_frames: np.ndarray, **kwargs):
        super().__init__(*args, extra_mask_modes=TEMPORAL_DIRECTIONAL_MODES, **kwargs)
        self.tracks = torch.as_tensor(tracks, dtype=torch.float32)
        self.anchor_frames = torch.as_tensor(anchor_frames, dtype=torch.long)
        if self.tracks.ndim != 3 or self.tracks.shape[-1] != 2:
            raise ValueError(f"expected tracks [frames, points, 2], got {self.tracks.shape}")
        if self.tracks.shape[1] != self.query_points.shape[0]:
            raise ValueError("track point count does not match cached object query points")
        self.query_token_indices_by_latent_frame: list[list[int]] | None = None
        self.temporal_zeroed_entries_per_head: int | None = None
        self.temporal_auxiliary_attention_calls = 0

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

    def _attention(self, q, k, v, original, block: int):
        if self.mask_mode not in TEMPORAL_DIRECTIONAL_MODES:
            return super()._attention(q, k, v, original, block)
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        if self.current_grid is None:
            raise RuntimeError("attention grid is unavailable")
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")

        self._rows(q.device)
        if not self.query_token_indices_by_latent_frame:
            raise RuntimeError("temporal object tube rows were not resolved")
        _, height, width = self.current_grid
        groups = temporal_directional_groups(
            self.query_token_indices_by_latent_frame,
            height * width,
            self.mask_mode,
            q.device,
        )
        output = original(q, k, v)
        auxiliary_calls, affected_rows, zeroed_entries = apply_temporal_directional_ablation(
            output, q, k, v, original, heads, num_heads, groups
        )
        if self.temporal_zeroed_entries_per_head is None:
            self.temporal_zeroed_entries_per_head = zeroed_entries
        elif self.temporal_zeroed_entries_per_head != zeroed_entries:
            raise RuntimeError("directional temporal mask size changed during generation")
        self.auxiliary_attention_calls += auxiliary_calls
        self.temporal_auxiliary_attention_calls += auxiliary_calls
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += output.shape[0] * affected_rows * len(heads)
        return output

    def audit(self) -> dict:
        result = super().audit()
        result["query_token_indices_by_latent_frame"] = self.query_token_indices_by_latent_frame
        result["latent_frame_token_counts"] = [
            len(row) for row in (self.query_token_indices_by_latent_frame or [])
        ]
        if self.mask_mode in TEMPORAL_DIRECTIONAL_MODES:
            result["temporal_directional_spec"] = TEMPORAL_DIRECTIONAL_SPECS[self.mask_mode]
            result["temporal_zeroed_entries_per_head"] = self.temporal_zeroed_entries_per_head
            result["temporal_auxiliary_attention_calls"] = (
                self.temporal_auxiliary_attention_calls
            )
        return result


def process(
    pipe,
    manifest: dict,
    ranking_entries: list[dict],
    head_scope_definitions: dict[str, dict],
    head_ranking_path: Path,
    sample: dict,
    case_lookup: dict,
    task: dict,
    output_root: Path,
    track_path: Path,
    overwrite: bool,
    record_dose: bool,
) -> None:
    output = task_root(task, output_root)
    complete_path = output / "complete.json"
    required_outputs = ["complete.json", "manifest.json", "generated.mp4"]
    if record_dose:
        required_outputs.append("dose_metrics.npz")
    ready = all((output / name).is_file() for name in required_outputs)
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
    head_scope = str(task.get("head_scope", "top100"))
    entries = selected_head_entries(ranking_entries, head_scope, head_scope_definitions)
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
        record_dose=record_dose,
    )
    ablator.install()
    try:
        video = generate_video(pipe, payload, args, image, int(task["seed"]))
    finally:
        ablator.remove()
    audit = ablator.audit()
    dose_path = output / "dose_metrics.npz"
    if record_dose:
        atomic_npz(dose_path, **ablator.dose_arrays())
    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")

    metadata = {
        **task,
        "variant_id": temporal_variant_id(task),
        "protocol": (
            "attention_matrix_ablation_temporal_direction_v1"
            if str(task["mask_mode"]) in TEMPORAL_DIRECTIONAL_MODES
            else PROTOCOL
        ),
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
        "zeroed_matrix_blocks": (
            [
                f"{TEMPORAL_DIRECTIONAL_SPECS[str(task['mask_mode'])]['base_block']}_"
                f"{TEMPORAL_DIRECTIONAL_SPECS[str(task['mask_mode'])]['direction']}"
            ]
            if str(task["mask_mode"]) in TEMPORAL_DIRECTIONAL_MODES
            else list(MASK_BLOCKS[str(task["mask_mode"])])
        ),
        "temporal_directional_spec": TEMPORAL_DIRECTIONAL_SPECS.get(
            str(task["mask_mode"])
        ),
        "semantic_qkv_projection_intervention": str(task["mask_mode"]) == "literal_kv_zero",
        "post_mask_renormalization": False,
        "softmax_recomputed_after_k_intervention": str(task["mask_mode"]) == "literal_kv_zero",
        "implementation": (
            "literal selected temporal-tube K/V vectors set to zero before attention"
            if str(task["mask_mode"]) == "literal_kv_zero"
            else (
                "post-softmax A@V temporal contribution subtraction; strict "
                "t_query>t_key for future, t_query<t_key for past, and "
                "t_query=t_key for same"
                if str(task["mask_mode"]) in TEMPORAL_DIRECTIONAL_MODES
                else "post-softmax A@V block decomposition; column masks use exact V_R=0 equivalence"
            )
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
        "head_scope": head_scope,
        "selected_head_count": len(entries),
        "head_ranking_snapshot": str(head_ranking_path),
        "denoising_steps": list(range(40)),
        "cfg_branches": ["conditional", "unconditional"],
        "audit": audit,
        "dose_metrics": (
            {
                "path": str(dose_path),
                "attention_mass": (
                    "mean over removed source attention probability for affected query rows"
                ),
                "removed_value_norm": (
                    "mean L2 norm of the exact removed sum_k A_qk V_k per affected query"
                ),
                "original_output_norm": (
                    "mean L2 norm of the original selected-head attention output on the same queries"
                ),
                "removed_to_output_ratio": "removed_value_norm / original_output_norm",
                "axes": ["denoising_step", "cfg_call", "block", "head"],
                "cfg_call_order": ["conditional_first_call", "unconditional_second_call"],
            }
            if record_dose
            else None
        ),
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
                "protocol": metadata["protocol"],
                "selected_temporal_tokens": len(audit["query_token_indices"]),
                "modified_head_events": audit["modified_head_events"],
                "dose_finite_events": audit["dose_finite_events"],
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
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    if args.ranking_tag and not re.fullmatch(r"[a-z0-9_]+", args.ranking_tag):
        raise ValueError("ranking-tag must contain only lowercase letters, digits, underscores")
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    if len(manifest.get("entries", [])) < TOP_N:
        raise RuntimeError("manifest does not contain the frozen Top100 ranking")
    head_ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest,
        head_ranking,
        allow_tagged_snapshot_change=bool(args.ranking_tag),
    )
    scope_definitions = dict(head_ranking.get("head_scopes") or {})
    scope_counts = head_scope_counts(head_ranking)
    unknown_scopes = sorted(set(args.head_scopes) - set(scope_counts))
    if unknown_scopes:
        raise ValueError(f"head scopes not declared by ranking JSON: {unknown_scopes}")
    if args.all_samples:
        samples = list(manifest["samples"])
    else:
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
        samples = [sample]
    if args.seeds is not None:
        selected_seeds = set(args.seeds)
        samples = [row for row in samples if int(row["seed"]) in selected_seeds]
        missing_seeds = selected_seeds - {int(row["seed"]) for row in samples}
        if missing_seeds:
            raise KeyError(f"requested seeds absent from selected manifest samples: {sorted(missing_seeds)}")

    selected_modes = set(args.mask_modes) if args.mask_modes is not None else None
    sample_tasks: list[tuple[dict, list[dict]]] = []
    for sample in samples:
        tasks = build_tasks(
            sample,
            args.head_scopes,
            args.ranking_tag,
            head_scope_counts=scope_counts,
        )
        if selected_modes is not None:
            tasks = [task for task in tasks if str(task["mask_mode"]) in selected_modes]
        if tasks:
            sample_tasks.append((sample, tasks))

    if args.task_index is not None:
        flattened = [
            (sample, task)
            for sample, tasks in sample_tasks
            for task in tasks
        ]
        if not 0 <= args.task_index < len(flattened):
            raise ValueError(f"task-index must be in [0, {len(flattened)})")
        sample, task = flattened[args.task_index]
        sample_tasks = [(sample, [task])]
    elif args.all_samples:
        sample_tasks = sample_tasks[args.worker_id :: args.num_workers]
    else:
        sample, tasks = sample_tasks[0]
        sample_tasks = [(sample, tasks[args.worker_id :: args.num_workers])]
    sample_tasks = [(sample, tasks) for sample, tasks in sample_tasks if tasks]
    if not sample_tasks:
        return
    if args.dry_run:
        flattened = [task for _, tasks in sample_tasks for task in tasks]
        print(
            json.dumps(
                {
                    "sample_count": len(sample_tasks),
                    "task_count": len(flattened),
                    "head_scopes": args.head_scopes,
                    "mask_modes": sorted({str(task["mask_mode"]) for task in flattened}),
                    "seeds": sorted({int(task["seed"]) for task in flattened}),
                    "first_task": flattened[0],
                    "last_task": flattened[-1],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES

    case_lookup = {case.key: case for case in CASES}
    track_paths: dict[tuple[str, int], Path] = {}
    for sample, _ in sample_tasks:
        sample_key = (str(sample["case"]), int(sample["seed"]))
        print(f"prepare frozen tracks {sample_key[0]}/seed_{sample_key[1]:05d}", flush=True)
        track_paths[sample_key] = prepare_tracks(
            sample,
            case_lookup,
            args.tracks_root or args.output_root,
            str(args.device),
            bool(args.overwrite),
        )

    total_tasks = sum(len(tasks) for _, tasks in sample_tasks)
    pipe = build_wan_ti2v_pipeline(build_args(int(sample_tasks[0][0]["seed"])))
    completed = 0
    for sample, tasks in sample_tasks:
        sample_key = (str(sample["case"]), int(sample["seed"]))
        track_path = track_paths[sample_key]
        for task in tasks:
            completed += 1
            output = task_root(task, args.output_root)
            print(
                f"[{completed}/{total_tasks}] start {output.relative_to(args.output_root)}",
                flush=True,
            )
            try:
                process(
                    pipe,
                    manifest,
                    ranking_entries,
                    scope_definitions,
                    args.head_ranking_path,
                    sample,
                    case_lookup,
                    task,
                    args.output_root,
                    track_path,
                    bool(args.overwrite),
                    bool(args.record_dose),
                )
            except Exception:
                output.mkdir(parents=True, exist_ok=True)
                (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                print(traceback.format_exc(), flush=True)
                raise
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"[{completed}/{total_tasks}] complete {output.relative_to(args.output_root)}",
                flush=True,
            )


if __name__ == "__main__":
    main()
