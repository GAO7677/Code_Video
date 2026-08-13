#!/usr/bin/env python3
"""Phase-B/D direct scaling of the exact all-time Top100 M1 contribution.

This runner is intentionally separate from ``run_m1_soft_scaling.py`` because
that v1 runner is currently used by an active TF-1 queue.  For selected heads,
it applies

    Y_R <- Y_R + alpha * M_RR(scope)

on both CFG branches.  The raw contribution is measured at all 40 denoising
steps, while the configured alpha is applied only inside the inclusive
denoising window.  Phase D varies only that denoising window.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

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
from AAA_my_test.object_query_ablation_metrics.run_top100_m1_perturbed_attention_guidance import (  # noqa: E402
    resolve_sample,
    resolve_target,
    scale_tag,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_soft_scaling import (  # noqa: E402
    DEFAULT_HEAD_RANKING,
    DEFAULT_MANIFEST,
    DEFAULT_TRACKS_ROOT,
    EXPERIMENT_ROOT,
    fp32_attention_decomposition_audit,
    soft_scaled_output,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    generation_inputs,
    object_queries,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (  # noqa: E402
    TemporalObjectTubeAblator,
    atomic_npz,
    prepare_tracks,
    selected_head_entries,
    sha256_file,
    tracks_root,
    validate_head_ranking,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache  # noqa: E402


PROTOCOL = "wan_top100_m1_direct_scaling_phase_bd_v2"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "training_free_m1_direct_enhancement_v2"
TIME_SCOPE_TO_MASK = {
    "all_time": "self_only",
}


def validate_denoising_window(start: int, end: int, steps: int = 40) -> None:
    if not 0 <= start <= end < steps:
        raise ValueError(
            f"expected inclusive denoising window inside [0,{steps - 1}], "
            f"got [{start},{end}]"
        )


def alpha_at_step(alpha: float, step: int, start: int, end: int) -> float:
    return float(alpha) if start <= step <= end else 0.0


def validate_phase_configuration(args: argparse.Namespace) -> None:
    """Enforce the frozen controls shared by the Phase-B/D launchers."""
    if args.sampling_steps != 40:
        raise ValueError("Phase-B/D freezes exactly 40 denoising steps")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale != 5.0:
        raise ValueError("Phase-B/D freezes cfg-scale=5")
    if not math.isfinite(args.alpha) or not 0.0 < args.alpha <= 1.0:
        raise ValueError("Phase-B/D alpha must be finite and in (0, 1]")
    validate_denoising_window(args.denoise_start, args.denoise_end)
    if not args.record_dose:
        raise ValueError("Phase-B/D requires --record-dose")
    if args.time_scope != "all_time":
        raise ValueError("Phase-B/D is frozen to all_time M1")
    if args.phase_label == "phase_b" and (
        args.time_scope != "all_time"
        or args.denoise_start != 0
        or args.denoise_end != 39
        or args.alpha not in (0.1, 0.25)
    ):
        raise ValueError(
            "Phase B is frozen to alpha={0.1,0.25}, all_time, denoise 0..39"
        )


class M1DirectScalingAblator(TemporalObjectTubeAblator):
    """Scale all-time M1 on a fixed denoising window."""

    def __init__(
        self,
        *args,
        alpha: float,
        time_scope: str,
        denoise_start: int,
        denoise_end: int,
        **kwargs,
    ) -> None:
        if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
            raise ValueError("Phase-B/D alpha must be finite and in (0, 1]")
        if time_scope not in TIME_SCOPE_TO_MASK:
            raise ValueError(f"unknown M1 time scope: {time_scope}")
        validate_denoising_window(denoise_start, denoise_end)
        super().__init__(*args, **kwargs)
        if self.mask_mode != TIME_SCOPE_TO_MASK[time_scope]:
            raise ValueError(
                f"time_scope={time_scope} requires mask_mode="
                f"{TIME_SCOPE_TO_MASK[time_scope]}"
            )
        self.alpha = float(alpha)
        self.time_scope = time_scope
        self.denoise_start = int(denoise_start)
        self.denoise_end = int(denoise_end)
        self.applied_head_events = 0

    def _effective_alpha(self) -> float:
        return alpha_at_step(
            self.alpha,
            self.current_step,
            self.denoise_start,
            self.denoise_end,
        )

    def _all_time_attention(self, q, k, v, original, block: int, heads, num_heads: int):
        rows = self._rows(q.device)
        if rows is None or not rows.numel():
            raise RuntimeError("M1 direct scaling requires non-empty object-tube rows")

        output = original(q, k, v)
        output_heads = self._head_view(output, num_heads)
        original_heads = output_heads.clone() if self.record_dose else output_heads
        selected_v = self._selected_values(v, rows, heads, num_heads)
        contribution = original(q, k, selected_v)
        contribution_heads = self._head_view(contribution, num_heads)

        if self.record_dose:
            selected_ones = self._selected_values(
                torch.ones_like(v), rows, heads, num_heads
            )
            selected_mass = original(q, k, selected_ones)
            self._record_removed_dose(
                block,
                heads,
                rows,
                contribution_heads,
                original_heads,
                self._head_view(selected_mass, num_heads),
            )

        effective_alpha = self._effective_alpha()
        if effective_alpha:
            for head in heads:
                output_heads[:, rows, head, :] = soft_scaled_output(
                    output_heads[:, rows, head, :],
                    contribution_heads[:, rows, head, :],
                    effective_alpha,
                )
            self.applied_head_events += len(heads)

        return output, int(rows.numel()), 2 if self.record_dose else 1

    def _attention(self, q, k, v, original, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")

        output, affected_rows, auxiliary_calls = self._all_time_attention(
            q, k, v, original, block, heads, num_heads
        )

        if not isinstance(output, torch.Tensor) or output.ndim != 3:
            raise RuntimeError(f"unexpected attention output: {type(output)}")
        self.auxiliary_attention_calls += auxiliary_calls
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += output.shape[0] * affected_rows * len(heads)
        return output

    def audit(self) -> dict:
        result = super().audit()
        active_steps = self.denoise_end - self.denoise_start + 1
        expected_applied = len(self.entries) * active_steps * 2
        if self.applied_head_events != expected_applied:
            raise RuntimeError(
                f"applied {self.applied_head_events} head events, expected "
                f"{expected_applied}"
            )
        result.update(
            {
                "alpha": self.alpha,
                "time_scope": self.time_scope,
                "denoise_start": self.denoise_start,
                "denoise_end": self.denoise_end,
                "raw_dose_steps": list(range(40)),
                "applied_denoising_steps": list(
                    range(self.denoise_start, self.denoise_end + 1)
                ),
                "cfg_branches": ["conditional", "unconditional"],
                "applied_head_events": self.applied_head_events,
                "expected_applied_head_events": expected_applied,
            }
        )
        return result

    def dose_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().dose_arrays()
        if not arrays:
            return arrays
        schedule = np.zeros(40, dtype=np.float32)
        schedule[self.denoise_start : self.denoise_end + 1] = np.float32(
            self.alpha
        )
        arrays["applied_delta_norm"] = (
            np.abs(schedule[:, None, None, None]) * arrays["removed_value_norm"]
        )
        arrays["alpha_by_denoising_step"] = schedule
        arrays["active_denoising_step"] = schedule != 0
        return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Phase-B/D Top100 M1 direct contribution scaling."
    )
    parser.add_argument("--case", default="0613pybullet_sample_001460_w002")
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--region", default="object_A")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument(
        "--time-scope", choices=tuple(TIME_SCOPE_TO_MASK), default="all_time"
    )
    parser.add_argument("--denoise-start", type=int, default=0)
    parser.add_argument("--denoise-end", type=int, default=39)
    parser.add_argument("--phase-label", choices=("phase_b", "phase_d"), required=True)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-tracks", action="store_true")
    parser.add_argument("--record-dose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_directory(args: argparse.Namespace) -> Path:
    variant = (
        f"single_object__{args.region}__m1_{args.time_scope}__top100"
        f"__alpha_{scale_tag(float(args.alpha))}"
        f"__denoise_{int(args.denoise_start):02d}_{int(args.denoise_end):02d}"
    )
    return (
        args.output_root
        / args.phase_label
        / args.case
        / f"seed_{args.seed:05d}"
        / variant
    )


def main() -> None:
    args = parse_args()
    validate_phase_configuration(args)

    fp32_audit = fp32_attention_decomposition_audit()
    if not fp32_audit["passed"]:
        raise RuntimeError(f"FP32 attention decomposition failed: {fp32_audit}")

    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest, ranking, allow_tagged_snapshot_change=True
    )
    entries = selected_head_entries(
        ranking_entries, "top100", dict(ranking.get("head_scopes") or {})
    )
    if len(entries) != 100:
        raise RuntimeError(f"expected exactly 100 selected heads, got {len(entries)}")

    sample = resolve_sample(manifest, args.case, args.seed)
    region = resolve_target(sample, "single_object", args.region)
    output = output_directory(args)
    complete_path = output / "complete.json"
    if complete_path.is_file() and not args.overwrite:
        print(f"skip complete run: {output}")
        return

    case_lookup = {case.key: case for case in CASES}
    json_path, cache_dir, payload, wan_args, image = generation_inputs(
        sample, case_lookup, args.seed
    )
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    points, query_regions = object_queries(cache)
    region_slices = {item.region_name: part for item, part in query_regions}
    if region not in region_slices:
        raise RuntimeError(f"region {region!r} is absent from cached object queries")

    track_path = tracks_root(args.tracks_root, args.case, args.seed) / "tracks.npz"
    if not track_path.is_file():
        if not args.prepare_tracks:
            raise FileNotFoundError(
                f"frozen tracks missing: {track_path}; pass --prepare-tracks"
            )
        track_path = prepare_tracks(
            sample, case_lookup, args.tracks_root, str(args.device), overwrite=False
        )
    with np.load(track_path) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)
    if anchors.shape != (13,):
        raise RuntimeError(f"expected 13 latent anchors, got {anchors.tolist()}")

    mask_mode = TIME_SCOPE_TO_MASK[args.time_scope]
    configuration = {
        "protocol": PROTOCOL,
        "phase": args.phase_label,
        "case": args.case,
        "seed": int(args.seed),
        "region": region,
        "alpha": float(args.alpha),
        "time_scope": args.time_scope,
        "mask_mode": mask_mode,
        "denoise_start": int(args.denoise_start),
        "denoise_end": int(args.denoise_end),
        "raw_dose_steps": list(range(40)),
        "applied_denoising_steps": list(
            range(args.denoise_start, args.denoise_end + 1)
        ),
        "equation": "Y_R=Y_R+alpha*M_RR(time_scope)",
        "intervention_location": "post-softmax A@V before attention output projection",
        "cfg_branches": ["conditional", "unconditional"],
        "head_scope": "top100",
        "selected_head_count": len(entries),
        "selected_entries": entries,
        "head_ranking_path": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": sha256_file(args.manifest_path),
        "input_json": str(json_path),
        "query_cache_dir": str(cache_dir),
        "tracks_npz": str(track_path),
        "latent_anchor_count": int(len(anchors)),
        "sampling_steps": 40,
        "cfg_scale": 5.0,
        "sample_shift": 5.0,
        "sample_solver": "unipc",
        "num_frames": 49,
        "height": 704,
        "width": 1280,
        "fps": 30,
        "output_directory": str(output),
        "fp32_attention_decomposition_audit": fp32_audit,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    if args.dry_run:
        print(json.dumps(configuration, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    wan_args.cfg_scale = 5.0
    wan_args.sampling_steps = 40
    pipe_wrapper = build_wan_ti2v_pipeline(wan_args)
    ablator = M1DirectScalingAblator(
        pipe_wrapper.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        "single_object",
        mask_mode,
        region,
        tracks=tracks,
        anchor_frames=anchors,
        record_dose=True,
        alpha=float(args.alpha),
        time_scope=args.time_scope,
        denoise_start=int(args.denoise_start),
        denoise_end=int(args.denoise_end),
    )
    ablator.install()
    try:
        video = _run_pipe_once(
            pipe=pipe_wrapper,
            prompt=str(payload["input_caption"]),
            negative_prompt=str(wan_args.negative_prompt),
            seed=int(args.seed),
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
    audit = ablator.audit()

    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")
    atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())

    metadata = {
        **configuration,
        "output_video": str(output / "generated.mp4"),
        "model": "Wan2.2-TI2V-5B Legacy DiffSynth",
        "audit": audit,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "phase": args.phase_label,
                "case": args.case,
                "seed": int(args.seed),
                "region": region,
                "alpha": float(args.alpha),
                "time_scope": args.time_scope,
                "denoise_start": int(args.denoise_start),
                "denoise_end": int(args.denoise_end),
                "applied_head_events": audit["applied_head_events"],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    del video, pipe_wrapper
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(output)


if __name__ == "__main__":
    main()
