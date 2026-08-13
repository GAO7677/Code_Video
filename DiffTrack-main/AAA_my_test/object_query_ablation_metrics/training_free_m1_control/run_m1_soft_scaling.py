#!/usr/bin/env python3
"""Scale the exact Top100 M1 (R->R) post-softmax A@V contribution.

For selected heads and frozen temporal object-tube query rows R, this runner
implements

    Y_R(alpha) = Y_R + alpha * A[R, R] V[R].

Both CFG branches are modified, matching the Stage-3 M1 knockout when
``alpha=-1``.  No Q/K/V projection or softmax probability is changed.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any

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


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1"
)
DEFAULT_MANIFEST = (
    EXPERIMENT_ROOT
    / "training_free_top100_m23_guidance_v1/guidance_grid_manifest.json"
)
DEFAULT_HEAD_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
DEFAULT_TRACKS_ROOT = EXPERIMENT_ROOT / "stage4_temporal_v1"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "training_free_m1_control_v1/soft_scaling"
DEFAULT_CASE = "0613pybullet_sample_001460_w002"
PROTOCOL = "wan_top100_m1_soft_scaling_v1"


def soft_scaled_output(
    original_output: torch.Tensor,
    m1_contribution: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Pure algebra helper used by unit tests and the attention hook."""
    if original_output.shape != m1_contribution.shape:
        raise ValueError("original output and M1 contribution must have equal shapes")
    if not math.isfinite(alpha):
        raise ValueError("alpha must be finite")
    return original_output + float(alpha) * m1_contribution


class M1SoftScalingAblator(TemporalObjectTubeAblator):
    """Apply ``Y_R += alpha * M_RR`` to selected physical heads."""

    def __init__(self, *args, alpha: float, audit_decomposition: bool = False, **kwargs):
        if not math.isfinite(alpha) or not -1.0 <= alpha <= 1.0:
            raise ValueError("alpha must be finite and in [-1, 1]")
        super().__init__(*args, **kwargs)
        if self.mask_mode != "self_only":
            raise ValueError("M1 soft scaling requires mask_mode=self_only")
        self.alpha = float(alpha)
        self.audit_decomposition = bool(audit_decomposition)
        self.noop_mismatch_count = 0
        self.decomposition_max_abs_error = 0.0
        self.decomposition_max_rel_error = 0.0

    def _attention(self, q, k, v, original, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads:
            return original(q, k, v)
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")

        rows = self._rows(q.device)
        if rows is None or not rows.numel():
            raise RuntimeError("M1 soft scaling requires non-empty object-tube rows")

        output = original(q, k, v)
        output_heads = self._head_view(output, num_heads)
        selected = torch.as_tensor(heads, device=q.device, dtype=torch.long)
        before = None
        if self.alpha == 0.0:
            before = output_heads[:, rows][:, :, selected].clone()

        selected_v = self._selected_values(v, rows, heads, num_heads)
        contribution = original(q, k, selected_v)
        self.auxiliary_attention_calls += 1
        contribution_heads = self._head_view(contribution, num_heads)

        if self.audit_decomposition:
            complement = v - selected_v
            complement_contribution = original(q, k, complement)
            self.auxiliary_attention_calls += 1
            complement_heads = self._head_view(complement_contribution, num_heads)
            reconstructed = (
                contribution_heads[:, rows][:, :, selected]
                + complement_heads[:, rows][:, :, selected]
            )
            reference = output_heads[:, rows][:, :, selected]
            absolute = (reconstructed.float() - reference.float()).abs()
            relative = absolute / reference.float().abs().clamp_min(1e-6)
            self.decomposition_max_abs_error = max(
                self.decomposition_max_abs_error, float(absolute.max().cpu())
            )
            self.decomposition_max_rel_error = max(
                self.decomposition_max_rel_error, float(relative.max().cpu())
            )

        if self.record_dose:
            selected_ones = self._selected_values(
                torch.ones_like(v), rows, heads, num_heads
            )
            selected_mass = original(q, k, selected_ones)
            self.auxiliary_attention_calls += 1
            self._record_removed_dose(
                block,
                heads,
                rows,
                contribution_heads,
                output_heads,
                self._head_view(selected_mass, num_heads),
            )

        for head in heads:
            output_heads[:, rows, head, :] = soft_scaled_output(
                output_heads[:, rows, head, :],
                contribution_heads[:, rows, head, :],
                self.alpha,
            )

        if before is not None:
            after = output_heads[:, rows][:, :, selected]
            if not torch.equal(before, after):
                self.noop_mismatch_count += 1

        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += output.shape[0] * int(rows.numel()) * len(heads)
        return output

    def audit(self) -> dict[str, Any]:
        result = super().audit()
        result.update(
            {
                "alpha": self.alpha,
                "cfg_branches": ["conditional", "unconditional"],
                "noop_mismatch_count": self.noop_mismatch_count,
                "decomposition_audited": self.audit_decomposition,
                "decomposition_max_abs_error": self.decomposition_max_abs_error,
                "decomposition_max_rel_error": self.decomposition_max_rel_error,
            }
        )
        if self.alpha == 0.0 and self.noop_mismatch_count:
            raise RuntimeError(
                f"alpha=0 changed {self.noop_mismatch_count} attention outputs"
            )
        if self.audit_decomposition and self.decomposition_max_abs_error > 1e-3:
            raise RuntimeError(
                "M_RR + M_RC decomposition exceeded atol=1e-3: "
                f"{self.decomposition_max_abs_error}"
            )
        return result

    def dose_arrays(self) -> dict[str, np.ndarray]:
        arrays = super().dose_arrays()
        if not arrays:
            return arrays
        arrays["applied_delta_norm"] = (
            np.abs(np.float32(self.alpha)) * arrays["removed_value_norm"]
        )
        arrays["alpha"] = np.asarray(self.alpha, dtype=np.float32)
        return arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Legacy Wan video with Top100 M1 soft scaling."
    )
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--region", default="object_A")
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prepare-tracks", action="store_true")
    parser.add_argument("--record-dose", action="store_true")
    parser.add_argument("--audit-decomposition", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_directory(args: argparse.Namespace) -> Path:
    variant = (
        f"single_object__{args.region}__m1_all_time__top100"
        f"__alpha_{scale_tag(float(args.alpha))}"
    )
    return args.output_root / args.case / f"seed_{args.seed:05d}" / variant


def main() -> None:
    args = parse_args()
    if args.sampling_steps != 40:
        raise ValueError("the inherited audit requires exactly 40 denoising steps")
    if not math.isfinite(args.alpha) or not -1.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be finite and in [-1, 1]")
    if not math.isfinite(args.cfg_scale) or args.cfg_scale != 5.0:
        raise ValueError("TF-1 freezes cfg-scale=5")

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

    configuration = {
        "protocol": PROTOCOL,
        "case": args.case,
        "seed": int(args.seed),
        "region": region,
        "alpha": float(args.alpha),
        "equation": "Y_R(alpha)=Y_R+alpha*A[R,R]V[R]",
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
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "sample_shift": 5.0,
        "sample_solver": "unipc",
        "num_frames": 49,
        "height": 704,
        "width": 1280,
        "fps": 30,
        "output_directory": str(output),
    }
    if args.dry_run:
        print(json.dumps(configuration, ensure_ascii=False, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    complete_path.unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    wan_args.cfg_scale = float(args.cfg_scale)
    wan_args.sampling_steps = int(args.sampling_steps)
    pipe_wrapper = build_wan_ti2v_pipeline(wan_args)
    ablator = M1SoftScalingAblator(
        pipe_wrapper.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        "single_object",
        "self_only",
        region,
        tracks=tracks,
        anchor_frames=anchors,
        record_dose=bool(args.record_dose),
        alpha=float(args.alpha),
        audit_decomposition=bool(args.audit_decomposition),
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
            cfg_scale=float(args.cfg_scale),
            num_inference_steps=int(args.sampling_steps),
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
    if args.record_dose:
        atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())

    metadata = {
        **configuration,
        "output_video": str(output / "generated.mp4"),
        "model": "Wan2.2-TI2V-5B Legacy DiffSynth",
        "audit": audit,
        "implementation_path": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    complete_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "case": args.case,
                "seed": int(args.seed),
                "region": region,
                "alpha": float(args.alpha),
                "modified_head_events": audit["modified_head_events"],
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
