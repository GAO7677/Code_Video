#!/usr/bin/env python3
"""Training-free Wan guidance from the clean-vs-Top100-M1 prediction difference.

The final denoising prediction matches DiffTrack/PAG's CFG equation exactly:

    eps = eps_u + cfg * (eps_c - eps_u) + pag * (eps_c - eps_m1)

``eps_m1`` is computed with the existing audited temporal-object-tube M1
implementation.  For the selected latest3350 Top100 layer-heads, M1 subtracts
the exact post-softmax R->R contribution from object-tube query rows without
renormalizing attention probabilities.

This runner deliberately lives outside the DiffTrack CogVideoX pipeline.  The
original processor is CogVideoX-specific; the guidance equation is reused here
while the perturbation is implemented by the existing Legacy Wan ablator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
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
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    build_args,
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
DEFAULT_MANIFEST = EXPERIMENT_ROOT / "stage4_runtime/stage4_manifest.json"
DEFAULT_HEAD_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
DEFAULT_TRACKS_ROOT = EXPERIMENT_ROOT / "stage4_temporal_v1"
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "training_free_top100_m1_guidance_v1"
DEFAULT_CASE = "0613pybullet_sample_001460_w002"
DEFAULT_SEED = 47326
PROTOCOL = "wan_top100_m1_perturbed_attention_guidance_v1"

# These names map directly to the already-audited M1 implementations.  The
# all-time variant is the Stage-3 M1 intervention: same, past, and future R->R.
M1_TIME_SCOPES = {
    "all_time": "self_only",
    "future": "self_future",
    "same": "self_same",
    "past": "self_past",
}


def adjusted_conditional_prediction(
    clean_conditional: torch.Tensor,
    perturbed_conditional: torch.Tensor,
    *,
    cfg_scale: float,
    pag_scale: float,
) -> torch.Tensor:
    """Return the conditional tensor that yields the exact PAG term after CFG.

    The surrounding Wan pipeline computes ``u + cfg * (returned_cond - u)``.
    Dividing the perturbation term by ``cfg`` here therefore makes the final
    coefficient exactly ``pag_scale`` rather than ``cfg * pag_scale``.
    """

    if not isinstance(clean_conditional, torch.Tensor) or not isinstance(
        perturbed_conditional, torch.Tensor
    ):
        raise TypeError("clean and perturbed predictions must be torch tensors")
    if clean_conditional.shape != perturbed_conditional.shape:
        raise ValueError(
            "clean/perturbed prediction shape mismatch: "
            f"{tuple(clean_conditional.shape)} != {tuple(perturbed_conditional.shape)}"
        )
    if not math.isfinite(cfg_scale) or cfg_scale <= 0:
        raise ValueError("cfg_scale must be finite and positive")
    if not math.isfinite(pag_scale) or pag_scale < 0:
        raise ValueError("pag_scale must be finite and non-negative")
    return clean_conditional + (pag_scale / cfg_scale) * (
        clean_conditional - perturbed_conditional
    )


class Top100M1Guidance:
    """Run clean+M1 conditional predictions and preserve the normal CFG branch.

    DiffSynth's Legacy Wan pipeline calls ``model_fn`` twice per denoising step:
    conditional first and unconditional second.  This wrapper performs one
    additional conditional forward with M1 active, then returns an adjusted
    conditional prediction.  The unconditional forward remains unchanged.
    """

    def __init__(
        self,
        pipe: Any,
        ablator: TemporalObjectTubeAblator,
        *,
        cfg_scale: float,
        pag_scale: float,
        expected_steps: int = 40,
    ) -> None:
        if not math.isfinite(cfg_scale) or cfg_scale <= 0 or cfg_scale == 1.0:
            raise ValueError(
                "this v1 runner requires a finite positive CFG scale different from 1"
            )
        if not math.isfinite(pag_scale) or pag_scale <= 0:
            raise ValueError("pag_scale must be finite and positive for an M1-guided run")
        self.pipe = pipe
        self.ablator = ablator
        self.cfg_scale = float(cfg_scale)
        self.pag_scale = float(pag_scale)
        self.expected_steps = int(expected_steps)
        self._clean_model_fn: Callable[..., torch.Tensor] | None = None
        self._perturbed_model_fn: Callable[..., torch.Tensor] | None = None
        self._installed = False
        self.pipeline_calls_by_step: dict[int, int] = {}
        self.guided_calls_by_step: dict[int, int] = {}
        self.delta_l2_by_step: dict[int, float] = {}
        self.clean_l2_by_step: dict[int, float] = {}

    def install(self) -> None:
        if self._installed:
            raise RuntimeError("guidance wrapper is already installed")
        self.ablator.install()
        self._clean_model_fn = self.ablator._original_model_fn
        self._perturbed_model_fn = self.pipe.model_fn
        if self._clean_model_fn is None:
            raise RuntimeError("M1 ablator did not capture the original model_fn")
        self.pipe.model_fn = self
        self._installed = True

    def remove(self) -> None:
        if not self._installed:
            return
        self.ablator.remove()
        self._installed = False

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        if self._clean_model_fn is None or self._perturbed_model_fn is None:
            raise RuntimeError("guidance wrapper is not installed")
        timestep = kwargs.get("timestep")
        latents = kwargs.get("latents")
        if timestep is None or latents is None:
            return self._clean_model_fn(*args, **kwargs)

        step = self.ablator._step(timestep)
        branch_call = self.pipeline_calls_by_step.get(step, 0)
        if branch_call not in (0, 1):
            raise RuntimeError(
                f"expected conditional/unconditional calls only, got call {branch_call} "
                f"at denoising step {step}"
            )

        # DiffSynth Wan ordering is positive/conditional first, negative second.
        clean = self._clean_model_fn(*args, **kwargs)
        if branch_call == 0:
            perturbed = self._perturbed_model_fn(*args, **kwargs)
            guided = adjusted_conditional_prediction(
                clean,
                perturbed,
                cfg_scale=self.cfg_scale,
                pag_scale=self.pag_scale,
            )
            delta = (clean - perturbed).detach().float()
            self.delta_l2_by_step[step] = float(torch.linalg.vector_norm(delta).cpu())
            self.clean_l2_by_step[step] = float(
                torch.linalg.vector_norm(clean.detach().float()).cpu()
            )
            self.guided_calls_by_step[step] = self.guided_calls_by_step.get(step, 0) + 1
            result = guided
        else:
            result = clean

        self.pipeline_calls_by_step[step] = branch_call + 1
        return result

    def audit(self) -> dict[str, Any]:
        expected_step_ids = list(range(self.expected_steps))
        if sorted(self.pipeline_calls_by_step) != expected_step_ids:
            raise RuntimeError(
                f"expected pipeline steps {expected_step_ids}, got "
                f"{sorted(self.pipeline_calls_by_step)}"
            )
        if any(value != 2 for value in self.pipeline_calls_by_step.values()):
            raise RuntimeError(
                "expected two pipeline CFG calls per step, got "
                f"{self.pipeline_calls_by_step}"
            )
        if sorted(self.guided_calls_by_step) != expected_step_ids or any(
            value != 1 for value in self.guided_calls_by_step.values()
        ):
            raise RuntimeError(
                "expected exactly one M1-guided conditional call per step, got "
                f"{self.guided_calls_by_step}"
            )
        if sorted(self.ablator.model_call_counts) != expected_step_ids or any(
            value != 1 for value in self.ablator.model_call_counts.values()
        ):
            raise RuntimeError(
                "expected exactly one perturbed M1 forward per step, got "
                f"{self.ablator.model_call_counts}"
            )
        expected_head_events = len(self.ablator.entries) * self.expected_steps
        if self.ablator.modified_head_events != expected_head_events:
            raise RuntimeError(
                f"modified {self.ablator.modified_head_events} head events, "
                f"expected {expected_head_events}"
            )
        if not self.ablator.query_token_indices:
            raise RuntimeError("temporal object-tube token indices were not resolved")

        dose_finite_events = int(np.isfinite(self.ablator.dose_attention_mass).sum())
        if self.ablator.record_dose and dose_finite_events != expected_head_events:
            raise RuntimeError(
                f"recorded {dose_finite_events} dose events, expected "
                f"{expected_head_events} conditional-only events"
            )
        relative_delta = {
            str(step): self.delta_l2_by_step[step]
            / max(self.clean_l2_by_step[step], 1e-12)
            for step in expected_step_ids
        }
        return {
            "pipeline_calls_by_step": self.pipeline_calls_by_step,
            "guided_calls_by_step": self.guided_calls_by_step,
            "perturbed_calls_by_step": self.ablator.model_call_counts,
            "modified_forward_calls": self.ablator.modified_forward_calls,
            "modified_head_events": self.ablator.modified_head_events,
            "expected_head_events": expected_head_events,
            "query_token_indices": self.ablator.query_token_indices,
            "query_token_indices_by_latent_frame": (
                self.ablator.query_token_indices_by_latent_frame
            ),
            "perturbation_delta_l2_by_step": self.delta_l2_by_step,
            "clean_prediction_l2_by_step": self.clean_l2_by_step,
            "relative_perturbation_l2_by_step": relative_delta,
            "dose_recorded": self.ablator.record_dose,
            "dose_finite_events": dose_finite_events,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one Legacy Wan video with latest3350 Top100 M1 guidance."
    )
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_HEAD_RANKING)
    parser.add_argument("--tracks-root", type=Path, default=DEFAULT_TRACKS_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--target-scope",
        choices=("single_object", "all_objects"),
        default="single_object",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="single-object region; defaults to the first object region in the manifest",
    )
    parser.add_argument(
        "--m1-time-scope",
        choices=tuple(M1_TIME_SCOPES),
        default="all_time",
        help=(
            "all_time reproduces Stage-3 M1 (same+future+past); future/same/past "
            "reuse the corresponding strict temporal M1 variant"
        ),
    )
    parser.add_argument("--pag-scale", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare-tracks",
        action="store_true",
        help="compute frozen baseline tracks if they are absent under tracks-root",
    )
    parser.add_argument("--record-dose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def scale_tag(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def resolve_sample(manifest: dict[str, Any], case: str, seed: int) -> dict[str, Any]:
    matches = [
        row
        for row in manifest.get("samples", [])
        if str(row.get("case")) == case and int(row.get("seed", -1)) == seed
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one manifest sample for {case}/seed_{seed:05d}, got {len(matches)}")
    return matches[0]


def resolve_target(
    sample: dict[str, Any], target_scope: str, requested_region: str | None
) -> str | None:
    object_regions = [
        str(row["region_name"])
        for row in sample.get("regions", [])
        if str(row.get("region_type")) == "object"
    ]
    if not object_regions:
        raise RuntimeError("sample manifest contains no object regions")
    if target_scope == "all_objects":
        if requested_region is not None:
            raise ValueError("--region cannot be used with --target-scope all_objects")
        return None
    region = requested_region or object_regions[0]
    if region not in object_regions:
        raise ValueError(f"unknown object region {region!r}; choices are {object_regions}")
    return region


def output_directory(args: argparse.Namespace, region: str | None) -> Path:
    target = region if region is not None else "all_objects"
    variant = (
        f"single_object__{target}" if region is not None else "all_objects__all_objects"
    )
    variant += (
        f"__m1_{args.m1_time_scope}__top100__pag{scale_tag(float(args.pag_scale))}"
    )
    return args.output_root / args.case / f"seed_{args.seed:05d}" / variant


def main() -> None:
    args = parse_args()
    if args.sampling_steps != 40:
        raise ValueError("the inherited ablator/dose audit currently requires 40 denoising steps")
    if (
        not math.isfinite(args.cfg_scale)
        or args.cfg_scale <= 0
        or args.cfg_scale == 1.0
    ):
        raise ValueError(
            "cfg-scale must be finite, positive, and different from 1 for this v1 runner"
        )
    if not math.isfinite(args.pag_scale) or args.pag_scale <= 0:
        raise ValueError("pag-scale must be finite and positive")

    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest,
        ranking,
        allow_tagged_snapshot_change=True,
    )
    entries = selected_head_entries(
        ranking_entries,
        "top100",
        dict(ranking.get("head_scopes") or {}),
    )
    if len(entries) != 100:
        raise RuntimeError(f"expected latest3350 Top100, got {len(entries)} heads")

    sample = resolve_sample(manifest, args.case, args.seed)
    region = resolve_target(sample, args.target_scope, args.region)
    output = output_directory(args, region)
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
    region_slices = {
        item.region_name: point_slice for item, point_slice in query_regions
    }
    if region is not None and region not in region_slices:
        raise RuntimeError(f"region {region!r} is absent from the cached object queries")

    track_path = tracks_root(args.tracks_root, args.case, args.seed) / "tracks.npz"
    if not track_path.is_file():
        if not args.prepare_tracks:
            raise FileNotFoundError(
                f"frozen baseline tracks are missing: {track_path}; pass --prepare-tracks"
            )
        track_path = prepare_tracks(
            sample,
            case_lookup,
            args.tracks_root,
            str(args.device),
            overwrite=False,
        )
    with np.load(track_path) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)

    configuration = {
        "protocol": PROTOCOL,
        "case": args.case,
        "seed": args.seed,
        "input_json": str(json_path),
        "query_cache_dir": str(cache_dir),
        "tracks_npz": str(track_path),
        "target_scope": args.target_scope,
        "region": region,
        "m1_time_scope": args.m1_time_scope,
        "mask_mode": M1_TIME_SCOPES[args.m1_time_scope],
        "head_scope": "top100",
        "selected_head_count": len(entries),
        "head_ranking_path": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "cfg_scale": float(args.cfg_scale),
        "pag_scale": float(args.pag_scale),
        "sampling_steps": int(args.sampling_steps),
        "output_directory": str(output),
        "selected_entries": entries,
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
    ablator = TemporalObjectTubeAblator(
        pipe_wrapper.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        args.target_scope,
        M1_TIME_SCOPES[args.m1_time_scope],
        region,
        tracks=tracks,
        anchor_frames=anchors,
        record_dose=bool(args.record_dose),
    )
    guidance = Top100M1Guidance(
        pipe_wrapper.pipe,
        ablator,
        cfg_scale=float(args.cfg_scale),
        pag_scale=float(args.pag_scale),
        expected_steps=int(args.sampling_steps),
    )
    guidance.install()
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
        guidance.remove()
    audit = guidance.audit()

    temporary_video = output / "generated.tmp.mp4"
    save_video_np(video, temporary_video, fps=30)
    temporary_video.replace(output / "generated.mp4")
    if args.record_dose:
        atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())

    metadata = {
        **configuration,
        "output_video": str(output / "generated.mp4"),
        "model": "Wan2.2-TI2V-5B Legacy DiffSynth",
        "num_frames": 49,
        "height": 704,
        "width": 1280,
        "fps": 30,
        "cfg_call_order": ["conditional", "unconditional"],
        "guidance_equation": (
            "eps_u + cfg*(eps_c-eps_u) + pag*(eps_c-eps_m1)"
        ),
        "conditional_return_equation": (
            "eps_c + (pag/cfg)*(eps_c-eps_m1)"
        ),
        "m1_definition": (
            "For selected heads and R query rows, subtract exact post-softmax "
            "sum_{k in R} A_qk V_k; no attention renormalization."
        ),
        "m1_object_partition": (
            "R is the frozen CoTracker object-token tube at 13 latent anchors"
        ),
        "perturbed_branches": ["conditional"],
        "perturbed_denoising_steps": list(range(args.sampling_steps)),
        "extra_dit_forwards_per_step": 1,
        "audit": audit,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    complete_path.write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "case": args.case,
                "seed": args.seed,
                "region": region,
                "m1_time_scope": args.m1_time_scope,
                "pag_scale": float(args.pag_scale),
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
