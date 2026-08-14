#!/usr/bin/env python3
"""Resume-safe batch generator for the test_5 Phase-B/D matrix.

The Wan pipeline is loaded once per process.  This avoids reloading the model
for every case/seed/alpha while preserving the exact single-run intervention.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for import_root in (REPO_ROOT, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (
    _run_pipe_once,
    build_wan_ti2v_pipeline,
    save_video_np,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_direct_scaling_phase_bd import (
    FULL_MASK_PROTOCOL,
    M1DirectScalingAblator,
    PROTOCOL,
    SAM2FullMaskM1DirectScalingAblator,
    TIME_SCOPE_TO_MASK,
    TOKEN_SOURCES,
    load_full_mask_partition,
    output_directory,
    validate_phase_configuration,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_m1_soft_scaling import (
    fp32_attention_decomposition_audit,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (
    generation_inputs,
    object_queries,
    process_baseline,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (
    atomic_npz,
    selected_head_entries,
    sha256_file,
    tracks_root,
    validate_head_ranking,
)
from AAA_my_test.sam2_region_query_utils import load_region_cache


PHASE_B_VARIANTS = ((0.1, 0, 39), (0.25, 0, 39))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baselines", "phase_b", "phase_d"), required=True)
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--head-ranking-path", type=Path, required=True)
    parser.add_argument("--tracks-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection-path", type=Path)
    parser.add_argument("--token-source", choices=TOKEN_SOURCES, default="sparse_points")
    parser.add_argument("--sam2-full-mask-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def phase_args(
    *,
    output_root: Path,
    sample: dict,
    phase: str,
    alpha: float,
    start: int,
    end: int,
    token_source: str = "sparse_points",
    sam2_full_mask_root: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        sampling_steps=40,
        cfg_scale=5.0,
        alpha=float(alpha),
        denoise_start=int(start),
        denoise_end=int(end),
        record_dose=True,
        phase_label=phase,
        time_scope="all_time",
        case=str(sample["case"]),
        seed=int(sample["seed"]),
        region=str(sample.get("target_region") or "object_A"),
        output_root=output_root,
        token_source=token_source,
        sam2_full_mask_root=sam2_full_mask_root,
    )


def variant_list(args: argparse.Namespace) -> tuple[tuple[float, int, int], ...]:
    if args.mode == "phase_b":
        return PHASE_B_VARIANTS
    if args.mode != "phase_d":
        return ()
    if args.selection_path is None or not args.selection_path.is_file():
        raise FileNotFoundError("Phase D requires --selection-path from Phase B")
    selection = json.loads(args.selection_path.read_text(encoding="utf-8"))
    alpha = float(selection["selected_alpha"])
    # Full 40 steps is reused from Phase B and registered after these two runs.
    return ((alpha, 0, 9), (alpha, 0, 19))


def load_protocol(args: argparse.Namespace, manifest: dict) -> tuple[list[dict], dict]:
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    ranking_entries = validate_head_ranking(
        manifest, ranking, allow_tagged_snapshot_change=True
    )
    entries = selected_head_entries(
        ranking_entries, "top100", dict(ranking.get("head_scopes") or {})
    )
    if len(entries) != 100:
        raise RuntimeError(f"expected exactly 100 Top heads, got {len(entries)}")
    return entries, ranking


def run_intervention(
    *,
    pipe_wrapper,
    sample: dict,
    entries: list[dict],
    ranking: dict,
    args: argparse.Namespace,
    alpha: float,
    start: int,
    end: int,
) -> Path:
    frozen = phase_args(
        output_root=args.output_root,
        sample=sample,
        phase=args.mode,
        alpha=alpha,
        start=start,
        end=end,
        token_source=args.token_source,
        sam2_full_mask_root=args.sam2_full_mask_root,
    )
    validate_phase_configuration(frozen)
    output = output_directory(frozen)
    ready = all(
        (output / name).is_file()
        for name in ("generated.mp4", "manifest.json", "complete.json", "dose_metrics.npz")
    )
    if ready and not args.overwrite:
        print(f"skip complete {output}", flush=True)
        return output

    case_lookup = {case.key: case for case in CASES}
    json_path, cache_dir, payload, wan_args, image = generation_inputs(
        sample, case_lookup, int(sample["seed"])
    )
    cache = load_region_cache(cache_dir.parent, cache_dir.name)
    points, query_regions = object_queries(cache)
    region_slices = {item.region_name: part for item, part in query_regions}
    region = str(sample.get("target_region") or "object_A")
    if region not in region_slices:
        raise RuntimeError(f"{sample['case']}: {region} absent from query cache")
    track_path = (
        tracks_root(args.tracks_root, str(sample["case"]), int(sample["seed"]))
        / "tracks.npz"
    )
    if not track_path.is_file():
        raise FileNotFoundError(f"frozen tracks missing: {track_path}")
    with np.load(track_path) as arrays:
        tracks = arrays["tracks"].astype(np.float32)
        anchors = arrays["anchor_pixel_frames"].astype(np.int64)
    if anchors.shape != (13,):
        raise RuntimeError(f"expected 13 latent anchors, got {anchors.tolist()}")

    partition = None
    full_mask_path = None
    if args.token_source == "sam2_full_mask":
        if args.sam2_full_mask_root is None:
            raise ValueError("--sam2-full-mask-root is required for sam2_full_mask")
        partition, full_mask_path = load_full_mask_partition(
            args.sam2_full_mask_root,
            str(sample["case"]),
            int(sample["seed"]),
            Path(str(sample["baseline_video"])),
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "complete.json").unlink(missing_ok=True)
    (output / "error.txt").unlink(missing_ok=True)
    ablator_class = (
        SAM2FullMaskM1DirectScalingAblator
        if partition is not None
        else M1DirectScalingAblator
    )
    ablator_kwargs = {"partition": partition} if partition is not None else {}
    ablator = ablator_class(
        pipe_wrapper.pipe,
        entries,
        points,
        region_slices,
        (704, 1280),
        "single_object",
        TIME_SCOPE_TO_MASK["all_time"],
        region,
        tracks=tracks,
        anchor_frames=anchors,
        record_dose=True,
        alpha=float(alpha),
        time_scope="all_time",
        denoise_start=int(start),
        denoise_end=int(end),
        **ablator_kwargs,
    )
    ablator.install()
    try:
        video = _run_pipe_once(
            pipe=pipe_wrapper,
            prompt=str(payload["input_caption"]),
            negative_prompt=str(wan_args.negative_prompt),
            seed=int(sample["seed"]),
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
    temporary = output / "generated.tmp.mp4"
    save_video_np(video, temporary, fps=30)
    temporary.replace(output / "generated.mp4")
    atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())
    configuration = {
        "protocol": FULL_MASK_PROTOCOL if partition is not None else PROTOCOL,
        "phase": args.mode,
        "case": str(sample["case"]),
        "seed": int(sample["seed"]),
        "region": region,
        "alpha": float(alpha),
        "time_scope": "all_time",
        "mask_mode": TIME_SCOPE_TO_MASK["all_time"],
        "denoise_start": int(start),
        "denoise_end": int(end),
        "raw_dose_steps": list(range(40)),
        "applied_denoising_steps": list(range(start, end + 1)),
        "equation": "Y_R=Y_R+alpha*M_RR(all_time)",
        "intervention_location": "post-softmax A@V before attention output projection",
        "cfg_branches": ["conditional", "unconditional"],
        "head_scope": "latest3350 Top100",
        "selected_head_count": len(entries),
        "selected_entries": entries,
        "head_ranking_path": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "manifest_path": str(args.manifest_path),
        "manifest_sha256": sha256_file(args.manifest_path),
        "input_json": str(json_path),
        "query_cache_dir": str(cache_dir),
        "token_source": args.token_source,
        "sam2_full_mask_cache": str(full_mask_path) if full_mask_path else None,
        "sam2_full_mask_cache_sha256": (
            sha256_file(full_mask_path) if full_mask_path else None
        ),
        "tracks_npz": str(track_path),
        "sampling_steps": 40,
        "cfg_scale": 5.0,
        "sample_shift": 5.0,
        "sample_solver": "unipc",
        "num_frames": 49,
        "height": 704,
        "width": 1280,
        "fps": 30,
        "output_directory": str(output),
        "ranking_completed_runs": int(ranking.get("completed_runs_at_selection", 3350)),
        "audit": audit,
        "output_video": str(output / "generated.mp4"),
    }
    atomic_json(output / "manifest.json", configuration)
    atomic_json(
        output / "complete.json",
        {
            "protocol": configuration["protocol"],
            "phase": args.mode,
            "case": str(sample["case"]),
            "seed": int(sample["seed"]),
            "alpha": float(alpha),
            "denoise_start": int(start),
            "denoise_end": int(end),
            "applied_head_events": int(audit["applied_head_events"]),
        },
    )
    del video
    gc.collect()
    torch.cuda.empty_cache()
    return output


def register_phase_d_full40(
    *,
    sample: dict,
    alpha: float,
    output_root: Path,
    token_source: str,
    sam2_full_mask_root: Path | None,
) -> None:
    source_args = phase_args(
        output_root=output_root,
        sample=sample,
        phase="phase_b",
        alpha=alpha,
        start=0,
        end=39,
        token_source=token_source,
        sam2_full_mask_root=sam2_full_mask_root,
    )
    target_args = phase_args(
        output_root=output_root,
        sample=sample,
        phase="phase_d",
        alpha=alpha,
        start=0,
        end=39,
        token_source=token_source,
        sam2_full_mask_root=sam2_full_mask_root,
    )
    source = output_directory(source_args)
    target = output_directory(target_args)
    if not all((source / name).is_file() for name in ("generated.mp4", "manifest.json", "complete.json")):
        raise FileNotFoundError(f"selected Phase-B full40 result is incomplete: {source}")
    target.mkdir(parents=True, exist_ok=True)
    for name in ("generated.mp4", "dose_metrics.npz"):
        link = target / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(source / name)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    atomic_json(
        target / "manifest.json",
        {
            **source_manifest,
            "phase": "phase_d",
            "output_directory": str(target),
            "output_video": str(target / "generated.mp4"),
            "reused_from_phase_b": str(source),
            "regenerated": False,
        },
    )
    atomic_json(
        target / "complete.json",
        {
            "protocol": source_manifest.get("protocol", PROTOCOL),
            "phase": "phase_d",
            "case": str(sample["case"]),
            "seed": int(sample["seed"]),
            "alpha": float(alpha),
            "denoise_start": 0,
            "denoise_end": 39,
            "reused_from_phase_b": str(source),
            "regenerated": False,
        },
    )


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest_path.read_text(encoding="utf-8"))
    samples = list(manifest.get("samples") or [])
    if len(samples) != 100:
        raise RuntimeError(f"expected 100 manifest samples, got {len(samples)}")
    entries, ranking = load_protocol(args, manifest)
    variants = variant_list(args)
    if args.mode != "baselines":
        audit = fp32_attention_decomposition_audit()
        if not audit["passed"]:
            raise RuntimeError(f"FP32 attention decomposition failed: {audit}")

    case_lookup = {case.key: case for case in CASES}
    _, _, _, wan_args, _ = generation_inputs(samples[0], case_lookup, int(samples[0]["seed"]))
    wan_args.cfg_scale = 5.0
    wan_args.sampling_steps = 40
    pipe_wrapper = build_wan_ti2v_pipeline(wan_args)
    total = len(samples) if args.mode == "baselines" else len(samples) * len(variants)
    completed = 0
    try:
        if args.mode == "baselines":
            for index, sample in enumerate(samples, start=1):
                task = {
                    "case": str(sample["case"]),
                    "seed": int(sample["seed"]),
                    "mode": "baseline",
                }
                print(
                    f"[{index}/{len(samples)}] baseline {task['case']}/seed_{task['seed']:05d}",
                    flush=True,
                )
                process_baseline(
                    pipe_wrapper, task, sample, case_lookup, bool(args.overwrite)
                )
                completed += 1
                gc.collect()
                torch.cuda.empty_cache()
        else:
            for sample in samples:
                for alpha, start, end in variants:
                    completed += 1
                    print(
                        f"[{completed}/{total}] {args.mode} {sample['case']}/"
                        f"seed_{int(sample['seed']):05d} alpha={alpha:g} denoise={start}..{end}",
                        flush=True,
                    )
                    try:
                        run_intervention(
                            pipe_wrapper=pipe_wrapper,
                            sample=sample,
                            entries=entries,
                            ranking=ranking,
                            args=args,
                            alpha=alpha,
                            start=start,
                            end=end,
                        )
                    except Exception:
                        frozen = phase_args(
                            output_root=args.output_root,
                            sample=sample,
                            phase=args.mode,
                            alpha=alpha,
                            start=start,
                            end=end,
                            token_source=args.token_source,
                            sam2_full_mask_root=args.sam2_full_mask_root,
                        )
                        output = output_directory(frozen)
                        output.mkdir(parents=True, exist_ok=True)
                        (output / "error.txt").write_text(
                            traceback.format_exc(), encoding="utf-8"
                        )
                        raise
    finally:
        del pipe_wrapper
        gc.collect()
        torch.cuda.empty_cache()

    if args.mode == "phase_d":
        selection = json.loads(args.selection_path.read_text(encoding="utf-8"))
        alpha = float(selection["selected_alpha"])
        for sample in samples:
            register_phase_d_full40(
                sample=sample,
                alpha=alpha,
                output_root=args.output_root,
                token_source=args.token_source,
                sam2_full_mask_root=args.sam2_full_mask_root,
            )
    atomic_json(
        args.output_root / f"{args.mode}_batch_complete.json",
        {
            "mode": args.mode,
            "manifest": str(args.manifest_path),
            "sample_count": len(samples),
            "variant_count": len(variants),
            "completed_loop_units": completed,
        },
    )
    print(f"{args.mode} batch complete", flush=True)


if __name__ == "__main__":
    main()
