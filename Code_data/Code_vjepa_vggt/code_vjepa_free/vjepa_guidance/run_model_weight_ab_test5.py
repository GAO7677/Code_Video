#!/usr/bin/env python3
from __future__ import annotations

"""
Run the 4-model-line baseline/guided A/B on the deduplicated test_5 list and
score physical metrics.

Generate only:
CUDA_VISIBLE_DEVICES=5,6,7 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py \
  --stage generate

Score only:
CUDA_VISIBLE_DEVICES=7 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py \
  --stage score
"""

import argparse
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

try:
    from .experiment_presets import resolve_train0705_preset
except ImportError:
    from experiment_presets import resolve_train0705_preset


THIS_FILE = Path(__file__).resolve()
GUIDANCE_DIR = THIS_FILE.parent
ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
WAN22_OFFICIAL_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main_official")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
PY_WAN_CU128 = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
PY_WAN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")

DEFAULT_INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705")

WAN22_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
LORA_STEP000500 = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500"
)
TRAIN0705_STEP002500 = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500"
)
TRAIN0705_STEP007000 = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-007000"
)

MID12 = ["8", "10", "11", "13", "14", "16", "17", "19", "20", "22", "23", "25"]
DEFAULT_FREQGUIDE_MODEL_SUFFIX = "freqguide_tunionx1_lp018_d5"


@dataclass(frozen=True)
class Family:
    family_id: str
    baseline_dir: Path
    guided_dir: Path
    score_baseline_dir: Path | None = None


TRAIN0705_BASELINE_002500 = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-002500"
)
TRAIN0705_BASELINE_007000 = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-007000"
)
LORA_BASELINE_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_early_lora_step000500/baseline"
)
OFFICIAL_BASELINE_SOURCE = Path("/data/gaoya/agent-data/outputs/official_freqguide_test5/baseline")


def build_families(output_root: Path) -> list[Family]:
    return [
        Family(
            family_id="wan22_official_ti2v5b",
            baseline_dir=output_root / "wan22_official_ti2v5b" / "baseline",
            guided_dir=output_root / "wan22_official_ti2v5b" / "guided",
        ),
        Family(
            family_id="wan22_early_lora_step000500",
            baseline_dir=output_root / "wan22_early_lora_step000500" / "baseline",
            guided_dir=output_root / "wan22_early_lora_step000500" / "guided",
        ),
        Family(
            family_id="train0705_step002500",
            baseline_dir=output_root / "train0705_step002500" / "baseline" / TRAIN0705_STEP002500.name,
            guided_dir=output_root / "train0705_step002500" / "guided" / TRAIN0705_STEP002500.name,
            score_baseline_dir=TRAIN0705_BASELINE_002500,
        ),
        Family(
            family_id="train0705_step007000",
            baseline_dir=output_root / "train0705_step007000" / "baseline" / TRAIN0705_STEP007000.name,
            guided_dir=output_root / "train0705_step007000" / "guided" / TRAIN0705_STEP007000.name,
            score_baseline_dir=TRAIN0705_BASELINE_007000,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/score model-weight baseline vs guided A/B on test_5.")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--families", nargs="*", default=None, help="Subset of family ids to run.")
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--main-gpu", type=int, default=5)
    parser.add_argument("--vjepa-gpu", type=int, default=6)
    parser.add_argument("--score-gpu", type=int, default=7)
    parser.add_argument("--guided-vjepa-preset", type=str, default="target_w24_s15_ratio_0025")
    parser.add_argument(
        "--guided-motion-mask-mode",
        choices=["per_frame", "temporal_union", "temporal_union_except_first"],
        default="temporal_union_except_first",
    )
    parser.add_argument(
        "--guided-use-spectral-guidance",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--guided-spectral-source", type=str, default="temporal_lowpass_residual")
    parser.add_argument("--guided-spectral-lowpass-ratio", type=float, default=0.18)
    parser.add_argument("--guided-spectral-normalize-percentile", type=float, default=95.0)
    parser.add_argument("--guided-spectral-weight-floor", type=float, default=0.25)
    parser.add_argument("--guided-spectral-weight-scale", type=float, default=1.0)
    parser.add_argument("--guided-spectral-mask-dilation", type=int, default=5)
    parser.add_argument("--guided-preview-downsample-factor", type=int, default=None)
    parser.add_argument("--guided-preview-frame-stride", type=int, default=None)
    parser.add_argument("--guided-model-suffix", type=str, default=DEFAULT_FREQGUIDE_MODEL_SUFFIX)
    parser.add_argument(
        "--reuse-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--train0705-initialize-model-on-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _dedup_list_path(output_root: Path) -> Path:
    return output_root / "inputs" / "test5_unique.txt"


def ensure_unique_input_list(*, output_root: Path, input_list: Path, limit_cases: int | None) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    dedup_list = _dedup_list_path(output_root)
    dedup_list.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in input_list.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
        if limit_cases is not None and len(ordered) >= max(0, int(limit_cases)):
            break
    dedup_list.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return dedup_list


def build_guided_extra_args(
    args: argparse.Namespace,
    *,
    motion_mask_flag: str = "--vjepa-motion-mask-mode",
) -> list[str]:
    extra = [
        str(motion_mask_flag),
        str(args.guided_motion_mask_mode),
    ]
    if bool(args.guided_use_spectral_guidance):
        extra.extend(
            [
                "--vjepa-use-spectral-guidance",
                "--vjepa-spectral-source",
                str(args.guided_spectral_source),
                "--vjepa-spectral-lowpass-ratio",
                str(float(args.guided_spectral_lowpass_ratio)),
                "--vjepa-spectral-normalize-percentile",
                str(float(args.guided_spectral_normalize_percentile)),
                "--vjepa-spectral-weight-floor",
                str(float(args.guided_spectral_weight_floor)),
                "--vjepa-spectral-weight-scale",
                str(float(args.guided_spectral_weight_scale)),
                "--vjepa-spectral-mask-dilation",
                str(max(0, int(args.guided_spectral_mask_dilation))),
            ]
        )
    if args.guided_preview_downsample_factor is not None:
        extra.extend(
            [
                "--vjepa-preview-downsample-factor",
                str(max(1, int(args.guided_preview_downsample_factor))),
            ]
        )
    if args.guided_preview_frame_stride is not None:
        extra.extend(
            [
                "--vjepa-preview-frame-stride",
                str(max(1, int(args.guided_preview_frame_stride))),
            ]
        )
    return extra


def build_explicit_vjepa_args_from_preset_name(preset_name: str) -> list[str]:
    preset = resolve_train0705_preset(str(preset_name))
    args: list[str] = [
        "--vjepa-guidance-mode",
        str(preset.vjepa_guidance_mode),
        "--vjepa-guidance-steps",
        str(int(preset.vjepa_guidance_steps)),
        "--vjepa-min-step-percent",
        str(float(preset.vjepa_min_step_percent)),
        "--vjepa-max-step-percent",
        str(float(preset.vjepa_max_step_percent)),
        "--vjepa-latent-step-size",
        str(float(preset.vjepa_latent_step_size)),
        "--vjepa-inner-k",
        str(int(preset.vjepa_inner_k)),
        "--vjepa-preview-downsample-factor",
        str(int(preset.vjepa_preview_downsample_factor)),
        "--vjepa-preview-frame-stride",
        str(int(preset.vjepa_preview_frame_stride)),
        "--vjepa-window-size",
        str(int(preset.vjepa_window_size)),
        "--vjepa-context-frames",
        str(int(preset.vjepa_context_frames)),
        "--vjepa-stride",
        str(int(preset.vjepa_stride)),
        "--vjepa-reduction",
        str(preset.vjepa_reduction),
        "--vjepa-grad-norm-mode",
        str(preset.vjepa_grad_norm_mode),
        "--vjepa-max-grad-norm",
        str(float(preset.vjepa_max_grad_norm)),
        "--vjepa-max-correction-ratio",
        str(float(preset.vjepa_max_correction_ratio)),
        "--vjepa-artifact-guard-mode",
        str(preset.vjepa_artifact_guard_mode),
    ]
    if preset.vjepa_target_step_indices:
        args.extend(
            [
                "--vjepa-target-step-indices",
                *[str(int(value)) for value in preset.vjepa_target_step_indices],
            ]
        )
    if preset.vjepa_target_timesteps:
        args.extend(
            [
                "--vjepa-target-timesteps",
                *[str(int(value)) for value in preset.vjepa_target_timesteps],
            ]
        )
    if bool(preset.vjepa_backtracking):
        args.append("--vjepa-backtracking")
    if float(preset.vjepa_stay_close_max_video_l1) > 0:
        args.extend(
            [
                "--vjepa-stay-close-max-video-l1",
                str(float(preset.vjepa_stay_close_max_video_l1)),
            ]
        )
    return args


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{DIFFSYNTH_ROOT}"
    env["WAN22_OFFICIAL_REPO"] = str(WAN22_OFFICIAL_REPO)
    return env


def run_cmd(cmd: list[str], *, env: dict[str, str], label: str, continue_on_error: bool) -> None:
    print(f"[run] {label}", flush=True)
    print(subprocess.list2cmdline(cmd), flush=True)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        if continue_on_error:
            print(f"[warn] {label} failed with returncode={result.returncode}", flush=True)
            return
        raise SystemExit(result.returncode)


def _try_symlink(src: Path, dst: Path) -> bool:
    try:
        dst.symlink_to(src, target_is_directory=src.is_dir())
        return True
    except OSError:
        return False


def _count_case_artifacts(root: Path) -> int:
    if not root.exists():
        return 0
    count = 0
    for pattern in ("*.mp4", "*.json"):
        count += len(list(root.glob(pattern)))
    return count


def reuse_baseline_dir(*, src: Path, dst: Path, family_id: str) -> bool:
    src = src.expanduser().resolve()
    dst = dst.expanduser().resolve()
    if not src.is_dir():
        print(f"[baseline-reuse] family={family_id} source missing: {src}", flush=True)
        return False
    src_artifact_count = _count_case_artifacts(src)
    if dst.exists():
        if dst.is_symlink():
            print(f"[baseline-reuse] family={family_id} target already exists: {dst}", flush=True)
            return True
        artifact_count = _count_case_artifacts(dst)
        if artifact_count > 0 and (src_artifact_count <= 0 or artifact_count >= src_artifact_count):
            print(
                f"[baseline-reuse] family={family_id} target already exists with {artifact_count} case artifacts: {dst}",
                flush=True,
            )
            return True
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if _try_symlink(src, dst):
        mode = "symlink"
    else:
        shutil.copytree(src, dst)
        mode = "copytree"
    manifest = {
        "family_id": family_id,
        "reused_from": str(src),
        "reuse_mode": mode,
    }
    marker = dst.parent / f"{dst.name}_reuse.json"
    marker.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[baseline-reuse] family={family_id} {mode}: {src} -> {dst}", flush=True)
    return True


def maybe_reuse_baseline(*, family_id: str, baseline_dir: Path, force: bool) -> bool:
    if force:
        return False
    if family_id == "wan22_official_ti2v5b":
        return reuse_baseline_dir(src=OFFICIAL_BASELINE_SOURCE, dst=baseline_dir, family_id=family_id)
    if family_id == "wan22_early_lora_step000500":
        return reuse_baseline_dir(src=LORA_BASELINE_SOURCE, dst=baseline_dir, family_id=family_id)
    if family_id == "train0705_step002500":
        return reuse_baseline_dir(src=TRAIN0705_BASELINE_002500, dst=baseline_dir, family_id=family_id)
    if family_id == "train0705_step007000":
        return reuse_baseline_dir(src=TRAIN0705_BASELINE_007000, dst=baseline_dir, family_id=family_id)
    return False


def wanti2v_generate(unique_list: Path, output_root: Path, args: argparse.Namespace) -> None:
    env = base_env()
    env["CUDA_VISIBLE_DEVICES"] = f"{args.main_gpu},{args.vjepa_gpu}"
    baseline_dir = output_root / "wan22_official_ti2v5b" / "baseline"
    guided_dir = output_root / "wan22_official_ti2v5b" / "guided"
    baseline_cmd = [
        str(PY_WAN_CU128),
        str(GUIDANCE_DIR / "wanti2v.py"),
        "--input-list",
        str(unique_list),
        "--output-root",
        str(baseline_dir),
        "--model-name",
        "wan22_official_ti2v5b_baseline",
        "--backend",
        "official",
        "--wan-root",
        str(WAN22_ROOT),
        "--size",
        "704*1280",
        "--frame-num",
        "49",
        "--sampling-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--fps",
        "30",
        "--seed",
        "42",
        "--offload-model",
        "--vjepa-preset",
        "baseline",
    ]
    guided_cmd = [
        str(PY_WAN_CU128),
        str(GUIDANCE_DIR / "wanti2v.py"),
        "--input-list",
        str(unique_list),
        "--output-root",
        str(guided_dir),
        "--model-name",
        f"wan22_official_ti2v5b_{args.guided_model_suffix}",
        "--backend",
        "official",
        "--wan-root",
        str(WAN22_ROOT),
        "--size",
        "704*1280",
        "--frame-num",
        "49",
        "--sampling-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--fps",
        "30",
        "--seed",
        "42",
        "--offload-model",
        "--vjepa-preset",
        str(args.guided_vjepa_preset),
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
        "--vjepa-device-id",
        "1",
    ]
    guided_cmd.extend(build_guided_extra_args(args, motion_mask_flag="--motion-mask-mode"))
    baseline_reused = False
    if bool(args.reuse_baselines):
        baseline_reused = maybe_reuse_baseline(
            family_id="wan22_official_ti2v5b",
            baseline_dir=baseline_dir,
            force=bool(args.force),
        )
    if args.force:
        baseline_cmd.append("--force")
        guided_cmd.append("--force")
    if not baseline_reused:
        run_cmd(baseline_cmd, env=env, label="wan22_official_ti2v5b/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label="wan22_official_ti2v5b/guided", continue_on_error=args.continue_on_error)


def lora_generate(unique_list: Path, output_root: Path, args: argparse.Namespace) -> None:
    env = base_env()
    env["CUDA_VISIBLE_DEVICES"] = f"{args.main_gpu},{args.vjepa_gpu}"
    script = GUIDANCE_DIR / "wan_openvid_0613pybullet_lorav2v_vjepa.py"
    common = [
        str(PY_WAN_CU128),
        str(script),
        "--weights-root",
        str(LORA_STEP000500),
        "--input-json-list-path",
        str(unique_list),
        "--wan-root",
        str(WAN22_ROOT),
        "--device",
        "cuda:0",
        "--vjepa-device",
        "cuda:1",
        "--num-frames",
        "49",
        "--context-frames",
        "8",
        "--num-inference-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--quality",
        "5",
        "--conditioning-mode",
        "context_aware",
        "--context-resize-mode",
        "auto",
        "--log-level",
        "INFO",
    ]
    baseline_cmd = common + [
        "--model-name",
        "wan22_early_lora_step000500_baseline",
        "--output-root",
        str(output_root / "wan22_early_lora_step000500" / "baseline"),
        "--runtime-root",
        str(output_root / "wan22_early_lora_step000500" / "baseline_runtime"),
        "--disable-vjepa-guidance",
    ]
    guided_cmd = common + [
        "--model-name",
        f"wan22_early_lora_step000500_{args.guided_model_suffix}",
        "--output-root",
        str(output_root / "wan22_early_lora_step000500" / "guided"),
        "--runtime-root",
        str(output_root / "wan22_early_lora_step000500" / "guided_runtime"),
        "--vjepa-model",
        "vith",
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
    ]
    guided_cmd.extend(build_explicit_vjepa_args_from_preset_name(str(args.guided_vjepa_preset)))
    guided_cmd.extend(build_guided_extra_args(args))
    if args.guided_preview_downsample_factor is None:
        guided_cmd.extend(["--vjepa-preview-downsample-factor", "8"])
    if args.guided_preview_frame_stride is None:
        guided_cmd.extend(["--vjepa-preview-frame-stride", "2"])
    baseline_reused = False
    if bool(args.reuse_baselines):
        baseline_reused = maybe_reuse_baseline(
            family_id="wan22_early_lora_step000500",
            baseline_dir=output_root / "wan22_early_lora_step000500" / "baseline",
            force=bool(args.force),
        )
    if args.force:
        baseline_cmd.append("--overwrite")
        guided_cmd.append("--overwrite")
    if not baseline_reused:
        run_cmd(baseline_cmd, env=env, label="wan22_early_lora_step000500/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label="wan22_early_lora_step000500/guided", continue_on_error=args.continue_on_error)


def train0705_generate(unique_list: Path, weights_root: Path, family_root: Path, family_name: str, args: argparse.Namespace) -> None:
    env = base_env()
    env["CUDA_VISIBLE_DEVICES"] = f"{args.main_gpu},{args.vjepa_gpu}"
    script = ROOT / "code_vjepa_vggt" / "train0705" / "wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py"
    common = [
        str(PY_WAN_CU128),
        str(script),
        "--weights-root",
        str(weights_root),
        "--input-json-list-path",
        str(unique_list),
        "--device",
        "cuda:0",
        "--wan-root",
        str(WAN22_ROOT),
        "--diffsynth-root",
        str(DIFFSYNTH_ROOT),
        "--height",
        "512",
        "--width",
        "896",
        "--num-frames",
        "24",
        "--context-frames",
        "8",
        "--fps",
        "30",
        "--sampling-mode",
        "prefix",
        "--num-inference-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--seed",
        "42",
        "--quality",
        "5",
        "--vjepa-device",
        "cuda:1",
    ]
    baseline_cmd = common + [
        "--model-name",
        f"{family_name}_baseline",
        "--output-root",
        str(family_root / "baseline"),
        "--vjepa-preset",
        "baseline",
    ]
    guided_cmd = common + [
        "--model-name",
        f"{family_name}_{args.guided_model_suffix}",
        "--output-root",
        str(family_root / "guided"),
        "--vjepa-preset",
        str(args.guided_vjepa_preset),
    ]
    guided_cmd.extend(build_guided_extra_args(args))
    baseline_dir = family_root / "baseline" / weights_root.name
    baseline_reused = False
    if bool(args.reuse_baselines):
        baseline_reused = maybe_reuse_baseline(
            family_id=family_name,
            baseline_dir=baseline_dir,
            force=bool(args.force),
        )
    if args.force:
        baseline_cmd.append("--overwrite")
        guided_cmd.append("--overwrite")
    if bool(args.train0705_initialize_model_on_cpu):
        baseline_cmd.append("--initialize-model-on-cpu")
        guided_cmd.append("--initialize-model-on-cpu")
    if not baseline_reused:
        run_cmd(baseline_cmd, env=env, label=f"{family_name}/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label=f"{family_name}/guided", continue_on_error=args.continue_on_error)


def generate(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    families = build_families(output_root)
    unique_list = ensure_unique_input_list(output_root=output_root, input_list=input_list, limit_cases=args.limit_cases)
    selected = set(args.families or [family.family_id for family in families])
    if "wan22_official_ti2v5b" in selected:
        wanti2v_generate(unique_list, output_root, args)
    if "wan22_early_lora_step000500" in selected:
        lora_generate(unique_list, output_root, args)
    if "train0705_step002500" in selected:
        train0705_generate(
            unique_list,
            TRAIN0705_STEP002500,
            output_root / "train0705_step002500",
            "train0705_step002500",
            args,
        )
    if "train0705_step007000" in selected:
        train0705_generate(
            unique_list,
            TRAIN0705_STEP007000,
            output_root / "train0705_step007000",
            "train0705_step007000",
            args,
        )


def score(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    families = build_families(output_root)
    selected = set(args.families or [family.family_id for family in families])
    score_root = output_root / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, object] = {}
    for family in families:
        if family.family_id not in selected:
            continue
        out_json = score_root / f"{family.family_id}_summary.json"
        out_md = score_root / f"{family.family_id}_summary.md"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/gaoya/Code_Video/Code_data/Code_try0526:{ROOT}"
        env["CUDA_VISIBLE_DEVICES"] = str(args.score_gpu)
        baseline_dir = family.score_baseline_dir if family.score_baseline_dir and family.score_baseline_dir.is_dir() else family.baseline_dir
        print(f"[score] {family.family_id} baseline_dir={baseline_dir}", flush=True)
        cmd = [
            str(PY_WAN),
            str(GUIDANCE_DIR / "score_multicase_allmetrics.py"),
            "--method-dir",
            f"baseline={baseline_dir}",
            "--method-dir",
            f"guided={family.guided_dir}",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--baseline-label",
            "baseline",
            "--videophy2-task",
            "pc",
            "--videophy2-device",
            "cuda:0",
            "--pmf-device",
            "cpu",
            "--phyground-general-only",
        ]
        run_cmd(cmd, env=env, label=f"score/{family.family_id}", continue_on_error=args.continue_on_error)
        if out_json.exists():
            combined[family.family_id] = json.loads(out_json.read_text(encoding="utf-8"))

    combined_path = score_root / "combined_summary.json"
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate(args)
    if args.stage in {"score", "all"}:
        score(args)


if __name__ == "__main__":
    main()
