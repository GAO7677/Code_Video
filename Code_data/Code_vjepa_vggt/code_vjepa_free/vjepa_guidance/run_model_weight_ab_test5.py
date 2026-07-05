#!/usr/bin/env python3
from __future__ import annotations

"""
Run the 5-model-line baseline/guided A/B on the deduplicated test_5 list and
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
import subprocess
from dataclasses import dataclass
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GUIDANCE_DIR = THIS_FILE.parent
ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
WAN22_OFFICIAL_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main_official")
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
PY_WAN_CU128 = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
PY_WAN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")

INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705")
DEDUP_LIST = OUTPUT_ROOT / "inputs" / "test5_unique.txt"

WAN22_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
WAN21_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers")
VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
LORA_STEP000500 = Path(
    "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
    "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500"
)
TRAIN0705_STEP002500 = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-002500"
)
TRAIN0705_STEP005000 = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
    "train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-005000"
)

MID12 = ["8", "10", "11", "13", "14", "16", "17", "19", "20", "22", "23", "25"]


@dataclass(frozen=True)
class Family:
    family_id: str
    baseline_dir: Path
    guided_dir: Path
    score_baseline_dir: Path | None = None


TRAIN0705_BASELINE_002500 = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-002500"
)
TRAIN0705_BASELINE_005000 = Path(
    "/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-005000"
)


FAMILIES = [
    Family(
        family_id="wan22_official_ti2v5b",
        baseline_dir=OUTPUT_ROOT / "wan22_official_ti2v5b" / "baseline",
        guided_dir=OUTPUT_ROOT / "wan22_official_ti2v5b" / "guided",
    ),
    Family(
        family_id="wan22_early_lora_step000500",
        baseline_dir=OUTPUT_ROOT / "wan22_early_lora_step000500" / "baseline",
        guided_dir=OUTPUT_ROOT / "wan22_early_lora_step000500" / "guided",
    ),
    Family(
        family_id="train0705_step002500",
        baseline_dir=OUTPUT_ROOT / "train0705_step002500" / "baseline" / TRAIN0705_STEP002500.name,
        guided_dir=OUTPUT_ROOT / "train0705_step002500" / "guided" / TRAIN0705_STEP002500.name,
        score_baseline_dir=TRAIN0705_BASELINE_002500,
    ),
    Family(
        family_id="train0705_step005000",
        baseline_dir=OUTPUT_ROOT / "train0705_step005000" / "baseline" / TRAIN0705_STEP005000.name,
        guided_dir=OUTPUT_ROOT / "train0705_step005000" / "guided" / TRAIN0705_STEP005000.name,
        score_baseline_dir=TRAIN0705_BASELINE_005000,
    ),
    Family(
        family_id="wan21_t2v_1p3b",
        baseline_dir=OUTPUT_ROOT / "wan21_t2v_1p3b" / "baseline",
        guided_dir=OUTPUT_ROOT / "wan21_t2v_1p3b" / "guided",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/score model-weight baseline vs guided A/B on test_5.")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--families", nargs="*", default=None, help="Subset of family ids to run.")
    parser.add_argument("--main-gpu", type=int, default=5)
    parser.add_argument("--vjepa-gpu", type=int, default=6)
    parser.add_argument("--score-gpu", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def ensure_unique_input_list() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DEDUP_LIST.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    DEDUP_LIST.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    return DEDUP_LIST


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


def wanti2v_generate(unique_list: Path, args: argparse.Namespace) -> None:
    env = base_env()
    env["CUDA_VISIBLE_DEVICES"] = str(args.main_gpu)
    baseline_dir = OUTPUT_ROOT / "wan22_official_ti2v5b" / "baseline"
    guided_dir = OUTPUT_ROOT / "wan22_official_ti2v5b" / "guided"
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
        "wan22_official_ti2v5b_target_w24_s15_ratio_0025",
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
        "target_w24_s15_ratio_0025",
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
        "--vjepa-device-id",
        "-1",
    ]
    if args.force:
        baseline_cmd.append("--force")
        guided_cmd.append("--force")
    run_cmd(baseline_cmd, env=env, label="wan22_official_ti2v5b/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label="wan22_official_ti2v5b/guided", continue_on_error=args.continue_on_error)


def lora_generate(unique_list: Path, args: argparse.Namespace) -> None:
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
        str(OUTPUT_ROOT / "wan22_early_lora_step000500" / "baseline"),
        "--runtime-root",
        str(OUTPUT_ROOT / "wan22_early_lora_step000500" / "baseline_runtime"),
        "--disable-vjepa-guidance",
    ]
    guided_cmd = common + [
        "--model-name",
        "wan22_early_lora_step000500_target_w24_s15_ratio_0025",
        "--output-root",
        str(OUTPUT_ROOT / "wan22_early_lora_step000500" / "guided"),
        "--runtime-root",
        str(OUTPUT_ROOT / "wan22_early_lora_step000500" / "guided_runtime"),
        "--vjepa-model",
        "vith",
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
        "--vjepa-guidance-mode",
        "context_anchored",
        "--vjepa-guidance-steps",
        "12",
        "--vjepa-target-step-indices",
        *MID12,
        "--vjepa-latent-step-size",
        "0.15",
        "--vjepa-preview-downsample-factor",
        "4",
        "--vjepa-preview-frame-stride",
        "1",
        "--vjepa-window-size",
        "24",
        "--vjepa-context-frames",
        "8",
        "--vjepa-stride",
        "4",
        "--vjepa-reduction",
        "mean",
        "--vjepa-grad-norm-mode",
        "rms",
        "--vjepa-max-grad-norm",
        "10.0",
        "--vjepa-max-correction-ratio",
        "0.025",
        "--vjepa-artifact-guard-mode",
        "none",
    ]
    if args.force:
        baseline_cmd.append("--overwrite")
        guided_cmd.append("--overwrite")
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
        f"{family_name}_target_w24_s15_ratio_0025",
        "--output-root",
        str(family_root / "guided"),
        "--vjepa-preset",
        "target_w24_s15_ratio_0025",
    ]
    if args.force:
        baseline_cmd.append("--overwrite")
        guided_cmd.append("--overwrite")
    run_cmd(baseline_cmd, env=env, label=f"{family_name}/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label=f"{family_name}/guided", continue_on_error=args.continue_on_error)


def wan21_generate(unique_list: Path, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["CUDA_VISIBLE_DEVICES"] = f"{args.main_gpu},{args.vjepa_gpu}"
    script = GUIDANCE_DIR / "wan21_t2v_1_3b_batch.py"
    common = [
        str(PY_WAN),
        str(script),
        "--input-list",
        str(unique_list),
        "--ckpt-dir",
        str(WAN21_ROOT),
        "--seed",
        "42",
        "--height",
        "480",
        "--width",
        "832",
        "--num-frames",
        "49",
        "--num-inference-steps",
        "10",
        "--guidance-scale",
        "6.0",
        "--flow-shift",
        "8.0",
        "--fps",
        "16",
        "--transformer-dtype",
        "bfloat16",
        "--vae-dtype",
        "bfloat16",
        "--device-id",
        "0",
        "--vjepa-device-id",
        "1",
        "--cpu-offload",
    ]
    baseline_cmd = common + [
        "--output-root",
        str(OUTPUT_ROOT / "wan21_t2v_1p3b" / "baseline"),
        "--model-name",
        "wan21_t2v_1p3b_baseline",
        "--disable-vjepa-guidance",
    ]
    guided_cmd = common + [
        "--output-root",
        str(OUTPUT_ROOT / "wan21_t2v_1p3b" / "guided"),
        "--model-name",
        "wan21_t2v_1p3b_guided_g2mid2s002",
        "--vjepa-model",
        "vith",
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
        "--vjepa-guidance-steps",
        "2",
        "--vjepa-min-step-percent",
        "0.35",
        "--vjepa-max-step-percent",
        "0.65",
        "--vjepa-latent-step-size",
        "0.02",
        "--preview-downsample-factor",
        "4",
        "--preview-frame-stride",
        "2",
        "--window-size",
        "8",
        "--context-frames",
        "4",
        "--stride",
        "2",
        "--reduction",
        "mean",
        "--gradient-normalization",
        "rms",
        "--max-grad-norm",
        "10.0",
    ]
    if args.force:
        baseline_cmd.append("--force")
        guided_cmd.append("--force")
    run_cmd(baseline_cmd, env=env, label="wan21_t2v_1p3b/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label="wan21_t2v_1p3b/guided", continue_on_error=args.continue_on_error)


def generate(args: argparse.Namespace) -> None:
    unique_list = ensure_unique_input_list()
    selected = set(args.families or [family.family_id for family in FAMILIES])
    if "wan22_official_ti2v5b" in selected:
        wanti2v_generate(unique_list, args)
    if "wan22_early_lora_step000500" in selected:
        lora_generate(unique_list, args)
    if "train0705_step002500" in selected:
        train0705_generate(
            unique_list,
            TRAIN0705_STEP002500,
            OUTPUT_ROOT / "train0705_step002500",
            "train0705_step002500",
            args,
        )
    if "train0705_step005000" in selected:
        train0705_generate(
            unique_list,
            TRAIN0705_STEP005000,
            OUTPUT_ROOT / "train0705_step005000",
            "train0705_step005000",
            args,
        )
    if "wan21_t2v_1p3b" in selected:
        wan21_generate(unique_list, args)


def score(args: argparse.Namespace) -> None:
    selected = set(args.families or [family.family_id for family in FAMILIES])
    score_root = OUTPUT_ROOT / "scores"
    score_root.mkdir(parents=True, exist_ok=True)
    combined: dict[str, object] = {}
    for family in FAMILIES:
        if family.family_id not in selected:
            continue
        out_json = score_root / f"{family.family_id}_summary.json"
        env = os.environ.copy()
        env["PYTHONPATH"] = f"/home/gaoya/Code_Video/Code_data/Code_try0526:{ROOT}"
        env["CUDA_VISIBLE_DEVICES"] = str(args.score_gpu)
        baseline_dir = family.score_baseline_dir if family.score_baseline_dir and family.score_baseline_dir.is_dir() else family.baseline_dir
        print(f"[score] {family.family_id} baseline_dir={baseline_dir}", flush=True)
        cmd = [
            str(PY_WAN),
            str(GUIDANCE_DIR / "score_multicase_methods.py"),
            "--method-dir",
            f"baseline={baseline_dir}",
            "--method-dir",
            f"guided={family.guided_dir}",
            "--out-json",
            str(out_json),
            "--baseline-label",
            "baseline",
            "--physics-iq",
            "--videophy2-task",
            "pc",
            "--videophy2-device",
            "cuda:0",
            "--cosmos-reason1",
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
