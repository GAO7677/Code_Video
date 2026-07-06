#!/usr/bin/env python3
from __future__ import annotations

"""
Run a small frequency-guidance sweep on test_5, reusing the 4-family A/B driver.

Generate one-case smoke for all modes:
CUDA_VISIBLE_DEVICES=6,7 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5_freqguide_sweep.py \
  --stage generate \
  --limit-cases 1 \
  --main-gpu 6 \
  --vjepa-gpu 7

Score one-case smoke for all modes:
CUDA_VISIBLE_DEVICES=6 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5_freqguide_sweep.py \
  --stage score \
  --limit-cases 1 \
  --score-gpu 6
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

try:
    from . import run_model_weight_ab_test5 as base
except ImportError:
    import run_model_weight_ab_test5 as base


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/model_weight_ab_test5_freqguide_sweep_20260706")


@dataclass(frozen=True)
class FreqGuideSweepPreset:
    mode_id: str
    description: str
    motion_mask_mode: str = "temporal_union_except_first"
    use_spectral_guidance: bool = True
    spectral_lowpass_ratio: float = 0.18
    spectral_normalize_percentile: float = 95.0
    spectral_weight_floor: float = 0.25
    spectral_weight_scale: float = 1.0
    spectral_mask_dilation: int = 5
    preview_downsample_factor: int | None = None
    preview_frame_stride: int | None = None
    guided_vjepa_preset: str = "target_w24_s15_ratio_0025"


PRESETS: tuple[FreqGuideSweepPreset, ...] = (
    FreqGuideSweepPreset(
        mode_id="mask_only_tunionx1",
        description="Motion-mask-only guided baseline: temporal union except first, no spectral weighting.",
        use_spectral_guidance=False,
        spectral_mask_dilation=0,
    ),
    FreqGuideSweepPreset(
        mode_id="freq_tunionx1_lp018_d5",
        description="Current reference frequency-guidance setting.",
    ),
    FreqGuideSweepPreset(
        mode_id="freq_tunionx1_lp018_d5_pd8_fs2",
        description="Current reference spectral weighting with more aggressive preview downsampling for OOM-sensitive families.",
        preview_downsample_factor=8,
        preview_frame_stride=2,
    ),
    FreqGuideSweepPreset(
        mode_id="freq_tunionx1_lp018_d0",
        description="Same spectral weighting as reference, but no mask dilation.",
        spectral_mask_dilation=0,
    ),
    FreqGuideSweepPreset(
        mode_id="freq_tunionx1_lp012_d3_wf010",
        description="Tighter low-pass emphasis and lower floor, aiming for more localized low-frequency correction.",
        spectral_lowpass_ratio=0.12,
        spectral_weight_floor=0.10,
        spectral_mask_dilation=3,
    ),
    FreqGuideSweepPreset(
        mode_id="freq_tunionx1_lp024_d7_wf040_ws125",
        description="Broader and stronger frequency weighting with larger spatial dilation.",
        spectral_lowpass_ratio=0.24,
        spectral_weight_floor=0.40,
        spectral_weight_scale=1.25,
        spectral_mask_dilation=7,
    ),
)

PRESET_MAP = {preset.mode_id: preset for preset in PRESETS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep frequency-guidance settings on the 4-family test_5 A/B suite.")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--mode-ids", nargs="*", default=None, help="Subset of sweep modes to run.")
    parser.add_argument("--families", nargs="*", default=None, help="Subset of family ids to run.")
    parser.add_argument("--input-list", type=Path, default=base.DEFAULT_INPUT_LIST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--main-gpu", type=int, default=5)
    parser.add_argument("--vjepa-gpu", type=int, default=6)
    parser.add_argument("--score-gpu", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--list-modes", action="store_true")
    return parser.parse_args()


def select_presets(mode_ids: list[str] | None) -> list[FreqGuideSweepPreset]:
    if not mode_ids:
        return list(PRESETS)
    selected: list[FreqGuideSweepPreset] = []
    seen: set[str] = set()
    for mode_id in mode_ids:
        try:
            preset = PRESET_MAP[str(mode_id)]
        except KeyError as exc:
            raise SystemExit(
                f"Unknown mode_id={mode_id}. Available: {', '.join(sorted(PRESET_MAP))}"
            ) from exc
        if preset.mode_id in seen:
            continue
        seen.add(preset.mode_id)
        selected.append(preset)
    return selected


def write_preset_manifest(mode_root: Path, preset: FreqGuideSweepPreset, *, input_list: Path, limit_cases: int | None) -> None:
    mode_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode_id": preset.mode_id,
        "description": preset.description,
        "input_list": str(input_list),
        "limit_cases": limit_cases,
        "preset": asdict(preset),
    }
    (mode_root / "freqguide_preset.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_mode(args: argparse.Namespace, preset: FreqGuideSweepPreset) -> None:
    mode_root = args.output_root.expanduser().resolve() / preset.mode_id
    write_preset_manifest(
        mode_root,
        preset,
        input_list=args.input_list.expanduser().resolve(),
        limit_cases=args.limit_cases,
    )
    argv = [
        str(Path(base.__file__).resolve()),
        "--stage",
        str(args.stage),
        "--input-list",
        str(args.input_list.expanduser().resolve()),
        "--output-root",
        str(mode_root),
        "--main-gpu",
        str(args.main_gpu),
        "--vjepa-gpu",
        str(args.vjepa_gpu),
        "--score-gpu",
        str(args.score_gpu),
        "--guided-vjepa-preset",
        str(preset.guided_vjepa_preset),
        "--guided-motion-mask-mode",
        str(preset.motion_mask_mode),
        "--guided-model-suffix",
        str(preset.mode_id),
        "--guided-spectral-lowpass-ratio",
        str(float(preset.spectral_lowpass_ratio)),
        "--guided-spectral-normalize-percentile",
        str(float(preset.spectral_normalize_percentile)),
        "--guided-spectral-weight-floor",
        str(float(preset.spectral_weight_floor)),
        "--guided-spectral-weight-scale",
        str(float(preset.spectral_weight_scale)),
        "--guided-spectral-mask-dilation",
        str(int(preset.spectral_mask_dilation)),
    ]
    if preset.preview_downsample_factor is not None:
        argv.extend(
            [
                "--guided-preview-downsample-factor",
                str(int(preset.preview_downsample_factor)),
            ]
        )
    if preset.preview_frame_stride is not None:
        argv.extend(
            [
                "--guided-preview-frame-stride",
                str(int(preset.preview_frame_stride)),
            ]
        )
    if args.limit_cases is not None:
        argv.extend(["--limit-cases", str(int(args.limit_cases))])
    if args.families:
        argv.extend(["--families", *[str(value) for value in args.families]])
    if preset.use_spectral_guidance:
        argv.append("--guided-use-spectral-guidance")
    else:
        argv.append("--no-guided-use-spectral-guidance")
    if args.force:
        argv.append("--force")
    if args.continue_on_error:
        argv.append("--continue-on-error")
    print(f"[sweep] mode={preset.mode_id}", flush=True)
    print(f"[sweep] description={preset.description}", flush=True)
    sys.argv = argv
    base.main()


def main() -> None:
    args = parse_args()
    if args.list_modes:
        for preset in PRESETS:
            print(f"{preset.mode_id}: {preset.description}", flush=True)
        return
    for preset in select_presets(args.mode_ids):
        run_mode(args, preset)


if __name__ == "__main__":
    main()
