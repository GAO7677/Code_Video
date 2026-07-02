#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

try:
    from .experiment_presets import ROUND1_MODES
    from .build_experiment_manifest import write_manifest
except ImportError:
    from experiment_presets import ROUND1_MODES
    from build_experiment_manifest import write_manifest


SOURCE_DIR = Path("/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons")
RESULTS_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results")
ROUND_NAME = "v2v_jsons_full_wan22"


def iter_cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []
    for json_path in sorted(SOURCE_DIR.glob("*.json")):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image = data.get("input_image")
        prompt = data.get("input_caption")
        if not image or not prompt:
            continue
        cases.append((json_path.stem, data))
    return cases


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    script_path = Path(__file__).resolve().parent / "wan_ti2v_vjepa.py"
    suite_root = RESULTS_ROOT / ROUND_NAME
    for case_name, data in iter_cases():
        prompt = str(data["input_caption"]).strip()
        source_image = str(data["input_image"]).strip()
        source_json = str(SOURCE_DIR / f"{case_name}.json")
        for mode in ROUND1_MODES:
            exp_id = f"{case_name}_{mode.mode_id}_seed0"
            output_dir = suite_root / case_name / mode.mode_id
            rows.append(
                {
                    "exp_id": exp_id,
                    "round_name": ROUND_NAME,
                    "prompt_id": case_name,
                    "prompt": prompt,
                    "source_json": source_json,
                    "source_image": source_image,
                    "main_category": "",
                    "sub_category": "",
                    "physical_laws": "",
                    "seed": 0,
                    "mode_id": mode.mode_id,
                    "mode_description": mode.description,
                    "disable_vjepa_guidance": int(mode.disable_vjepa_guidance),
                    "vjepa_model": mode.vjepa_model,
                    "vjepa_guidance_steps": mode.vjepa_guidance_steps,
                    "vjepa_min_step_percent": mode.vjepa_min_step_percent,
                    "vjepa_max_step_percent": mode.vjepa_max_step_percent,
                    "vjepa_latent_step_size": mode.vjepa_latent_step_size,
                    "vjepa_preview_downsample_factor": mode.vjepa_preview_downsample_factor,
                    "vjepa_preview_frame_stride": mode.vjepa_preview_frame_stride,
                    "vjepa_window_size": mode.vjepa_window_size,
                    "vjepa_context_frames": mode.vjepa_context_frames,
                    "vjepa_stride": mode.vjepa_stride,
                    "vjepa_reduction": mode.vjepa_reduction,
                    "vjepa_grad_norm_mode": mode.vjepa_grad_norm_mode,
                    "vjepa_max_grad_norm": mode.vjepa_max_grad_norm,
                    "sample_steps": 10,
                    "sample_solver": "unipc",
                    "sample_shift": 5.0,
                    "sample_guide_scale": 5.0,
                    "frame_num": 41,
                    "size": "1280*704",
                    "device_id": 0,
                    "vjepa_device_id": 1,
                    "ckpt_dir": "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B",
                    "vjepa_ckpt": "/data/gaoya/ckpt/VJEPA2/vith.pt",
                    "python_bin": "/data/gaoya/miniconda3/envs/wan/bin/python",
                    "script_path": str(script_path),
                    "output_video": str(output_dir / f"{exp_id}.mp4"),
                    "output_json": str(output_dir / f"{exp_id}.json"),
                }
            )
    return rows


def main() -> None:
    rows = build_rows()
    suite_root = RESULTS_ROOT / ROUND_NAME
    manifests_dir = suite_root / "manifests"
    reports_dir = suite_root / "reports"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "manifest.csv"
    write_manifest(manifest_path, rows)

    metadata = {
        "round_name": ROUND_NAME,
        "source_dir": str(SOURCE_DIR),
        "results_root": str(RESULTS_ROOT),
        "num_cases": len(iter_cases()),
        "num_rows": len(rows),
        "notes": [
            "Only JSON files with both input_image and input_caption are included.",
            "Outputs are stored under results/v2v_jsons_full_wan22/<json_basename>/<mode_id>/",
            "Use offload_model + t5_cpu + convert_model_dtype when running.",
        ],
    }
    (suite_root / "suite_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (suite_root / "README.txt").write_text(
        "Full Wan2.2 TI2V suite from v2v_jsons.\n"
        "Cases include only jsons with both input_image and input_caption.\n"
        "Directory layout: results/v2v_jsons_full_wan22/<json_basename>/<mode_id>/<exp_id>.{mp4,json,log}\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(f"cases={len(iter_cases())}")
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
