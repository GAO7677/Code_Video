#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

try:
    from .experiment_presets import ROUND1_MODES
    from .build_experiment_manifest import write_manifest
except ImportError:
    from experiment_presets import ROUND1_MODES
    from build_experiment_manifest import write_manifest


RESULTS_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results")
ROUND_NAME = "videophy_62_wan21_1p3b"
PROMPT_ID = "videophy_0062"
PROMPT = "A steel ball bearing rolls across a flat surface and collides with another ball bearing, both ceasing motion."


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    script_path = Path(__file__).resolve().parent / "wan21_t2v_1_3b_vjepa.py"
    for mode in ROUND1_MODES:
        exp_id = f"{ROUND_NAME}_{PROMPT_ID}_{mode.mode_id}_seed0"
        output_dir = RESULTS_ROOT / ROUND_NAME / mode.mode_id / PROMPT_ID
        rows.append(
            {
                "exp_id": exp_id,
                "round_name": ROUND_NAME,
                "prompt_id": PROMPT_ID,
                "prompt": PROMPT,
                "source_json": "",
                "source_image": "",
                "main_category": "Force",
                "sub_category": "Collision",
                "physical_laws": "collision; impenetrability; momentum; material",
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
                "height": 480,
                "width": 832,
                "num_frames": 49,
                "num_inference_steps": 10,
                "guidance_scale": 6.0,
                "flow_shift": 8.0,
                "fps": 16,
                "transformer_dtype": "bfloat16",
                "vae_dtype": "bfloat16",
                "device_id": 0,
                "vjepa_device_id": 2,
                "ckpt_dir": "/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers",
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
    (suite_root / "manifests").mkdir(parents=True, exist_ok=True)
    manifest_path = suite_root / "manifests" / "manifest.csv"
    write_manifest(manifest_path, rows)

    readme = suite_root / "README.txt"
    readme.write_text(
        "T2V smoke suite for videophy_0062 with Wan2.1 T2V 1.3B.\n"
        "No first-frame image is used.\n"
        "Use cpu_offload when running.\n",
        encoding="utf-8",
    )
    print(manifest_path)
    print(f"rows={len(rows)}")


if __name__ == "__main__":
    main()
