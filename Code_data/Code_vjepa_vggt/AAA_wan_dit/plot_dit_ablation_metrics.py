#!/usr/bin/env python3
"""Summarize and plot the 67-case Wan DiT ablation metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan")
DEFAULT_ALLOWLIST = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
)
DEFAULT_OUTPUT_DIR = DEFAULT_ROOT / "_metric_plots"
DEFAULT_PHYRVG_VALIDATION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physrvg_full_sa_train_validation_30cases"
)
DEFAULT_PHYRVG_TEST5_REFERENCE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physrvg_test5_lora_onoff_cfg40_20260808"
)
METHOD_PATTERN = re.compile(
    r"^(whole_block|self_attn_zero|object_cross_attn|"
    r"text_cross_attn_zero|ffn_zero|lora_off)_block(\d{2})$"
)
TRAINING_CHECKPOINT_PATTERN = re.compile(
    r"^xssc_lora_("
    r"object_only|"
    r"full_sa|full_sa_resume|"
    r"s_head59|s_head59_resume|"
    r"t_head70|t_head70_resume|"
    r"t_head70_no_object|"
    r"t_head70_slot_dedup_merge|"
    r"t_head70_slot_dedup_merge_xssc_step050000|"
    r"t_head100_lora_pck32_no_object|"
    r"slot_dedup_merge|"
    r"slot_dedup_merge_xssc_step050000|"
    r"full_sa_no_object|"
    r"full_sa_no_object_pybullet100|"
    r"full_sa_no_object_kubric100|"
    r"full_sa_no_object_vjepa_loss|"
    r"full_sa_no_object_xssc_loss_dinov3_movic_step50000|"
    r"full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000|"
    r"full_sa_no_object_cotracker_trajectory_loss|"
    r"full_sa_no_object_gt_latent_mask_loss"
    r"|full_sa_physrvg_dit"
    r"|full_sa_physrvg_dit_gpu56"
    r"|full_sa_physrvg_vjepa_loss"
    r"|full_sa_physrvg_latent_mask_loss"
    r"|full_sa_physrvg_vjepa_loss_0613_b2g2"
    r"|full_sa_physrvg_object_xssc_loss"
    r"|full_sa_physrvg_no_vjepa_0717_b2g2"
    r"|full_sa_physrvg_vjepa_rect384x672_0717_b2g2"
    r")_step-(\d+)_steps\d+_\d+x\d+_ctx\d+_\d+f(?:_.+)?$"
)
TRAINING_VARIANT_ALIASES = {
    "full_sa_resume": "full_sa",
    "s_head59_resume": "s_head59",
    "t_head70_resume": "t_head70",
}
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODE_LABELS = {
    "baseline": "Baseline",
    "whole_block": "Whole block bypass",
    "self_attn_zero": "Self-attention output = 0",
    "object_cross_attn": "Object cross-attention output = 0",
    "text_cross_attn_zero": "Text cross-attention output = 0",
    "ffn_zero": "FFN output = 0",
    "lora_off": "LoRA disabled",
}
MODE_ORDER = {
    mode: index
    for index, mode in enumerate(
        (
            "baseline",
            "whole_block",
            "self_attn_zero",
            "object_cross_attn",
            "text_cross_attn_zero",
            "ffn_zero",
            "lora_off",
        )
    )
}
MODEL_STYLES = {
    "wan_lora": {"linestyle": "--", "marker": "o"},
    "xssc": {"linestyle": "-", "marker": "s"},
    "physrvg": {"linestyle": "-", "marker": "D"},
}
MODE_COLORS = {
    "whole_block": "#E41A1C",
    "self_attn_zero": "#2166D1",
    "object_cross_attn": "#009E73",
    "text_cross_attn_zero": "#7B2CBF",
    "ffn_zero": "#E67E22",
    "lora_off": "#5D6D7E",
}
TRAINING_VARIANT_LABELS = {
    "object_only": "Object-only",
    "full_sa": "Full-SA + Object",
    "s_head59": "S-head59 + Object",
    "t_head70": "T-head70 + Object",
    "t_head70_no_object": "T-head70 + No-Object",
    "t_head70_slot_dedup_merge": "T-head70 + Object + Slot-Dedup",
    "t_head70_slot_dedup_merge_xssc_step050000": (
        "T-head70 + Object + Slot-Dedup (xSSC-50k)"
    ),
    "t_head100_lora_pck32_no_object": (
        "Motion-head100 (LoRA-PCK32 Top100) + No-Object"
    ),
    "slot_dedup_merge": "Full-SA + Object + Slot-Dedup",
    "slot_dedup_merge_xssc_step050000": (
        "Full-SA + Object + Slot-Dedup (xSSC-50k)"
    ),
    "full_sa_no_object": "Full-SA + No-Object",
    "full_sa_no_object_pybullet100": "Full-SA + No-Object (PyBullet 100%)",
    "full_sa_no_object_kubric100": "Full-SA + No-Object (Kubric 100%)",
    "full_sa_no_object_vjepa_loss": "Full-SA + No-Object + V-JEPA Loss",
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": (
        "Full-SA + No-Object + xSSC Loss (DINOv3 MOVi-C 50k)"
    ),
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": (
        "Full-SA + Object + Slot-Dedup (xSSC-50k) + xSSC Loss (DINOv3 MOVi-C 50k)"
    ),
    "full_sa_no_object_cotracker_trajectory_loss": (
        "Full-SA + No-Object + CoTracker Trajectory Loss"
    ),
    "full_sa_no_object_gt_latent_mask_loss": (
        "Full-SA + No-Object + GT Latent-Mask CE"
    ),
    "full_sa_physrvg_dit": "PHYRVG-Full-SA (PhysRVG DiT)",
    "full_sa_physrvg_dit_gpu56": "PHYRVG-Full-SA · GPU5/6 batch",
    "full_sa_physrvg_vjepa_loss": "PHYRVG-Full-SA + V-JEPA Loss",
    "full_sa_physrvg_latent_mask_loss": "PHYRVG-Full-SA + Latent-Mask Loss",
    "full_sa_physrvg_vjepa_loss_0613_b2g2": (
        "PHYRVG-Full-SA + V-JEPA Loss · 0613 · b2-gacc2"
    ),
    "full_sa_physrvg_object_xssc_loss": "PHYRVG-Full-SA + Object + XSSC Loss",
    "full_sa_physrvg_no_vjepa_0717_b2g2": "PHYRVG-Full-SA · 0717 ·",
    "full_sa_physrvg_vjepa_rect384x672_0717_b2g2": (
        "PHYRVG-Full-SA + V-JEPA Loss · Rect384×672 · 0717 · b2-gacc2"
    ),
}
TRAINING_VARIANT_COLORS = {
    "object_only": "#4D4D4D",
    "full_sa": "#D62728",
    "s_head59": "#2CA02C",
    "t_head70": "#9467BD",
    "t_head70_no_object": "#E377C2",
    "t_head70_slot_dedup_merge": "#17BECF",
    "t_head70_slot_dedup_merge_xssc_step050000": "#00B894",
    "t_head100_lora_pck32_no_object": "#0072B2",
    "slot_dedup_merge": "#1F77B4",
    "slot_dedup_merge_xssc_step050000": "#8C564B",
    "full_sa_no_object": "#FF7F0E",
    "full_sa_no_object_pybullet100": "#00A6A6",
    "full_sa_no_object_kubric100": "#F28E2B",
    "full_sa_no_object_vjepa_loss": "#E45756",
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": "#4C78A8",
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": "#72B7B2",
    "full_sa_no_object_cotracker_trajectory_loss": "#B279A2",
    "full_sa_no_object_gt_latent_mask_loss": "#FF9DA6",
    "full_sa_physrvg_dit": "#355C7D",
    "full_sa_physrvg_dit_gpu56": "#6C5B7B",
    "full_sa_physrvg_vjepa_loss": "#C06C84",
    "full_sa_physrvg_latent_mask_loss": "#F67280",
    "full_sa_physrvg_vjepa_loss_0613_b2g2": "#F8B195",
    "full_sa_physrvg_object_xssc_loss": "#2A9D8F",
    "full_sa_physrvg_no_vjepa_0717_b2g2": "#F28E2B",
    "full_sa_physrvg_vjepa_rect384x672_0717_b2g2": "#76B7B2",
}
TRAINING_VARIANT_MARKERS = {
    "object_only": "P",
    "full_sa": "o",
    "s_head59": "s",
    "t_head70": "^",
    "t_head70_no_object": "h",
    "t_head70_slot_dedup_merge": "v",
    "t_head70_slot_dedup_merge_xssc_step050000": "<",
    "t_head100_lora_pck32_no_object": "8",
    "slot_dedup_merge": "D",
    "slot_dedup_merge_xssc_step050000": ">",
    "full_sa_no_object": "X",
    "full_sa_no_object_pybullet100": "p",
    "full_sa_no_object_kubric100": "*",
    "full_sa_no_object_vjepa_loss": "H",
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": "h",
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": "d",
    "full_sa_no_object_cotracker_trajectory_loss": "8",
    "full_sa_no_object_gt_latent_mask_loss": "P",
    "full_sa_physrvg_dit": "o",
    "full_sa_physrvg_dit_gpu56": "s",
    "full_sa_physrvg_vjepa_loss": "^",
    "full_sa_physrvg_latent_mask_loss": "D",
    "full_sa_physrvg_vjepa_loss_0613_b2g2": "P",
    "full_sa_physrvg_object_xssc_loss": "X",
    "full_sa_physrvg_no_vjepa_0717_b2g2": "o",
    "full_sa_physrvg_vjepa_rect384x672_0717_b2g2": "v",
}
TRAINING_VARIANT_LINESTYLES = {
    "object_only": "--",
    "full_sa": "-",
    "s_head59": "--",
    "t_head70": "-.",
    "t_head70_no_object": (0, (3, 2)),
    "t_head70_slot_dedup_merge": (0, (3, 1, 1, 1)),
    "t_head70_slot_dedup_merge_xssc_step050000": (0, (5, 1, 1, 1)),
    "t_head100_lora_pck32_no_object": (0, (1, 1)),
    "slot_dedup_merge": ":",
    "slot_dedup_merge_xssc_step050000": (0, (5, 2, 1, 2)),
    "full_sa_no_object": (0, (5, 1)),
    "full_sa_no_object_pybullet100": (0, (4, 1)),
    "full_sa_no_object_kubric100": (0, (2, 1)),
    "full_sa_no_object_vjepa_loss": (0, (6, 1)),
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000": (0, (4, 2, 1, 2)),
    "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000": (0, (2, 1)),
    "full_sa_no_object_cotracker_trajectory_loss": (0, (1, 2)),
    "full_sa_no_object_gt_latent_mask_loss": (0, (7, 1, 1, 1)),
    "full_sa_physrvg_dit": "-",
    "full_sa_physrvg_dit_gpu56": "--",
    "full_sa_physrvg_vjepa_loss": "-.",
    "full_sa_physrvg_latent_mask_loss": (0, (5, 1)),
    "full_sa_physrvg_vjepa_loss_0613_b2g2": (0, (3, 1, 1, 1)),
    "full_sa_physrvg_object_xssc_loss": (0, (1, 1)),
    "full_sa_physrvg_no_vjepa_0717_b2g2": "-",
    "full_sa_physrvg_vjepa_rect384x672_0717_b2g2": (0, (7, 1, 1, 1)),
}
TRAINING_VARIANT_ORDER = {
    variant: index
    for index, variant in enumerate(
        (
            "object_only",
            "full_sa",
            "s_head59",
            "t_head70",
            "t_head70_no_object",
            "t_head70_slot_dedup_merge",
            "t_head70_slot_dedup_merge_xssc_step050000",
            "t_head100_lora_pck32_no_object",
            "slot_dedup_merge",
            "slot_dedup_merge_xssc_step050000",
            "full_sa_no_object",
            "full_sa_no_object_pybullet100",
            "full_sa_no_object_kubric100",
            "full_sa_no_object_vjepa_loss",
            "full_sa_no_object_xssc_loss_dinov3_movic_step50000",
            "full_sa_object_slot_dedup_xssc50k_xssc_loss_dinov3_movic_step50000",
            "full_sa_no_object_cotracker_trajectory_loss",
            "full_sa_no_object_gt_latent_mask_loss",
            "full_sa_physrvg_dit",
            "full_sa_physrvg_dit_gpu56",
            "full_sa_physrvg_no_vjepa_0717_b2g2",
            "full_sa_physrvg_vjepa_loss",
            "full_sa_physrvg_latent_mask_loss",
            "full_sa_physrvg_vjepa_loss_0613_b2g2",
            "full_sa_physrvg_object_xssc_loss",
            "full_sa_physrvg_vjepa_rect384x672_0717_b2g2",
        )
    )
}

BASELINE_SPECS = (
    {
        "key": "wan22_base",
        "label": "Wan2.2-TI2V-5B",
        "color": "#222222",
        "result_dir": Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/"
            "physicIQ/basemodel/"
            "wan2p2_ti2v5B_aligned49_steps40_512x896_49f_defaultnegprompt"
        ),
    },
    {
        "key": "openvid_pybullet_lora_step500",
        "label": "OpenVid+PyBullet LoRA · step 500",
        "color": "#E69F00",
        "result_dir": Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/"
            "physicIQ/loramodel/"
            "wan_openvid_0613pybullet_lorav2v_step000500_aligned49_steps40_"
            "512x896_ctx08_49f_defaultnegprompt"
        ),
    },
    {
        "key": "openvid_lora_step10000",
        "label": "OpenVid LoRA · step 10000",
        "color": "#0072B2",
        "result_dir": Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/"
            "physicIQ/loramodel/"
            "wan_openvid_lorav2v_step10000_aligned49_steps40_512x896_ctx08_"
            "49f_defaultnegprompt"
        ),
    },
    {
        "key": "physrvg",
        "label": "PhysRVG",
        "color": "#CC79A7",
        "result_dir": Path(
            "/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/"
            "physicIQ/physRVG_steps40_512x896_08_49f"
        ),
    },
    {
        "key": "physrvg_finetuned_dit_lora_off_reference",
        "label": "PHYRVG-PhysRVG finetuned DiT · LoRA OFF · reference",
        "color": "#536170",
        "result_dir": DEFAULT_PHYRVG_TEST5_REFERENCE_ROOT
        / "physRVG_test5_LoRA_OFF_steps40_512x896_08_49f",
        "expected_cases": 20,
        "allow_all_result_inputs": True,
        "scope_label": "test5 · 20 cases",
        "source_page": "../physrvg-test5-lora-ablation/",
    },
    {
        "key": "physrvg_finetuned_dit_lora_on_reference",
        "label": "PHYRVG-PhysRVG finetuned DiT + LoRA · reference",
        "color": "#C26D5A",
        "result_dir": DEFAULT_PHYRVG_TEST5_REFERENCE_ROOT
        / "physRVG_test5_LoRA_ON_steps40_512x896_08_49f",
        "expected_cases": 20,
        "allow_all_result_inputs": True,
        "scope_label": "test5 · 20 cases",
        "source_page": "../physrvg-test5-lora-ablation/",
    },
)

INTERACTIVE_METRIC_PRIORITY = (
    "physics_iq_with_context",
    "videophy2_pc_raw",
    "cosmos_reason1",
)


def nested_score(*keys: str) -> Callable[[dict[str, Any]], float | None]:
    def extract(payload: dict[str, Any]) -> float | None:
        value: Any = payload
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        value = float(value)
        return value if math.isfinite(value) else None

    return extract


@dataclass(frozen=True)
class Metric:
    key: str
    title: str
    direction: str
    extract: Callable[[dict[str, Any]], float | None]


METRICS = (
    Metric(
        "physics_iq_with_context",
        "Physics-IQ with context",
        "higher",
        nested_score("physics_iq_with_context", "score"),
    ),
    Metric(
        "physics_iq_without_context",
        "Physics-IQ without context",
        "higher",
        nested_score("physics_iq_without_context", "score"),
    ),
    Metric(
        "physics_iq_verified_proxy",
        "Physics-IQ Verified proxy",
        "higher",
        nested_score("physics_iq_verified_proxy", "score"),
    ),
    Metric(
        "pmf_with_context",
        "PMF with context",
        "higher",
        nested_score("pmf_with_context", "score"),
    ),
    Metric(
        "pmf_without_context",
        "PMF without context",
        "higher",
        nested_score("pmf_without_context", "score"),
    ),
    Metric(
        "wmreward",
        "WMReward surprise",
        "lower",
        nested_score("wmreward", "surprise"),
    ),
    Metric(
        "vbench_subject_consistency",
        "VBench subject consistency",
        "higher",
        nested_score("vbench_subject_consistency", "score"),
    ),
    Metric(
        "vbench_background_consistency",
        "VBench background consistency",
        "higher",
        nested_score("vbench_background_consistency", "score"),
    ),
    Metric(
        "vbench_temporal_flickering",
        "VBench temporal flickering",
        "higher",
        nested_score("vbench_temporal_flickering", "score"),
    ),
    Metric(
        "vbench_motion_smoothness",
        "VBench motion smoothness",
        "higher",
        nested_score("vbench_motion_smoothness", "score"),
    ),
    Metric(
        "vbench_dynamic_degree",
        "VBench dynamic degree",
        "higher",
        nested_score("vbench_dynamic_degree", "score"),
    ),
    Metric(
        "vbench_aesthetic_quality",
        "VBench aesthetic quality",
        "higher",
        nested_score("vbench_aesthetic_quality", "score"),
    ),
    Metric(
        "vbench_imaging_quality",
        "VBench imaging quality",
        "higher",
        nested_score("vbench_imaging_quality", "score"),
    ),
    Metric(
        "videophy2_sa",
        "VideoPhy2 SA (generated only)",
        "higher",
        nested_score("videophy2", "sa_score"),
    ),
    Metric(
        "videophy2_pc_raw",
        "VideoPhy2 PC raw (full video)",
        "higher",
        nested_score("videophy2", "pc_raw_score"),
    ),
    Metric(
        "videophy2_joint_rate",
        "VideoPhy2 joint rate (generated only)",
        "higher",
        nested_score("videophy2", "joint_pass"),
    ),
    Metric(
        "videophy2_pc",
        "VideoPhy2 PC (generated only)",
        "higher",
        nested_score("videophy2", "pc_score"),
    ),
    Metric(
        "cosmos_reason1",
        "Cosmos-Reason1",
        "higher",
        nested_score("cosmos_reason1", "score"),
    ),
)


@dataclass(frozen=True)
class MetricStat:
    method: Method
    metric: Metric
    count: int
    mean: float | None
    complete: bool


@dataclass(frozen=True)
class Method:
    method_id: str
    model: str
    mode: str
    block_id: int | None
    result_dir: Path

    @property
    def sort_key(self) -> tuple[int, int, int]:
        model_order = {"wan_lora": 0, "xssc": 1, "physrvg": 2}.get(
            self.model, 99
        )
        block_order = -1 if self.block_id is None else self.block_id
        return model_order, block_order, MODE_ORDER[self.mode]


@dataclass(frozen=True)
class TrainingCheckpoint:
    variant: str
    step: int
    result_dir: Path

    @property
    def method_id(self) -> str:
        return f"xssc_lora/{self.variant}/step-{self.step:06d}"

    @property
    def sort_key(self) -> tuple[int, int]:
        return TRAINING_VARIANT_ORDER[self.variant], self.step


@dataclass(frozen=True)
class TrainingMetricStat:
    checkpoint: TrainingCheckpoint
    metric: Metric
    count: int
    mean: float | None
    complete: bool


@dataclass(frozen=True)
class BaselineMetricStat:
    key: str
    label: str
    color: str
    result_dir: Path
    metric: Metric
    count: int
    expected_cases: int
    mean: float | None
    complete: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute 67-case metric means and create one multi-panel DiT ablation plot."
        )
    )
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--input-txt",
        type=Path,
        default=None,
        help=(
            "Optional txt containing one result leaf directory per line. "
            "When set, only listed directories are plotted."
        ),
    )
    parser.add_argument("--input-json-allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--expected-cases", type=int, default=67)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="Write and plot only metric records complete for all expected cases.",
    )
    return parser.parse_args()


def discover_methods(root: Path) -> list[Method]:
    methods: list[Method] = []
    for model in ("wan_lora", "xssc"):
        model_root = root / model
        if not model_root.is_dir():
            continue
        for config_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            if config_dir.name.startswith("_"):
                continue
            if config_dir.name == "baseline":
                mode = "baseline"
                block_id = None
            else:
                match = METHOD_PATTERN.fullmatch(config_dir.name)
                if match is None:
                    continue
                mode = match.group(1)
                block_id = int(match.group(2))
            result_dir = config_dir if model == "wan_lora" else config_dir / "results"
            if result_dir.is_dir():
                methods.append(
                    Method(
                        method_id=f"{model}/{config_dir.name}",
                        model=model,
                        mode=mode,
                        block_id=block_id,
                        result_dir=result_dir.resolve(),
                    )
                )
    return sorted(methods, key=lambda method: method.sort_key)


def infer_model(result_dir: Path) -> str:
    path_parts = {part.lower() for part in result_dir.parts}
    if "wan_lora" in path_parts:
        return "wan_lora"
    if "xssc" in path_parts:
        return "xssc"
    if "phyrvg" in path_parts or "physrvg" in path_parts:
        return "physrvg"
    raise ValueError(f"Cannot infer model from result directory: {result_dir}")


def infer_config(result_dir: Path) -> tuple[str, int | None, str]:
    for candidate in (result_dir, *result_dir.parents):
        if candidate.name == "baseline":
            return "baseline", None, candidate.name
        match = METHOD_PATTERN.fullmatch(candidate.name)
        if match is not None:
            return match.group(1), int(match.group(2)), candidate.name
    raise ValueError(f"Cannot infer ablation config from result directory: {result_dir}")


def read_result_dirs_from_txt(path: Path) -> list[Path]:
    result_dirs = [
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not result_dirs:
        raise ValueError(f"No result directories found in {path}")
    if len(result_dirs) != len(set(result_dirs)):
        raise ValueError(f"Duplicate result directories found in {path}")
    return result_dirs


def infer_training_checkpoint(result_dir: Path) -> TrainingCheckpoint | None:
    for candidate in (result_dir, *result_dir.parents):
        match = TRAINING_CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match is not None:
            variant = TRAINING_VARIANT_ALIASES.get(match.group(1), match.group(1))
            return TrainingCheckpoint(
                variant=variant,
                step=int(match.group(2)),
                result_dir=result_dir,
            )
    return None


def discover_methods_from_txt(path: Path) -> list[Method]:
    result_dirs = read_result_dirs_from_txt(path)

    methods: list[Method] = []
    method_ids: set[str] = set()
    for result_dir in result_dirs:
        if infer_training_checkpoint(result_dir) is not None:
            continue
        model = infer_model(result_dir)
        try:
            mode, block_id, config_name = infer_config(result_dir)
        except ValueError:
            # The shared watcher leaf list also contains reference runs and
            # newer dashboard-only entries. They are represented by the
            # explicit baseline/current dashboards, not this ablation curve.
            if result_dir.name.startswith("xssc_lora_"):
                continue
            raise
        method_id = f"{model}/{config_name}"
        if method_id in method_ids:
            raise ValueError(f"Duplicate method inferred from {path}: {method_id}")
        method_ids.add(method_id)
        methods.append(
            Method(
                method_id=method_id,
                model=model,
                mode=mode,
                block_id=block_id,
                result_dir=result_dir,
            )
        )
    return sorted(methods, key=lambda method: method.sort_key)


def discover_training_checkpoints_from_txt(path: Path) -> list[TrainingCheckpoint]:
    checkpoints = [
        checkpoint
        for result_dir in read_result_dirs_from_txt(path)
        if (checkpoint := infer_training_checkpoint(result_dir)) is not None
    ]
    method_ids = [checkpoint.method_id for checkpoint in checkpoints]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError(f"Duplicate training checkpoint inferred from {path}")
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.sort_key)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def read_allowlist(path: Path) -> set[Path]:
    paths = {
        Path(line.strip()).expanduser().resolve()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return paths


def resolve_input_json(payload: dict[str, Any]) -> Path | None:
    for key in ("input_json", "case_json"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).expanduser().resolve()
    return None


def load_allowed_payloads(
    result_dir: Path, allowed_input_jsons: set[Path]
) -> dict[Path, dict[str, Any]]:
    return {
        input_json: payload
        for input_json, payload in load_result_payloads(result_dir).items()
        if input_json in allowed_input_jsons
    }


def load_result_payloads(result_dir: Path) -> dict[Path, dict[str, Any]]:
    """Load all case-result payloads from a result directory."""
    payloads: dict[Path, dict[str, Any]] = {}
    for path in sorted(result_dir.glob("*.json")):
        if path.name in {
            "summary.json",
            "result.json",
            "batch_manifest.json",
            "eval_summary.json",
        } or path.name.startswith("eval_summary_"):
            continue
        payload = load_json(path)
        if payload is None:
            continue
        input_json = resolve_input_json(payload)
        if input_json is not None:
            payloads[input_json] = payload
    return payloads


def compute_stats(
    methods: list[Method],
    allowed_input_jsons: set[Path],
    expected_cases: int,
) -> list[MetricStat]:
    stats: list[MetricStat] = []
    for method in methods:
        payloads = load_allowed_payloads(method.result_dir, allowed_input_jsons)
        for metric in METRICS:
            values = [
                value
                for payload in payloads.values()
                if (value := metric.extract(payload)) is not None
            ]
            count = len(values)
            stats.append(
                MetricStat(
                    method=method,
                    metric=metric,
                    count=count,
                    mean=float(np.mean(values)) if values else None,
                    complete=count == expected_cases,
                )
            )
    return stats


def compute_training_stats(
    checkpoints: list[TrainingCheckpoint],
    allowed_input_jsons: set[Path],
    expected_cases: int,
) -> list[TrainingMetricStat]:
    stats: list[TrainingMetricStat] = []
    for checkpoint in checkpoints:
        payloads = load_allowed_payloads(checkpoint.result_dir, allowed_input_jsons)
        for metric in METRICS:
            values = [
                value
                for payload in payloads.values()
                if (value := metric.extract(payload)) is not None
            ]
            count = len(values)
            stats.append(
                TrainingMetricStat(
                    checkpoint=checkpoint,
                    metric=metric,
                    count=count,
                    mean=float(np.mean(values)) if values else None,
                    complete=count == expected_cases,
                )
            )
    return stats


def compute_baseline_stats(
    allowed_input_jsons: set[Path], expected_cases: int
) -> list[BaselineMetricStat]:
    stats: list[BaselineMetricStat] = []
    for spec in BASELINE_SPECS:
        result_dir = spec["result_dir"].expanduser().resolve()
        baseline_expected_cases = int(spec.get("expected_cases", expected_cases))
        if bool(spec.get("allow_all_result_inputs", False)):
            payloads = load_result_payloads(result_dir)
        else:
            payloads = load_allowed_payloads(result_dir, allowed_input_jsons)
        for metric in METRICS:
            values = [
                value
                for payload in payloads.values()
                if (value := metric.extract(payload)) is not None
            ]
            count = len(values)
            stats.append(
                BaselineMetricStat(
                    key=str(spec["key"]),
                    label=str(spec["label"]),
                    color=str(spec["color"]),
                    result_dir=result_dir,
                    metric=metric,
                    count=count,
                    expected_cases=baseline_expected_cases,
                    mean=float(np.mean(values)) if values else None,
                    complete=count == baseline_expected_cases,
                )
            )
    return stats


def write_stats_csv(
    path: Path,
    stats: list[MetricStat],
    expected_cases: int,
    complete_only: bool = False,
) -> None:
    fieldnames = (
        "method_id",
        "model",
        "model_label",
        "ablation",
        "ablation_label",
        "layer",
        "metric",
        "direction",
        "score_count",
        "expected_count",
        "complete_67",
        "mean",
        "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            if complete_only and not stat.complete:
                continue
            writer.writerow(
                {
                    "method_id": stat.method.method_id,
                    "model": stat.method.model,
                    "model_label": MODEL_LABELS[stat.method.model],
                    "ablation": stat.method.mode,
                    "ablation_label": MODE_LABELS[stat.method.mode],
                    "layer": (
                        "" if stat.method.block_id is None else stat.method.block_id
                    ),
                    "metric": stat.metric.key,
                    "direction": stat.metric.direction,
                    "score_count": stat.count,
                    "expected_count": expected_cases,
                    "complete_67": stat.complete,
                    "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                    "result_dir": str(stat.method.result_dir),
                }
            )


def write_training_stats_csv(
    path: Path,
    stats: list[TrainingMetricStat],
    expected_cases: int,
    complete_only: bool = False,
) -> None:
    fieldnames = (
        "method_id",
        "variant",
        "variant_label",
        "training_step",
        "metric",
        "direction",
        "score_count",
        "expected_count",
        "complete_67",
        "mean",
        "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            if complete_only and not stat.complete:
                continue
            writer.writerow(
                {
                    "method_id": stat.checkpoint.method_id,
                    "variant": stat.checkpoint.variant,
                    "variant_label": TRAINING_VARIANT_LABELS[
                        stat.checkpoint.variant
                    ],
                    "training_step": stat.checkpoint.step,
                    "metric": stat.metric.key,
                    "direction": stat.metric.direction,
                    "score_count": stat.count,
                    "expected_count": expected_cases,
                    "complete_67": stat.complete,
                    "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                    "result_dir": str(stat.checkpoint.result_dir),
                }
            )


def write_baseline_stats_csv(
    path: Path, stats: list[BaselineMetricStat], expected_cases: int
) -> None:
    fieldnames = (
        "baseline_id",
        "baseline_label",
        "metric",
        "direction",
        "score_count",
        "expected_count",
        "complete_67",
        "mean",
        "result_dir",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for stat in stats:
            writer.writerow(
                {
                    "baseline_id": stat.key,
                    "baseline_label": stat.label,
                    "metric": stat.metric.key,
                    "direction": stat.metric.direction,
                    "score_count": stat.count,
                    "expected_count": stat.expected_cases,
                    "complete_67": stat.complete,
                    "mean": "" if stat.mean is None else f"{stat.mean:.8f}",
                    "result_dir": str(stat.result_dir),
                }
            )


def stat_index(
    stats: list[MetricStat],
) -> dict[tuple[str, str, int | None, str], MetricStat]:
    return {
        (
            stat.method.model,
            stat.method.mode,
            stat.method.block_id,
            stat.metric.key,
        ): stat
        for stat in stats
    }


def complete_value(stat: MetricStat | None) -> float:
    if stat is None or not stat.complete or stat.mean is None:
        return np.nan
    return stat.mean


def plot_metrics(
    output_png: Path,
    output_pdf: Path,
    stats: list[MetricStat],
    expected_cases: int,
    dpi: int,
    model: str,
    metrics: tuple[Metric, ...] = METRICS,
) -> dict[str, dict[str, int]]:
    indexed = stat_index(stats)
    model_methods = [
        method for method in {stat.method for stat in stats} if method.model == model
    ]
    block_ids = sorted(
        {
            method.block_id
            for method in model_methods
            if method.block_id is not None
        }
    )
    plot_modes = sorted(
        {method.mode for method in model_methods if method.mode != "baseline"},
        key=MODE_ORDER.__getitem__,
    )
    available_points = {
        (method.mode, method.block_id)
        for method in model_methods
        if method.block_id is not None
    }
    x_positions = np.arange(len(block_ids) + 1)
    x_labels = ("Baseline",) + tuple(str(block_id) for block_id in block_ids)
    num_columns = 2
    num_rows = math.ceil(len(metrics) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(19, 4.5 * num_rows),
        constrained_layout=False,
    )
    axes_flat = list(np.atleast_1d(axes).flat)
    completeness: dict[str, dict[str, int]] = {}

    for axis, metric in zip(axes_flat, metrics):
        plotted_ablation_points = 0
        total_ablation_points = 0
        baseline = indexed.get((model, "baseline", None, metric.key))
        baseline_value = complete_value(baseline)
        for mode in plot_modes:
            # Baseline is shown independently and must not connect to layer 0.
            values = [np.nan]
            for block_id in block_ids:
                stat = indexed.get((model, mode, block_id, metric.key))
                if (mode, block_id) in available_points:
                    total_ablation_points += 1
                if stat is not None and stat.complete:
                    plotted_ablation_points += 1
                values.append(complete_value(stat))
            if np.isfinite(values).any():
                style = MODEL_STYLES.get(
                    model, {"linestyle": "-", "marker": "o"}
                )
                axis.plot(
                    x_positions,
                    values,
                    color=MODE_COLORS[mode],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markersize=6,
                    linewidth=2,
                    alpha=0.95,
                    zorder=2,
                )

        if np.isfinite(baseline_value):
            axis.axhline(
                baseline_value,
                color="#202020",
                linestyle="--",
                linewidth=2,
                alpha=0.9,
                zorder=1,
            )

        completeness[metric.key] = {
            "complete_ablation_points": plotted_ablation_points,
            "expected_ablation_points": total_ablation_points,
        }
        direction_symbol = "\u2191" if metric.direction == "higher" else "\u2193"
        axis.set_title(
            f"{metric.title} ({direction_symbol})",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xticks(x_positions, x_labels)
        axis.set_xlabel("Layer", fontsize=11)
        axis.set_ylabel("Mean score", fontsize=11)
        axis.grid(axis="both", color="#D9DDDF", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        if plotted_ablation_points == 0:
            axis.text(
                0.5,
                0.5,
                f"No ablation result has {expected_cases}/{expected_cases} scores",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6A7175",
                fontsize=11,
            )

    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=MODE_COLORS[mode],
            linewidth=3,
            label=MODE_LABELS[mode],
        )
        for mode in plot_modes
    ]
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#202020",
            linewidth=2,
            linestyle="--",
            label=f"{MODEL_LABELS[model]} baseline",
        )
    )
    fig.suptitle(
        f"{MODEL_LABELS[model]} DiT Block Ablation Metrics",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=4,
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.014,
        (
            f"Only points with {expected_cases}/{expected_cases} finite case scores are "
            "shown. WMReward uses surprise (lower is better); all other metrics are "
            "higher is better."
        ),
        ha="center",
        fontsize=11,
        color="#4D5559",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.91,
        hspace=0.38,
        wspace=0.24,
    )
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return completeness


def plot_training_metrics(
    output_png: Path,
    output_pdf: Path,
    stats: list[TrainingMetricStat],
    expected_cases: int,
    dpi: int,
    metrics: tuple[Metric, ...],
) -> dict[str, dict[str, int]]:
    indexed = {
        (stat.checkpoint.variant, stat.checkpoint.step, stat.metric.key): stat
        for stat in stats
    }
    checkpoints = {stat.checkpoint for stat in stats}
    variants = sorted(
        {checkpoint.variant for checkpoint in checkpoints},
        key=TRAINING_VARIANT_ORDER.__getitem__,
    )
    steps = sorted({checkpoint.step for checkpoint in checkpoints})
    num_columns = 2
    num_rows = math.ceil(len(metrics) / num_columns)
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize=(19, 4.5 * num_rows),
        constrained_layout=False,
    )
    axes_flat = list(np.atleast_1d(axes).flat)
    completeness: dict[str, dict[str, int]] = {}

    for axis, metric in zip(axes_flat, metrics):
        complete_points = 0
        expected_points = 0
        for variant in variants:
            variant_steps = sorted(
                checkpoint.step
                for checkpoint in checkpoints
                if checkpoint.variant == variant
            )
            values: list[float] = []
            plotted_steps: list[int] = []
            for step in variant_steps:
                expected_points += 1
                stat = indexed.get((variant, step, metric.key))
                if stat is not None and stat.complete and stat.mean is not None:
                    complete_points += 1
                    plotted_steps.append(step)
                    values.append(stat.mean)
            if values:
                axis.plot(
                    plotted_steps,
                    values,
                    color=TRAINING_VARIANT_COLORS[variant],
                    linestyle=TRAINING_VARIANT_LINESTYLES[variant],
                    marker=TRAINING_VARIANT_MARKERS[variant],
                    markersize=7,
                    linewidth=2,
                    label=TRAINING_VARIANT_LABELS[variant],
                )

        completeness[metric.key] = {
            "complete_points": complete_points,
            "expected_points": expected_points,
        }
        direction_symbol = "\u2191" if metric.direction == "higher" else "\u2193"
        axis.set_title(
            f"{metric.title} ({direction_symbol})",
            fontsize=14,
            fontweight="semibold",
            pad=10,
        )
        axis.set_xlabel("Training step", fontsize=11)
        axis.set_ylabel("Mean score", fontsize=11)
        axis.set_xticks(steps)
        axis.grid(axis="both", color="#D9DDDF", linewidth=0.8, alpha=0.75)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=10)
        if complete_points == 0:
            axis.text(
                0.5,
                0.5,
                f"No result has {expected_cases}/{expected_cases} scores",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#6A7175",
                fontsize=11,
            )

    for axis in axes_flat[len(metrics) :]:
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=TRAINING_VARIANT_COLORS[variant],
            marker=TRAINING_VARIANT_MARKERS[variant],
            linestyle=TRAINING_VARIANT_LINESTYLES[variant],
            linewidth=2,
            label=TRAINING_VARIANT_LABELS[variant],
        )
        for variant in variants
    ]
    fig.suptitle(
        "Wan+xSSC LoRA Training Checkpoint Metrics",
        fontsize=24,
        fontweight="bold",
        y=0.985,
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=12,
    )
    fig.text(
        0.5,
        0.014,
        (
            f"Only points with {expected_cases}/{expected_cases} finite case scores are "
            "shown. WMReward uses surprise (lower is better); all other metrics are "
            "higher is better."
        ),
        ha="center",
        fontsize=11,
        color="#4D5559",
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.985,
        bottom=0.055,
        top=0.91,
        hspace=0.38,
        wspace=0.24,
    )
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return completeness


def write_plot_index(
    path: Path,
    plots: dict[str, dict[str, str]],
    training_plot: dict[str, str] | None,
    training_stats: list[TrainingMetricStat],
    baseline_stats: list[BaselineMetricStat],
    expected_cases: int,
    metrics: tuple[Metric, ...],
    phyrvg_validation: dict[str, Any] | None = None,
) -> None:
    metric_priority = {
        key: index for index, key in enumerate(INTERACTIVE_METRIC_PRIORITY)
    }
    interactive_metrics = tuple(
        sorted(
            metrics,
            key=lambda metric: (
                metric_priority.get(metric.key, len(metric_priority)),
                next(
                    index
                    for index, candidate in enumerate(metrics)
                    if candidate.key == metric.key
                ),
            ),
        )
    )
    indexed = {
        (stat.checkpoint.variant, stat.checkpoint.step, stat.metric.key): stat
        for stat in training_stats
    }
    checkpoints = sorted(
        {stat.checkpoint for stat in training_stats},
        key=lambda checkpoint: checkpoint.sort_key,
    )
    methods = []
    for variant in sorted(
        {checkpoint.variant for checkpoint in checkpoints},
        key=TRAINING_VARIANT_ORDER.__getitem__,
    ):
        points = []
        for checkpoint in (
            item for item in checkpoints if item.variant == variant
        ):
            point_metrics = {}
            for metric in interactive_metrics:
                stat = indexed.get((variant, checkpoint.step, metric.key))
                if stat is None:
                    continue
                point_metrics[metric.key] = {
                    "mean": stat.mean,
                    "count": stat.count,
                    "complete": stat.complete,
                }
            points.append(
                {
                    "step": checkpoint.step,
                    "result_dir": str(checkpoint.result_dir),
                    "metrics": point_metrics,
                }
            )
        methods.append(
            {
                "key": variant,
                "label": TRAINING_VARIANT_LABELS[variant],
                "color": TRAINING_VARIANT_COLORS[variant],
                "points": points,
            }
        )

    baseline_index = {
        (stat.key, stat.metric.key): stat for stat in baseline_stats
    }
    baselines = []
    for spec in BASELINE_SPECS:
        key = str(spec["key"])
        baseline_metrics = {}
        for metric in interactive_metrics:
            stat = baseline_index.get((key, metric.key))
            if stat is None:
                continue
            baseline_metrics[metric.key] = {
                "mean": stat.mean,
                "count": stat.count,
                "complete": stat.complete,
            }
        baseline_expected_cases = next(
            (
                stat.expected_cases
                for stat in baseline_stats
                if stat.key == key
            ),
            expected_cases,
        )
        baselines.append(
            {
                "key": key,
                "label": str(spec["label"]),
                "color": str(spec["color"]),
                "result_dir": str(spec["result_dir"].resolve()),
                "expected_cases": baseline_expected_cases,
                "scope_label": str(spec.get("scope_label", "")),
                "source_page": str(spec.get("source_page", "")),
                "metrics": baseline_metrics,
            }
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "expected_cases": expected_cases,
        "metrics": [
            {
                "key": metric.key,
                "title": metric.title,
                "direction": metric.direction,
            }
            for metric in interactive_metrics
        ],
        "methods": methods,
        "baselines": baselines,
        "phyrvg_validation": phyrvg_validation,
    }
    payload_text = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    (path.parent / "interactive_training_metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    static_links: list[str] = []
    if training_plot is not None:
        static_links.append(
            f'<a href="{Path(training_plot["png"]).name}">训练曲线 PNG</a>'
            f'<a href="{Path(training_plot["pdf"]).name}">训练曲线 PDF</a>'
        )
    for model, files in plots.items():
        static_links.append(
            f'<a href="{Path(files["png"]).name}">'
            f'{MODEL_LABELS[model]} 消融 PNG</a>'
        )
    template = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>PhysicIQ checkpoint 指标曲线</title>
  <style>
    :root{--ink:#202428;--muted:#626b70;--line:#d8dcdf;--surface:#fff;--bg:#f4f5f6;--accent:#0b6f73}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"Noto Sans SC",sans-serif}
    header{background:#fff;border-bottom:1px solid var(--line);padding:20px 24px}header>div,main{max-width:1760px;margin:auto}
    .header-links{display:flex;flex-wrap:wrap;gap:10px;margin-top:10px}.header-links a{margin:0;font-size:12px}
    h1{margin:0 0 7px;font-size:25px;letter-spacing:0}h2{font-size:16px;letter-spacing:0;margin:0 0 10px}
    p{margin:0;color:var(--muted);font-size:13px;line-height:1.55}main{padding:18px 24px 32px}
    .controls{display:grid;grid-template-columns:minmax(0,2fr) minmax(280px,1fr);gap:18px;padding:16px 0 18px;border-bottom:1px solid var(--line)}
    .checks{display:flex;flex-wrap:wrap;gap:8px 14px}.check{display:inline-flex;align-items:center;gap:7px;font-size:13px;line-height:20px;cursor:pointer}
    .check input{width:15px;height:15px;margin:0;accent-color:var(--accent)}.swatch{width:18px;height:3px;background:var(--c);display:inline-block}
    .swatch.base{height:0;border-top:2px dashed var(--c);background:none}.commands{display:flex;flex-wrap:wrap;align-items:center;gap:8px;margin-top:11px}
    button{border:1px solid #aeb6ba;background:#fff;color:var(--ink);padding:6px 10px;border-radius:4px;cursor:pointer;font-size:12px}button:hover{border-color:var(--accent);color:var(--accent)}
    #status{margin-left:4px;color:var(--muted);font-size:12px}.charts{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:18px}
    .chart{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:13px;min-width:0}.chart h3{font-size:14px;margin:0 0 2px;letter-spacing:0}
    .chart .direction{font-size:11px;color:var(--muted)}svg{display:block;width:100%;height:auto;min-height:270px}.legend{display:flex;flex-wrap:wrap;gap:6px 13px;margin:2px 4px 0;font-size:10px;color:#4b5357}
    .validation{margin-top:18px;padding:14px;background:#fff;border:1px solid var(--line);border-radius:6px}.validation h2{margin:0 0 4px}.validation p{margin-bottom:10px}.validation-chart{max-width:1100px}.validation-chart svg{min-height:300px}
    .legend span{display:inline-flex;align-items:center;gap:5px}.empty{padding:90px 16px;text-align:center;color:var(--muted);font-size:13px}
    details{margin-top:18px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--muted);font-size:12px}.downloads{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}.downloads a{font-size:12px;color:#155ca2}
    @media(max-width:1200px){.charts{grid-template-columns:repeat(2,minmax(0,1fr))}}@media(max-width:900px){.controls,.charts{grid-template-columns:1fr}main,header{padding-left:12px;padding-right:12px}}
  </style>
</head>
<body>
<header><div><h1>PhysicIQ · Checkpoint 指标曲线</h1><p>选择一个或多个训练方案；每条实线连接该方案全部 67/67 完整 checkpoint。六组对照结果作为虚线基线横贯训练 step；两条 reference baseline 保留各自 test5 20-case 评测范围。</p><div class="header-links"><a href="../">返回 8844 总览</a><a href="../physiciq-average-metrics/">67-case 平均指标表</a><a href="../phyrvg-full-sa-train-validation-30cases/">PHYRVG-Full-SA · 30-case 验证</a><a href="../physrvg-test5-lora-ablation/">LoRA OFF/ON reference 明细</a></div></div></header>
<main>
  <section class="controls">
    <div><h2>训练方案（可多选）</h2><div id="methodChecks" class="checks"></div><div class="commands"><button id="selectFullSA">全部 Full-SA</button><button id="selectObject">全部 +Object</button><button id="selectNoObject">全部 No-Object</button><button id="selectDedup">全部 +Slot-Dedup</button><button id="selectOther">其余方案</button><button id="selectAll">全选</button><button id="clearAll">清空</button><span id="status"></span></div></div>
    <div><h2>Baseline 横线</h2><div id="baselineChecks" class="checks"></div><p class="baseline-note">两条 reference 横线来自独立的 test5 20-case 评测，图例会标注评测范围。</p></div>
  </section>
  <section id="charts" class="charts"></section>
  <section id="phyrvgValidation" class="validation" hidden><h2>PHYRVG-Full-SA · 30-case 验证曲线</h2><p>来自 30-case 子页面的 deterministic flow MSE；只在同一验证集内比较，越低越好。该指标与上方 67-case PhysicIQ 指标保持分栏，避免混用不同评测协议。</p><div id="validationChart" class="validation-chart"></div></section>
  <details><summary>数据与历史静态图</summary><div class="downloads"><a href="interactive_training_metrics.json">交互数据 JSON</a><a href="xssc_lora_training_step_metric_stats.csv">checkpoint CSV</a><a href="physiciq_baseline_metric_stats.csv">baseline CSV</a>__STATIC_LINKS__</div></details>
</main>
<script>
const DATA=__PAYLOAD__;
const NS="http://www.w3.org/2000/svg";
const methodRoot=document.getElementById("methodChecks"), baselineRoot=document.getElementById("baselineChecks"), charts=document.getElementById("charts"), statusEl=document.getElementById("status");
function checkbox(root,item,type,checked){const label=document.createElement("label");label.className="check";const input=document.createElement("input");input.type="checkbox";input.dataset.type=type;input.value=item.key;input.checked=checked;const swatch=document.createElement("i");swatch.className="swatch"+(type==="baseline"?" base":"");swatch.style.setProperty("--c",item.color);label.append(input,swatch,document.createTextNode(item.label));root.append(label);input.addEventListener("change",render);return input}
let saved=[];try{saved=JSON.parse(localStorage.getItem("physiciq-selected-methods")||"[]")}catch(e){}
const validSaved=saved.filter(key=>DATA.methods.some(item=>item.key===key));
const methodInputs=DATA.methods.map((item,index)=>checkbox(methodRoot,item,"method",validSaved.length?validSaved.includes(item.key):index===0||/^(?:PHYRVG-)?Full-SA/.test(item.label)));
const baselineInputs=DATA.baselines.map(item=>checkbox(baselineRoot,item,"baseline",true));
document.getElementById("selectAll").onclick=()=>{methodInputs.forEach(input=>input.checked=true);render()};
document.getElementById("clearAll").onclick=()=>{methodInputs.forEach(input=>input.checked=false);render()};
function selectGroup(predicate){methodInputs.forEach(input=>{const method=DATA.methods.find(item=>item.key===input.value);input.checked=Boolean(method&&predicate(method))});render()}
document.getElementById("selectFullSA").onclick=()=>selectGroup(method=>/^(?:PHYRVG-)?Full-SA/.test(method.label));
document.getElementById("selectObject").onclick=()=>selectGroup(method=>method.label.includes("+ Object"));
document.getElementById("selectNoObject").onclick=()=>selectGroup(method=>method.key.includes("no_object"));
document.getElementById("selectDedup").onclick=()=>selectGroup(method=>method.key.includes("slot_dedup"));
document.getElementById("selectOther").onclick=()=>selectGroup(method=>!method.key.includes("no_object")&&!method.key.includes("slot_dedup"));
function svgEl(tag,attrs={},text=""){const node=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([key,value])=>node.setAttribute(key,String(value)));if(text)node.textContent=text;return node}
function fmt(value){if(Math.abs(value)>=10)return value.toFixed(2);if(Math.abs(value)>=1)return value.toFixed(3);return value.toFixed(4)}
function selected(inputs){return new Set(inputs.filter(input=>input.checked).map(input=>input.value))}
function chart(metric,methods,baselines){const records=[];methods.forEach(method=>method.points.forEach(point=>{const stat=point.metrics[metric.key];if(stat&&stat.complete&&Number.isFinite(stat.mean))records.push({method,step:point.step,value:stat.mean,count:stat.count})}));const baseRecords=baselines.map(base=>({base,stat:base.metrics[metric.key]})).filter(row=>row.stat&&row.stat.count>0&&Number.isFinite(row.stat.mean));if(!records.length&&!baseRecords.length)return null;
  const article=document.createElement("article");article.className="chart";const heading=document.createElement("h3");heading.textContent=metric.title;const direction=document.createElement("div");direction.className="direction";direction.textContent=metric.direction==="higher"?"越高越好 ↑":"越低越好 ↓";article.append(heading,direction);
  const W=760,H=330,M={l:62,r:18,t:18,b:48},iw=W-M.l-M.r,ih=H-M.t-M.b;const steps=records.map(row=>row.step);let xmin=steps.length?Math.min(...steps):0,xmax=steps.length?Math.max(...steps):1;if(xmin===xmax){const pad=Math.max(1,Math.round(Math.abs(xmin)*.05));xmin-=pad;xmax+=pad}const values=records.map(row=>row.value).concat(baseRecords.map(row=>row.stat.mean));let ymin=Math.min(...values),ymax=Math.max(...values);let ypad=(ymax-ymin)*.1;if(!ypad)ypad=Math.max(Math.abs(ymin)*.05,.01);ymin-=ypad;ymax+=ypad;
  const x=value=>M.l+(value-xmin)/(xmax-xmin)*iw,y=value=>M.t+(ymax-value)/(ymax-ymin)*ih;const svg=svgEl("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":metric.title});
  for(let i=0;i<5;i++){const value=ymin+(ymax-ymin)*i/4,py=y(value);svg.append(svgEl("line",{x1:M.l,y1:py,x2:W-M.r,y2:py,stroke:"#e1e4e6","stroke-width":1}),svgEl("text",{x:M.l-9,y:py+4,"text-anchor":"end",fill:"#697176","font-size":10},fmt(value)))}
  const unique=[...new Set(steps)].sort((a,b)=>a-b);const tickSteps=unique.length<=8?unique:unique.filter((_,i)=>i===0||i===unique.length-1||i%Math.ceil(unique.length/7)===0);tickSteps.forEach(step=>{const px=x(step);svg.append(svgEl("line",{x1:px,y1:M.t,x2:px,y2:H-M.b,stroke:"#eef0f1","stroke-width":1}),svgEl("text",{x:px,y:H-M.b+18,"text-anchor":"middle",fill:"#697176","font-size":10},String(step)))});svg.append(svgEl("line",{x1:M.l,y1:H-M.b,x2:W-M.r,y2:H-M.b,stroke:"#788086"}),svgEl("line",{x1:M.l,y1:M.t,x2:M.l,y2:H-M.b,stroke:"#788086"}),svgEl("text",{x:M.l+iw/2,y:H-9,"text-anchor":"middle",fill:"#697176","font-size":10},"Training step"));
  baseRecords.forEach(({base,stat})=>{const line=svgEl("line",{x1:M.l,y1:y(stat.mean),x2:W-M.r,y2:y(stat.mean),stroke:base.color,"stroke-width":2,"stroke-dasharray":"8 5",opacity:.9});line.append(svgEl("title",{},`${base.label}: ${fmt(stat.mean)} (${stat.count}/${base.expected_cases||DATA.expected_cases}${base.scope_label?` · ${base.scope_label}`:""})`));svg.append(line)});
  methods.forEach(method=>{const rows=records.filter(row=>row.method.key===method.key).sort((a,b)=>a.step-b.step);if(!rows.length)return;svg.append(svgEl("polyline",{points:rows.map(row=>`${x(row.step)},${y(row.value)}`).join(" "),fill:"none",stroke:method.color,"stroke-width":2.5,"stroke-linejoin":"round","stroke-linecap":"round"}));rows.forEach(row=>{const circle=svgEl("circle",{cx:x(row.step),cy:y(row.value),r:4,fill:method.color,stroke:"#fff","stroke-width":1.2});circle.append(svgEl("title",{},`${method.label} · step ${row.step}: ${fmt(row.value)} (${row.count}/${DATA.expected_cases})`));svg.append(circle)})});article.append(svg);
  const legend=document.createElement("div");legend.className="legend";methods.filter(method=>records.some(row=>row.method.key===method.key)).forEach(method=>{const item=document.createElement("span");item.innerHTML=`<i class="swatch" style="--c:${method.color}"></i>${method.label}`;legend.append(item)});baseRecords.forEach(({base})=>{const item=document.createElement("span");item.innerHTML=`<i class="swatch base" style="--c:${base.color}"></i>${base.label}${base.scope_label?` · ${base.scope_label}`:""}`;legend.append(item)});article.append(legend);return article}
function validationChart(data){if(!data||!data.series||!data.series.length)return null;const rows=data.series.flatMap(series=>(series.points||[]).filter(point=>point.complete&&Number.isFinite(point.mean)).map(point=>({series,step:point.step,value:point.mean,count:point.count})));if(!rows.length)return null;const W=1060,H=380,M={l:72,r:22,t:18,b:52},iw=W-M.l-M.r,ih=H-M.t-M.b;let xmin=Math.min(...rows.map(row=>row.step)),xmax=Math.max(...rows.map(row=>row.step));if(xmin===xmax){xmin-=1;xmax+=1}const values=rows.map(row=>row.value);let ymin=Math.min(...values),ymax=Math.max(...values),ypad=(ymax-ymin)*.12;if(!ypad)ypad=Math.max(Math.abs(ymin)*.05,.001);ymin-=ypad;ymax+=ypad;const x=value=>M.l+(value-xmin)/(xmax-xmin)*iw,y=value=>M.t+(ymax-value)/(ymax-ymin)*ih;const svg=svgEl("svg",{viewBox:`0 0 ${W} ${H}`,role:"img","aria-label":data.metric_title});for(let i=0;i<5;i++){const value=ymin+(ymax-ymin)*i/4,py=y(value);svg.append(svgEl("line",{x1:M.l,y1:py,x2:W-M.r,y2:py,stroke:"#e1e4e6","stroke-width":1}),svgEl("text",{x:M.l-9,y:py+4,"text-anchor":"end",fill:"#697176","font-size":10},fmt(value)))}const steps=[...new Set(rows.map(row=>row.step))].sort((a,b)=>a-b);const tickSteps=steps.length<=10?steps:steps.filter((_,i)=>i===0||i===steps.length-1||i%Math.ceil(steps.length/9)===0);tickSteps.forEach(step=>{const px=x(step);svg.append(svgEl("line",{x1:px,y1:M.t,x2:px,y2:H-M.b,stroke:"#eef0f1","stroke-width":1}),svgEl("text",{x:px,y:H-M.b+18,"text-anchor":"middle",fill:"#697176","font-size":10},String(step)))});svg.append(svgEl("line",{x1:M.l,y1:H-M.b,x2:W-M.r,y2:H-M.b,stroke:"#788086"}),svgEl("line",{x1:M.l,y1:M.t,x2:M.l,y2:H-M.b,stroke:"#788086"}),svgEl("text",{x:M.l+iw/2,y:H-10,"text-anchor":"middle",fill:"#697176","font-size":10},"Training step"));data.series.forEach(series=>{const points=rows.filter(row=>row.series.key===series.key).sort((a,b)=>a.step-b.step);if(!points.length)return;svg.append(svgEl("polyline",{points:points.map(row=>`${x(row.step)},${y(row.value)}`).join(" "),fill:"none",stroke:series.color,"stroke-width":2.5,"stroke-linejoin":"round","stroke-linecap":"round"}));points.forEach(row=>{const circle=svgEl("circle",{cx:x(row.step),cy:y(row.value),r:4,fill:series.color,stroke:"#fff","stroke-width":1.2});circle.append(svgEl("title",{},`${series.label} · step ${row.step}: ${fmt(row.value)} (${row.count}/${data.expected_cases})`));svg.append(circle)})});const legend=document.createElement("div");legend.className="legend";data.series.filter(series=>rows.some(row=>row.series.key===series.key)).forEach(series=>{const item=document.createElement("span");item.innerHTML=`<i class="swatch" style="--c:${series.color}"></i>${series.label}`;legend.append(item)});const article=document.createElement("article");article.append(svg,legend);return article}
function render(){const methodKeys=selected(methodInputs),baselineKeys=selected(baselineInputs),methods=DATA.methods.filter(item=>methodKeys.has(item.key)),baselines=DATA.baselines.filter(item=>baselineKeys.has(item.key));try{localStorage.setItem("physiciq-selected-methods",JSON.stringify([...methodKeys]))}catch(e){}statusEl.textContent=`已选 ${methods.length}/${DATA.methods.length} 个方案`;charts.replaceChildren();let count=0;DATA.metrics.forEach(metric=>{const node=chart(metric,methods,baselines);if(node){charts.append(node);count++}});if(!count){const empty=document.createElement("div");empty.className="empty";empty.textContent="当前选择没有完整指标。";charts.append(empty)}}
const validationPanel=document.getElementById("phyrvgValidation");const validationNode=validationChart(DATA.phyrvg_validation);if(validationNode){document.getElementById("validationChart").append(validationNode);validationPanel.hidden=false}
render();
</script>
</body></html>
'''
    path.write_text(
        template.replace("__PAYLOAD__", payload_text).replace(
            "__STATIC_LINKS__", "".join(static_links)
        ),
        encoding="utf-8",
    )


def load_phyrvg_validation_series(
    root: Path = DEFAULT_PHYRVG_VALIDATION_ROOT,
) -> dict[str, Any] | None:
    """Load the independent 30-case PHYRVG-Full-SA validation curves.

    The 30-case page evaluates deterministic flow MSE, not the 67-case PhysicIQ
    metrics above. Keeping this payload separate lets the restored page expose
    every PHYRVG-Full-SA checkpoint (including smoke and 0613 runs) without
    implying that the two evaluation protocols are interchangeable.
    """
    root = root.expanduser().resolve()
    inventory_path = root / "inventory.json"
    losses_root = root / "losses"
    if not inventory_path.is_file() or not losses_root.is_dir():
        return None
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    entries = inventory.get("entries", [])
    if not isinstance(entries, list):
        return None

    # Match the 30-case validation labels to the canonical 67-case method keys
    # so the two sections use the same naming and color language.
    key_by_validation_method = {
        "full_sa": "full_sa_physrvg_dit",
        "latent_mask": "full_sa_physrvg_latent_mask_loss",
        "object_xssc": "full_sa_physrvg_object_xssc_loss",
        "vjepa": "full_sa_physrvg_vjepa_loss",
        "vjepa_0613": "full_sa_physrvg_vjepa_loss_0613_b2g2",
    }
    label_by_key = {
        "full_sa_physrvg_dit": "PHYRVG-Full-SA",
        "full_sa_physrvg_latent_mask_loss": "PHYRVG-Full-SA + Latent-Mask Loss",
        "full_sa_physrvg_object_xssc_loss": "PHYRVG-Full-SA + Object + XSSC Loss",
        "full_sa_physrvg_vjepa_loss": "PHYRVG-Full-SA + V-JEPA Loss",
        "full_sa_physrvg_vjepa_loss_0613_b2g2": "PHYRVG-Full-SA + V-JEPA Loss · 0613 · b2-gacc2",
    }
    series_by_key: dict[str, dict[str, Any]] = {}
    expected_cases = 30
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("method_label", ""))
        if not label.upper().startswith("PHYRVG-FULL-SA"):
            continue
        validation_method = str(entry.get("method_key", ""))
        key = key_by_validation_method.get(validation_method)
        if key is None:
            # Keep future PHYRVG-Full-SA variants visible even if their method
            # key is new; the label remains a stable fallback identifier.
            key = f"phyrvg_validation_{validation_method or entry.get('entry_id', 'unknown')}"
        loss_path = losses_root / f"{entry.get('entry_id', '')}.json"
        if not loss_path.is_file():
            continue
        try:
            payload = json.loads(loss_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cases = payload.get("cases", [])
        values = [
            float(case["loss_main"])
            for case in cases
            if isinstance(case, dict)
            and isinstance(case.get("loss_main"), (int, float))
            and math.isfinite(float(case["loss_main"]))
        ]
        if not values:
            continue
        series = series_by_key.setdefault(
            key,
            {
                "key": key,
                "label": label_by_key.get(key, label),
                "color": str(entry.get("color", "#0b6f73")),
                "points": [],
            },
        )
        series["points"].append(
            {
                "step": int(entry.get("step", 0)),
                "mean": float(np.mean(values)),
                "count": len(values),
                "complete": len(values) == expected_cases,
                "entry_id": str(entry.get("entry_id", "")),
            }
        )
    if not series_by_key:
        return None
    for series in series_by_key.values():
        series["points"].sort(key=lambda point: (point["step"], point["entry_id"]))
        for point in series["points"]:
            point.pop("entry_id", None)
    ordered_keys = (
        "full_sa_physrvg_dit",
        "full_sa_physrvg_vjepa_loss",
        "full_sa_physrvg_vjepa_loss_0613_b2g2",
        "full_sa_physrvg_latent_mask_loss",
        "full_sa_physrvg_object_xssc_loss",
    )
    ordered_series = [
        series_by_key[key] for key in ordered_keys if key in series_by_key
    ]
    ordered_series.extend(
        series
        for key, series in series_by_key.items()
        if key not in ordered_keys
    )
    return {
        "expected_cases": expected_cases,
        "metric_key": "fixed_pybullet_train_30case_deterministic_flow_mse",
        "metric_title": "Mean deterministic flow MSE",
        "metric_direction": "lower",
        "source_page": "../phyrvg-full-sa-train-validation-30cases-metrics/",
        "series": ordered_series,
    }


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    input_txt = (
        args.input_txt.expanduser().resolve() if args.input_txt is not None else None
    )
    allowlist_path = args.input_json_allowlist.expanduser().resolve()
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif input_txt is not None:
        output_dir = input_txt.parent / "_metric_plots" / input_txt.stem
    else:
        output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_input_jsons = read_allowlist(allowlist_path)
    if len(allowed_input_jsons) != args.expected_cases:
        raise ValueError(
            f"Allowlist has {len(allowed_input_jsons)} unique cases, "
            f"but --expected-cases={args.expected_cases}"
        )

    methods = (
        discover_methods_from_txt(input_txt)
        if input_txt is not None
        else discover_methods(result_root)
    )
    training_checkpoints = (
        discover_training_checkpoints_from_txt(input_txt)
        if input_txt is not None
        else []
    )
    if not methods and not training_checkpoints:
        source = input_txt if input_txt is not None else result_root
        raise RuntimeError(
            f"No ablation methods or training checkpoints found from {source}"
        )

    stats = compute_stats(methods, allowed_input_jsons, args.expected_cases)
    training_stats = compute_training_stats(
        training_checkpoints,
        allowed_input_jsons,
        args.expected_cases,
    )
    baseline_stats = compute_baseline_stats(
        allowed_input_jsons,
        args.expected_cases,
    )
    missing_result_dirs = [
        str(method.result_dir) for method in methods if not method.result_dir.is_dir()
    ]
    missing_training_result_dirs = [
        str(checkpoint.result_dir)
        for checkpoint in training_checkpoints
        if not checkpoint.result_dir.is_dir()
    ]
    csv_path = output_dir / "dit_ablation_metric_stats.csv"
    training_csv_path = output_dir / "xssc_lora_training_step_metric_stats.csv"
    baseline_csv_path = output_dir / "physiciq_baseline_metric_stats.csv"
    manifest_path = output_dir / "dit_ablation_metric_plot_manifest.json"

    write_stats_csv(
        csv_path,
        stats,
        args.expected_cases,
        complete_only=args.complete_only,
    )
    write_training_stats_csv(
        training_csv_path,
        training_stats,
        args.expected_cases,
        complete_only=args.complete_only,
    )
    write_baseline_stats_csv(
        baseline_csv_path,
        baseline_stats,
        args.expected_cases,
    )
    plotted_metrics = (
        tuple(
            metric
            for metric in METRICS
            if any(
                stat.metric.key == metric.key and stat.complete
                for stat in (*stats, *training_stats)
            )
        )
        if args.complete_only
        else METRICS
    )
    if not plotted_metrics:
        raise RuntimeError("No metric has a complete result point yet")
    model_ids = sorted(
        {method.model for method in methods},
        key=lambda model: {"wan_lora": 0, "xssc": 1, "physrvg": 2}.get(model, 99),
    )
    plots: dict[str, dict[str, str]] = {}
    metric_completeness: dict[str, dict[str, dict[str, int]]] = {}
    for model in model_ids:
        png_path = output_dir / f"dit_ablation_{model}_all_metrics.png"
        pdf_path = output_dir / f"dit_ablation_{model}_all_metrics.pdf"
        metric_completeness[model] = plot_metrics(
            png_path,
            pdf_path,
            stats,
            args.expected_cases,
            args.dpi,
            model,
            plotted_metrics,
        )
        plots[model] = {"png": str(png_path), "pdf": str(pdf_path)}

    training_plot: dict[str, str] | None = None
    training_metric_completeness: dict[str, dict[str, int]] = {}
    if training_checkpoints:
        training_png_path = output_dir / "xssc_lora_training_step_all_metrics.png"
        training_pdf_path = output_dir / "xssc_lora_training_step_all_metrics.pdf"
        training_metric_completeness = plot_training_metrics(
            training_png_path,
            training_pdf_path,
            training_stats,
            args.expected_cases,
            args.dpi,
            plotted_metrics,
        )
        training_plot = {
            "png": str(training_png_path),
            "pdf": str(training_pdf_path),
        }

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "result_root": str(result_root),
        "input_txt": None if input_txt is None else str(input_txt),
        "input_json_allowlist": str(allowlist_path),
        "expected_cases": args.expected_cases,
        "num_methods": len(methods),
        "num_training_checkpoints": len(training_checkpoints),
        "missing_result_dirs": missing_result_dirs,
        "missing_training_result_dirs": missing_training_result_dirs,
        "num_metrics": len(plotted_metrics),
        "complete_only": args.complete_only,
        "plotted_metrics": [metric.key for metric in plotted_metrics],
        "stats_csv": str(csv_path),
        "training_stats_csv": str(training_csv_path),
        "baseline_stats_csv": str(baseline_csv_path),
        "baselines": [
            {
                "key": spec["key"],
                "label": spec["label"],
                "result_dir": str(spec["result_dir"].resolve()),
                "expected_cases": int(spec.get("expected_cases", args.expected_cases)),
                "scope_label": spec.get("scope_label", ""),
                "source_page": spec.get("source_page", ""),
            }
            for spec in BASELINE_SPECS
        ],
        "plots": plots,
        "training_plot": training_plot,
        "metric_completeness": metric_completeness,
        "training_metric_completeness": training_metric_completeness,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    phyrvg_validation = load_phyrvg_validation_series()
    write_plot_index(
        output_dir / "index.html",
        plots,
        training_plot,
        training_stats,
        baseline_stats,
        args.expected_cases,
        plotted_metrics,
        phyrvg_validation,
    )

    print(f"Methods: {len(methods)}")
    print(f"Training checkpoints: {len(training_checkpoints)}")
    print(f"Missing result directories: {len(missing_result_dirs)}")
    print(
        "Missing training result directories: "
        f"{len(missing_training_result_dirs)}"
    )
    print(f"Allowed cases: {len(allowed_input_jsons)}")
    for model in model_ids:
        completeness = metric_completeness[model]
        print(f"[{MODEL_LABELS[model]}]")
        for metric in plotted_metrics:
            progress = completeness[metric.key]
            print(
                f"{metric.key}: "
                f"{progress['complete_ablation_points']}/"
                f"{progress['expected_ablation_points']} complete ablation points"
            )
    print(f"Stats CSV: {csv_path}")
    if training_plot is not None:
        print(f"Training stats CSV: {training_csv_path}")
        print(f"Training PNG: {training_plot['png']}")
        print(f"Training PDF: {training_plot['pdf']}")
    print(f"Baseline stats CSV: {baseline_csv_path}")
    for model in model_ids:
        print(f"{MODEL_LABELS[model]} PNG: {plots[model]['png']}")
        print(f"{MODEL_LABELS[model]} PDF: {plots[model]['pdf']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
