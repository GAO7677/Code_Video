"""Prefix-causal counterpart of the V-JEPA2.1 MOVi-C transfer run."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-slot512-transfer15000.py"
    ),
    name="_xssc_vjepa_noncausal_movic_base",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_slot512_prefix_causal_transfer15000"
)
source_variant_name = "vjepa2_1_vitl16_video_256_ytvis_hq_slot512_prefix_causal"
temporal_mode = "prefix_causal"
transfer_expected_source_variant = source_variant_name
model["encode_backbone"]["temporal_mode"] = temporal_mode

del _base, _Path, _importlib_cfg
