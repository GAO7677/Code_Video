#!/usr/bin/env python3
"""Verify the official-dimension DINOv3 xSSC configuration and model shapes."""

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "third_party/dinov3"))
sys.path.insert(0, str(ROOT / "upstream"))

from object_centric_bench.model import ModelWrap  # noqa: E402
from object_centric_bench.util import Config, build_from_config  # noqa: E402


CONFIG = (
    ROOT
    / "upstream/config-randsfq/"
    "rsfq2_c-movi_c-dinov3_vits16_256-official_dims.py"
)
OFFICIAL_CHECKPOINT = Path(
    "/data/gaoya/agent-data/weights/xssc_official_archive_rsfq2/"
    "rsfq2_c-movi_c/42-0035.pth"
)


def main() -> None:
    cfg = Config.fromfile(CONFIG)
    assert cfg.emb_dim == 256
    assert cfg.vfm_dim == 384
    assert cfg.decoder_dynamic_ratio == 0.25
    assert cfg.transition_num_heads == 4
    assert cfg.decoder_num_heads == 4
    assert cfg.resolut1 == [16, 16]

    model = ModelWrap(
        build_from_config(cfg.model),
        cfg.model_imap,
        cfg.model_omap,
    )
    backbone = model.m.encode_backbone[0]
    assert backbone.embed_dim == 384
    assert backbone.patch_size == 16
    assert backbone.out_size == 16
    assert backbone.load_report["all_source_tensors_consumed"]

    model.eval()
    video = torch.zeros(1, 2, 3, 256, 256)
    bbox = torch.zeros(1, 2, cfg.max_num, 4)
    with torch.inference_mode():
        output = model.m(video, bbox)
    feature, slotz, attenta, recon, attentd = output

    expected = {
        "feature": (1, 2, 384, 16, 16),
        "slotz": (1, 2, 11, 256),
        "attenta": (1, 2, 11, 16, 16),
        "recon": (1, 2, 384, 16, 16),
        "attentd": (1, 2, 11, 16, 16),
    }
    actual = {
        "feature": tuple(feature.shape),
        "slotz": tuple(slotz.shape),
        "attenta": tuple(attenta.shape),
        "recon": tuple(recon.shape),
        "attentd": tuple(attentd.shape),
    }
    assert actual == expected, (actual, expected)

    with torch.inference_mode():
        decoder_dim = model.m.decode.project2(slotz[:, 0]).shape[-1]
    static_dim = int(decoder_dim * (1 - cfg.decoder_dynamic_ratio))
    dynamic_dim = decoder_dim - static_dim
    assert (decoder_dim, static_dim, dynamic_dim) == (384, 288, 96)

    trainable = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if not name.startswith("m.encode_backbone.")
    )
    official = torch.load(
        OFFICIAL_CHECKPOINT,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    current_non_backbone = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("m.encode_backbone.")
    }
    official_non_backbone = {
        key: value
        for key, value in official.items()
        if not key.startswith("m.encode_backbone.")
    }
    assert current_non_backbone.keys() == official_non_backbone.keys()
    assert all(
        current_non_backbone[key].shape == official_non_backbone[key].shape
        for key in current_non_backbone
    )
    official_non_backbone_parameters = sum(
        value.numel() for value in official_non_backbone.values()
    )
    assert trainable == official_non_backbone_parameters

    print(
        {
            "config": str(CONFIG),
            "backbone": "DINOv3-S/16 LVD1689M",
            "shapes": actual,
            "decoder_split": [static_dim, dynamic_dim],
            "non_backbone_parameter_count": trainable,
            "official_non_backbone_state_keys": len(official_non_backbone),
            "official_non_backbone_shapes_exact": True,
            "weight_load_report": backbone.load_report,
        }
    )


if __name__ == "__main__":
    main()
