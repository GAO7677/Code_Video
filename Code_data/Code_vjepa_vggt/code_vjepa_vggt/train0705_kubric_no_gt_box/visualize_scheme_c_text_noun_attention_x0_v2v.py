#!/usr/bin/env python3
"""Capture noun cross-attention with the actual Scheme-C entity-binding runtime."""
from __future__ import annotations

import sys

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    visualize_text_noun_attention_x0_v2v as attention,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_v2v,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_scheme_c_entity_caption_physical_v2v as scheme_c,
)


_PHYSIQ_025_NOUNS = {
    "pillows": ("pillows",),
    "table": ("table",),
    "grabber_tools": ("grabber tools", "tools"),
    "tennis_ball": ("tennis ball", "ball"),
    "block": ("block",),
}

attention.NOUN_SPECS.update(
    {
        "physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px": _PHYSIQ_025_NOUNS,
        "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper": {
            "grabber_arm": ("grabber arm", "grabber"),
            "tennis_ball": ("tennis ball", "ball"),
            "cardstock": ("cardstock",),
            "rotating_platform": ("rotating platform", "platform"),
            "table": ("table",),
        },
    }
)


def main() -> None:
    weights_root = scheme_c._option_value(sys.argv, "--weights-root")
    if weights_root is None:
        raise ValueError("--weights-root is required")
    audit = scheme_c.audit_entity_checkpoint(weights_root)
    print(f"[entity-checkpoint-audit] {audit}")
    scheme_c._install_training_matched_defaults(sys.argv)
    attention.base._install_kubric_runtime_hooks = (
        entity_v2v._install_entity_runtime_hooks
    )
    attention.base.main()


if __name__ == "__main__":
    main()
