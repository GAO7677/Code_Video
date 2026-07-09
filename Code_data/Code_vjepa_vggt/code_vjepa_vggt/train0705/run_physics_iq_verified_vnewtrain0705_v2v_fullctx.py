from __future__ import annotations

"""
Physics-IQ Verified wrapper that routes generation through the full-conditioning
Kubric V2V runner variant.
"""

from code_vjepa_vggt.train0705 import (
    run_physics_iq_verified_vnewtrain0705_v2v as base,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v_fullctx as fullctx_batchmod,
)


def main() -> None:
    base.batchmod = fullctx_batchmod
    base.main()


if __name__ == "__main__":
    main()
