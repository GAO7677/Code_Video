from __future__ import annotations

"""Dedicated no-object wrapper using the isolated object-adapter-safe v2v copy."""

import sys

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric_no_object_branch as kubric_no_object_infer,
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_no_object_safe_v2v as safe_v2v,
)


def _parse_args_force_no_object_branch():
    args = _ORIG_PARSE_ARGS()
    if "--disable-object-branch" not in sys.argv:
        raise SystemExit(
            "This dedicated no-object-branch entry requires an explicit "
            "--disable-object-branch flag."
        )
    args.disable_object_branch = True
    return args


_ORIG_PARSE_ARGS = safe_v2v.parse_args


def main() -> None:
    safe_v2v.kubric_infer = kubric_no_object_infer
    safe_v2v.parse_args = _parse_args_force_no_object_branch
    try:
        safe_v2v.main()
    finally:
        safe_v2v.parse_args = _ORIG_PARSE_ARGS


if __name__ == "__main__":
    main()
