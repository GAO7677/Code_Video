from __future__ import annotations

"""Compatibility wrapper for the Wan 2.1-1.3B stage1b context-only inference entry.

This file exists so the 1.3B workflow has a stable, explicit script name under
`train0706_wan1p3b/` while reusing the maintained implementation in
`infer_stage1b_context_only_no_gt_box_v_newtrain0705.py`.
"""

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from code_vjepa_vggt.train0706_wan1p3b.infer_stage1b_context_only_no_gt_box_v_newtrain0705 import (  # noqa: E501
    main as _main,
)


def main() -> None:
    _main()


if __name__ == "__main__":
    main()
