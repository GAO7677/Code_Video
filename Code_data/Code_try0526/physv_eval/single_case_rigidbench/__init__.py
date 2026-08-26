"""Single-metric RigidBench-style score functions.

These modules intentionally accept metric inputs, not task/case identifiers.
Directory scanning and result-file updates live in the test70 runner.
"""

METRIC_NAMES = (
    "iou",
    "l2",
    "chamfer",
    "ate",
    "si_mse",
    "lpips",
    "ssim",
    "ate3d",
    "iddrift",
    "bgdrift",
)

__all__ = ["METRIC_NAMES"]
