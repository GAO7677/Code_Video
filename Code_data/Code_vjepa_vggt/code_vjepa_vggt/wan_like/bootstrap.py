from __future__ import annotations

from copy import deepcopy

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()

from wan.configs import WAN_CONFIGS  # type: ignore


def load_wan_config(task: str):
    return deepcopy(WAN_CONFIGS[task])

