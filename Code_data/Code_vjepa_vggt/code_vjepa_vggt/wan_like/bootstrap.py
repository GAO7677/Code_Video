from __future__ import annotations

import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path

from code_vjepa_vggt.utils.paths import ensure_upstream_paths

ensure_upstream_paths()


WAN_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main/wan")
WAN_MODULES_ROOT = WAN_ROOT / "modules"
WAN_CONFIGS_ROOT = WAN_ROOT / "configs"


def _ensure_fake_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ensure_wan_module_packages() -> None:
    _ensure_fake_package("wan", WAN_ROOT)
    _ensure_fake_package("wan.modules", WAN_MODULES_ROOT)
    _ensure_fake_package("wan.configs", WAN_CONFIGS_ROOT)


def load_wan_config(task: str):
    ensure_wan_module_packages()
    _load_module("wan.configs.shared_config", WAN_CONFIGS_ROOT / "shared_config.py")
    config_module_map = {
        "ti2v-5B": ("wan.configs.wan_ti2v_5B", WAN_CONFIGS_ROOT / "wan_ti2v_5B.py", "ti2v_5B"),
        "t2v-14B": ("wan.configs.wan_t2v_14B", WAN_CONFIGS_ROOT / "wan_t2v_14B.py", "t2v_14B"),
        "i2v-14B": ("wan.configs.wan_i2v_14B", WAN_CONFIGS_ROOT / "wan_i2v_14B.py", "i2v_14B"),
        "ti2v-14B": ("wan.configs.wan_ti2v_14B", WAN_CONFIGS_ROOT / "wan_ti2v_14B.py", "ti2v_14B"),
    }
    if task not in config_module_map:
        raise KeyError(f"unsupported Wan task config: {task}")
    module_name, module_path, attr_name = config_module_map[task]
    module = _load_module(module_name, module_path)
    return deepcopy(getattr(module, attr_name))


def load_wan_t5_encoder():
    ensure_wan_module_packages()
    _load_module("wan.modules.tokenizers", WAN_MODULES_ROOT / "tokenizers.py")
    module = _load_module("wan.modules.t5", WAN_MODULES_ROOT / "t5.py")
    return module.T5EncoderModel


def load_wan_vae():
    ensure_wan_module_packages()
    module = _load_module("wan.modules.vae2_2", WAN_MODULES_ROOT / "vae2_2.py")
    return module.Wan2_2_VAE


def load_wan_model():
    ensure_wan_module_packages()
    module = _load_module("wan.modules.model", WAN_MODULES_ROOT / "model.py")
    return module.WanModel
