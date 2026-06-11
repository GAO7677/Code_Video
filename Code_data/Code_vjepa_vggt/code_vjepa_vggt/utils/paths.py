import sys
import types
from pathlib import Path


PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
WAN_UPSTREAM_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
VJEPA_UPSTREAM_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/vjepa2-main")
VGGT_LOCAL_ROOT = Path("/home/gaoya/Code_Video/DreamWorld-main/extract/VGGT")


def ensure_upstream_paths() -> None:
    for path in (WAN_UPSTREAM_ROOT, VJEPA_UPSTREAM_ROOT, VGGT_LOCAL_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    external_pkg = sys.modules.get("external")
    if external_pkg is None:
        external_pkg = types.ModuleType("external")
        external_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["external"] = external_pkg

    if "external.vggt" not in sys.modules:
        vggt_alias = types.ModuleType("external.vggt")
        vggt_alias.__path__ = [str(VGGT_LOCAL_ROOT / "vggt")]  # type: ignore[attr-defined]
        sys.modules["external.vggt"] = vggt_alias
        setattr(external_pkg, "vggt", vggt_alias)
