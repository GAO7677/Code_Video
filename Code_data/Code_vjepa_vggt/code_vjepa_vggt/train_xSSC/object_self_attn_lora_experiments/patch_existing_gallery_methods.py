#!/usr/bin/env python3
"""Add the Utonia Scene Enabled method column to existing gallery pages.

The pages already contain the Scene Enabled records; this lightweight repair
only updates the embedded method registry and leaves all video/metric files
untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
import re


HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")
PAGES = (HUB_ROOT / "test5" / "index.html", HUB_ROOT / "physiciq" / "index.html")
NO_SCENE_KEY = "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_b2gacc2"
ENABLED_METHOD = {
    "key": "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled",
    "label": (
        "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
        "Scene Enabled · formal · b2/b4"
    ),
    "color": "#F28E2B",
    "displayGroup": "phyrvg",
    "schemeKey": "full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled",
    "schemeLabel": (
        "PHYRVG-Full-SA + V-JEPA Loss · Utonia Scene Weights · "
        "Scene Enabled · formal · b2/b4"
    ),
}


def patch_page(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});\n    const caseSelect", text, re.DOTALL)
    if match is None:
        raise RuntimeError(f"embedded dashboard data not found: {path}")
    data = json.loads(match.group(1))
    methods = list(data.get("methods", []))
    enabled_key = ENABLED_METHOD["key"]
    if any(str(method.get("key")) == enabled_key for method in methods):
        return False
    insert_at = next(
        (index + 1 for index, method in enumerate(methods)
         if str(method.get("key")) == NO_SCENE_KEY),
        len(methods),
    )
    methods.insert(insert_at, ENABLED_METHOD)
    data["methods"] = methods
    replacement = "const D=" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n    const caseSelect"
    path.write_text(text[: match.start()] + replacement + text[match.end() :], encoding="utf-8")
    return True


def main() -> None:
    changed = [str(path) for path in PAGES if patch_page(path)]
    print(json.dumps({"updated_pages": changed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
