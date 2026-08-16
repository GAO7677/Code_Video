import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from latent_mask_cache import PyBulletLatentMaskCache, mask_case_root


def test_latent_mask_cache_preflight_and_load(tmp_path: Path):
    logical_key = "F2/case001"
    config = {
        "schema_version": 1,
        "status": "complete",
        "num_frames": 49,
        "anchor_frame": 4,
        "native_height": 8,
        "native_width": 12,
    }
    (tmp_path / "cache_config.json").write_text(json.dumps(config), encoding="utf-8")
    case_root = mask_case_root(tmp_path, logical_key)
    case_root.mkdir(parents=True)
    masks = np.ones((1, 49, 8, 12), dtype=np.uint8)
    np.savez_compressed(
        case_root / "object_masks.npz",
        masks_othw=masks,
        frame_indices=np.arange(49),
    )
    metadata = {
        "logical_key": logical_key,
        "object_count": 1,
        "mask_shape": [1, 49, 8, 12],
        "reverse_recall": 1.0,
    }
    (case_root / "entry.json").write_text(json.dumps(metadata), encoding="utf-8")

    cache = PyBulletLatentMaskCache(
        tmp_path,
        num_frames=49,
        anchor_frame=4,
        native_height=8,
        native_width=12,
    )
    cache.validate_records([SimpleNamespace(key=logical_key)])
    loaded = cache.load(logical_key)

    assert np.array_equal(loaded["masks_othw"], masks)
    assert np.array_equal(loaded["frame_indices"], np.arange(49))
