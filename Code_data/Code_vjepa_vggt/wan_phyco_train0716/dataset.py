from __future__ import annotations

from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from wan_phyco_train0716.property_maps import (
    build_kubric_property_map,
    build_null_property_map,
    build_pybullet_property_map,
)


class PhysicalPropertyMapDataset(Dataset):
    """Attach compact PhyCo maps to a Scheme-D-compatible source dataset."""

    def __init__(self, dataset: Dataset, *, source: str, map_height: int, map_width: int) -> None:
        self.dataset = dataset
        self.source = str(source)
        self.map_height = int(map_height)
        self.map_width = int(map_width)
        self.load_from_cache = getattr(dataset, "load_from_cache", False)
        self.dataset_stats = dict(getattr(dataset, "dataset_stats", {}))
        self.dataset_stats.update(
            {
                "phyco_source": self.source,
                "property_map_resolution": [self.map_height, self.map_width],
            }
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        metadata = dict(sample.get("metadata", {}))
        if self.source == "pybullet":
            result = build_pybullet_property_map(
                metadata,
                height=self.map_height,
                width=self.map_width,
            )
        elif self.source == "kubric":
            sample_dir = Path(str(metadata["source_video_path"])).parent
            result = build_kubric_property_map(
                sample_dir,
                height=self.map_height,
                width=self.map_width,
            )
        elif self.source == "openvid":
            result = build_null_property_map(
                height=self.map_height,
                width=self.map_width,
                source="openvid",
            )
        else:
            raise ValueError(f"unsupported PhyCo source: {self.source}")
        sample["phyco_control_maps"] = result.maps
        sample["phyco_branch_valid"] = result.branch_valid
        metadata["phyco"] = result.diagnostics
        sample["metadata"] = metadata
        return sample

