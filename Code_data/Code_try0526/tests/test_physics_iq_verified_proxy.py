from __future__ import annotations

import sys

import pandas as pd

from physv_eval.single_case.physics_iq_verified_proxy import (
    DEFAULT_OFFICIAL_REPO_ROOT,
    _component_scores,
)


def test_component_formula_matches_official_verified_iqtable() -> None:
    repo_text = str(DEFAULT_OFFICIAL_REPO_ROOT)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from physiq.calculate_iq_score import VIEWS
    from physiq.calculate_iq_score_stable import IQTable

    metrics = {
        "spatiotemporal_iou_v1": [0.2, 0.4],
        "spatial_iou_v1": 0.3,
        "weighted_spatial_iou_v1": 0.4,
        "v1_mse": [0.08, 0.12],
        "variance_spatiotemporal_iou": [0.4, 0.4],
        "variance_spatial": 0.6,
        "variance_weighted_spatial": 0.5,
        "variance_mse": [0.04, 0.06],
    }
    proxy_components = _component_scores(metrics)

    row = {}
    for view in VIEWS:
        row[f"spatiotemporal_iou_v1_{view}"] = metrics["spatiotemporal_iou_v1"]
        row[f"spatial_iou_v1_{view}"] = metrics["spatial_iou_v1"]
        row[f"weighted_spatial_iou_v1_{view}"] = metrics["weighted_spatial_iou_v1"]
        row[f"v1_mse_{view}"] = metrics["v1_mse"]
        row[f"variance_spatiotemporal_iou_{view}"] = metrics[
            "variance_spatiotemporal_iou"
        ]
        row[f"variance_spatial_{view}"] = metrics["variance_spatial"]
        row[f"variance_weighted_spatial_{view}"] = metrics[
            "variance_weighted_spatial"
        ]
        row[f"variance_mse_{view}"] = metrics["variance_mse"]

    official_score = IQTable(pd.DataFrame([row])).get_output_dict()["final_score_view"]
    proxy_score = sum(
        proxy_components[key]
        for key in (
            "score_spatiotemporal_iou",
            "score_spatial_iou",
            "score_weighted_spatial_iou",
            "score_mse",
        )
    ) / 4.0
    assert proxy_score == official_score
