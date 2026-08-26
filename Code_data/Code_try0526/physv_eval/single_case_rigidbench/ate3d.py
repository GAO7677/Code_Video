from __future__ import annotations

import argparse

from rigidbench.eval.score.trajectory import compute_ate3d

from .common import cli_print


def score_case(pred_centroids, gt_trajectories: dict, actors: list[str]) -> dict:
    """Return ATE-3D from reconstructed centroids and GT trajectories.

    ``pred_centroids`` is a list of N arrays shaped (T,3), in world meters.
    ``gt_trajectories`` maps ``<actor>_positions`` to (T,3) world-meter arrays.
    ``actors`` is the ordered active-actor name list; it is not a case ID.
    """
    if len(pred_centroids) != len(actors):
        raise ValueError("pred_centroids and actors must have the same length")
    return compute_ate3d(pred_centroids, gt_trajectories, actors)


if __name__ == "__main__":
    raise SystemExit("Use score_case(pred_centroids, gt_trajectories, actors) from Python; this metric requires reconstructed 3D inputs.")
