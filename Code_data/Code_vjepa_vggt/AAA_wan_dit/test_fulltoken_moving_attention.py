#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from fulltoken_moving_attention import (
    FULL_FEATURE_NAMES,
    OBJECT_FEATURE_NAMES,
    fulltoken_moving_statistics,
)


class FullTokenMovingStatisticsTest(unittest.TestCase):
    def test_uniform_attention_matches_analytic_baseline(self) -> None:
        grid = (3, 3, 4)
        token_count = int(np.prod(grid))
        q = torch.zeros(1, token_count, 2, 4)
        k = torch.zeros_like(q)
        output = fulltoken_moving_statistics(
            q,
            k,
            num_heads=2,
            trajectory_coords=tuple((time, 1, 1) for time in range(grid[0])),
            grid=grid,
            query_chunk=5,
        )
        self.assertTrue(
            np.allclose(output["temporal_matrix"], 1.0 / grid[0], atol=1.0e-6)
        )
        full_index = {name: index for index, name in enumerate(FULL_FEATURE_NAMES)}
        object_index = {
            name: index for index, name in enumerate(OBJECT_FEATURE_NAMES)
        }
        for name in ("local_enrichment", "context_enrichment", "aligned_enrichment"):
            self.assertTrue(
                np.allclose(
                    output["full_features"][:, full_index[name]], 1.0, atol=1.0e-5
                ),
                name,
            )
        for name in (
            "trajectory_enrichment",
            "shift_enrichment",
            "shuffle_enrichment",
            "fixed_position_enrichment",
        ):
            self.assertTrue(
                np.allclose(
                    output["object_features"][:, object_index[name]],
                    1.0,
                    atol=1.0e-5,
                ),
                name,
            )
        self.assertTrue(
            np.allclose(
                output["object_features"][
                    :, object_index["trajectory_selectivity_log2"]
                ],
                0.0,
                atol=1.0e-5,
            )
        )

    def test_partial_trajectory_preserves_full_stats_and_marks_missing_rows(self) -> None:
        grid = (3, 2, 2)
        token_count = int(np.prod(grid))
        q = torch.zeros(1, token_count, 2, 4)
        k = torch.zeros_like(q)
        output = fulltoken_moving_statistics(
            q,
            k,
            num_heads=2,
            trajectory_coords=((0, 0, 0), (2, 1, 1)),
            grid=grid,
            query_chunk=3,
        )
        self.assertEqual(output["trajectory_valid_times"].tolist(), [True, False, True])
        self.assertTrue(np.isfinite(output["full_features"]).all())
        self.assertTrue(np.isnan(output["object_features_by_query_time"][:, 1]).all())
        self.assertTrue(np.isnan(output["trajectory_enrichment"][:, 0, 1]).all())
        object_index = {
            name: index for index, name in enumerate(OBJECT_FEATURE_NAMES)
        }
        self.assertTrue(
            np.allclose(
                output["object_features"][
                    :, object_index["trajectory_selectivity_log2"]
                ],
                0.0,
                atol=1.0e-5,
            )
        )


if __name__ == "__main__":
    unittest.main()
