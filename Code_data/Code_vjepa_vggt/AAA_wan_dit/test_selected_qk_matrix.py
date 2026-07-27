#!/usr/bin/env python3

import unittest

import numpy as np
import torch

from selected_qk_matrix import pool_selected_qk_matrices


class SelectedQKMatrixTest(unittest.TestCase):
    def test_uniform_qk_and_attention(self) -> None:
        q = torch.zeros(1, 20, 3, 4)
        k = torch.zeros_like(q)
        raw, attention, metadata = pool_selected_qk_matrices(
            q,
            k,
            num_heads=3,
            selected_heads=(0, 2),
            output_bins=7,
            query_chunk=6,
        )
        self.assertEqual(raw.shape, (2, 7, 7))
        self.assertEqual(attention.shape, (2, 7, 7))
        self.assertTrue(np.allclose(raw, 0.0))
        self.assertTrue(np.allclose(attention.sum(axis=-1), 1.0, atol=2.0e-3))
        self.assertEqual(metadata["token_count"], 20)


if __name__ == "__main__":
    unittest.main()
