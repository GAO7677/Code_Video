#!/usr/bin/env python3

import numpy as np
import torch

from self_attention_matrix import pool_full_attention_matrix


def test_pool_matches_direct_attention() -> None:
    generator = torch.Generator().manual_seed(7)
    q = torch.randn(1, 12, 2, 4, generator=generator)
    k = torch.randn(1, 12, 2, 4, generator=generator)
    block_mean, key_mass, metadata = pool_full_attention_matrix(
        q,
        k,
        num_heads=2,
        output_bins=4,
        query_chunk=3,
    )
    qh = q[0].permute(1, 0, 2)
    kh = k[0].permute(1, 0, 2)
    attention = torch.softmax(torch.matmul(qh, kh.transpose(-1, -2)) / 2.0, -1)
    expected = attention.reshape(2, 4, 3, 4, 3).mean(dim=(2, 4)).numpy()
    expected_mass = expected * 3
    np.testing.assert_allclose(block_mean, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(key_mass, expected_mass, rtol=1e-5, atol=1e-6)
    assert metadata["query_sampling"] == "none"


if __name__ == "__main__":
    test_pool_matches_direct_attention()
    print("ok")
