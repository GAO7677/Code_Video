#!/usr/bin/env python3
"""Batch v2v inference for Scheme-E checkpoints."""
from __future__ import annotations

from code_vjepa_vggt.train0715_scheme_d_object_tube_resampler import infer as scheme_d_infer
from code_vjepa_vggt.train0717_scheme_e_object_joint_self_attention import train


def main() -> None:
    # Reuse the JSON-native object construction and entity binding pipeline;
    # only the model factory/checkpoint contract changes in Scheme-E.
    scheme_d_infer.train = train
    scheme_d_infer.main()


if __name__ == "__main__":
    main()

