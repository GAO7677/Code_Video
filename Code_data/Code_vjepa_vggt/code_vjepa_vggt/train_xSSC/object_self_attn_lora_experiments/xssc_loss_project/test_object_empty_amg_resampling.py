from __future__ import annotations

from types import MethodType
import unittest

import torch
import torch.nn as nn

import train_xssc_object_self_attn_lora as core
from xssc_loss_project.train_full_sa_object_slot_dedup_xssc_loss import (
    SlotDedupXSSCFeatureLossWanModule,
)


class ObjectEmptyAMGResamplingTests(unittest.TestCase):
    def test_dict_batch_resamples_empty_amg_sample(self) -> None:
        model = SlotDedupXSSCFeatureLossWanModule.__new__(
            SlotDedupXSSCFeatureLossWanModule
        )
        nn.Module.__init__(model)
        model.enable_object_branch = True
        model.xssc_filter_empty_amg = True
        model.xssc_box_source = "amg"
        model.xssc_empty_amg_max_resample_attempts = 20
        calls: list[str] = []

        def forward_batch(self, samples):
            sample_id = str(samples[0]["id"])
            calls.append(sample_id)
            if sample_id == "empty":
                raise core.EmptyAMGConditionError([0])
            return torch.tensor(7.0)

        def replacement(self):
            return {"id": "replacement"}

        model._forward_sample_batch = MethodType(forward_batch, model)
        model._sample_empty_amg_replacement = MethodType(replacement, model)

        result = model({"id": "empty"})

        self.assertEqual(float(result.item()), 7.0)
        self.assertEqual(calls, ["empty", "replacement"])
        self.assertEqual(model._last_empty_amg_resample_count, 1)


if __name__ == "__main__":
    unittest.main()
