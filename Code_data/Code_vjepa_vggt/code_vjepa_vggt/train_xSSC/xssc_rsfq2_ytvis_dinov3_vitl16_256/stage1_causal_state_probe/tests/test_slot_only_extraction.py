import torch
import torch.nn as nn

from object_centric_bench.model.randsfq2_vjepa_video import RandSFQ2VJEPAVideo


class Backbone(nn.Module):
    def forward(self, value):
        batch, raw_time = value.shape[:2]
        return torch.ones(batch, raw_time // 2, 4, 2, 2)


class Initializer(nn.Module):
    def forward(self, value):
        if isinstance(value, int):
            return torch.zeros(value, 3, 4)
        return value


class Aggregator(nn.Module):
    def forward(self, encoded, query, num_iter=None):
        attention = torch.ones(query.shape[0], query.shape[1], encoded.shape[1])
        return query + encoded.mean(dim=1, keepdim=True), attention


class Transit(nn.Module):
    def forward(self, slots, encoded):
        return slots[:, -1]


class ForbiddenDecoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("slot-only extraction called the decoder")


def build_model():
    model = RandSFQ2VJEPAVideo.__new__(RandSFQ2VJEPAVideo)
    nn.Module.__init__(model)
    model.encode_backbone = Backbone()
    model.encode_posit_embed = nn.Identity()
    model.encode_project = nn.Identity()
    model.initializ = Initializer()
    model.aggregat = Aggregator()
    model.transit = Transit()
    model.decode = ForbiddenDecoder()
    return model


def test_slot_only_path_has_expected_shapes_and_skips_decoder():
    model = build_model()
    video = torch.randn(2, 24, 3, 32, 32)
    condition = torch.randn(2, 3, 4)
    feature, slots, attention = model.extract_slot_trajectory(video, condition)
    assert feature.shape == (2, 12, 4, 2, 2)
    assert slots.shape == (2, 12, 3, 4)
    assert attention.shape == (2, 12, 3, 2, 2)


def test_slot_only_path_rejects_time_varying_conditions():
    model = build_model()
    video = torch.randn(2, 24, 3, 32, 32)
    condition = torch.randn(2, 12, 3, 4)
    try:
        model.extract_slot_trajectory(video, condition)
    except ValueError as error:
        assert "[B,S,C]" in str(error)
    else:
        raise AssertionError("time-varying future conditions were accepted")

