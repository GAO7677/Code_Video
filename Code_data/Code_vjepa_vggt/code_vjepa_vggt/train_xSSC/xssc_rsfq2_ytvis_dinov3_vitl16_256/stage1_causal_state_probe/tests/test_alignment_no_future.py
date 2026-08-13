import torch

from stage1_causal_state_probe.alignment import calibrate_identity


def make_attention_and_masks():
    masks = torch.zeros(12, 2, 4, 4, dtype=torch.bool)
    masks[:, 0, :, :2] = True
    masks[:, 1, :, 2:] = True
    attention = masks.float().clone()
    return attention, masks


def test_prefix_mapping_is_invariant_to_future_masks():
    attention, masks = make_attention_and_masks()
    valid = torch.tensor([True, True])
    reference = calibrate_identity(
        attention, masks, valid, calibration_states=4, mode="prefix_oracle"
    )
    perturbed = masks.clone()
    perturbed[4:] = perturbed[4:].flip(1)
    candidate = calibrate_identity(
        attention, perturbed, valid, calibration_states=4, mode="prefix_oracle"
    )
    assert torch.equal(reference.slot_to_object, candidate.slot_to_object)


def test_boundary_mapping_uses_only_last_observed_state():
    attention, masks = make_attention_and_masks()
    valid = torch.tensor([True, True])
    reference = calibrate_identity(
        attention, masks, valid, calibration_states=4, mode="boundary_frozen"
    )
    masks[:3] = masks[:3].flip(1)
    candidate = calibrate_identity(
        attention, masks, valid, calibration_states=4, mode="boundary_frozen"
    )
    assert torch.equal(reference.slot_to_object, candidate.slot_to_object)

