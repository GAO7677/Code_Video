import torch

from stage1_causal_state_probe import NUM_OBJECTS, NUM_SLOTS, NUM_STATES, SLOT_DIM
from stage1_causal_state_probe.data import (
    TrajectoryDataset,
    gather_object_targets,
    validate_record,
)


def make_record():
    mapping = torch.arange(NUM_SLOTS) - 1
    mapping[mapping >= NUM_OBJECTS] = -1
    return {
        "slots": torch.zeros(NUM_STATES, NUM_SLOTS, SLOT_DIM),
        "slot_attention": torch.zeros(NUM_STATES, NUM_SLOTS, 16, 16),
        "gt_mask": torch.zeros(NUM_STATES, NUM_OBJECTS, 16, 16, dtype=torch.bool),
        "gt_position": torch.zeros(NUM_STATES, NUM_OBJECTS, 3),
        "gt_velocity": torch.zeros(NUM_STATES, NUM_OBJECTS, 3),
        "gt_image_position": torch.zeros(NUM_STATES, NUM_OBJECTS, 2),
        "gt_bbox": torch.zeros(NUM_STATES, NUM_OBJECTS, 4),
        "gt_visibility": torch.ones(NUM_STATES, NUM_OBJECTS, dtype=torch.long),
        "object_valid": torch.ones(NUM_OBJECTS, dtype=torch.bool),
        "slot_valid": torch.ones(NUM_SLOTS, dtype=torch.bool),
        "prefix_slot_to_object": mapping,
        "boundary_slot_to_object": mapping.clone(),
        "source": {"index": 0, "video_name": "test"},
    }


def test_fixed_cache_schema_and_target_gather():
    record = make_record()
    validate_record(record)
    targets, valid = gather_object_targets(record)
    assert targets["position"].shape == (NUM_STATES, NUM_SLOTS, 3)
    assert valid.shape == (NUM_STATES, NUM_SLOTS)
    assert not bool(valid[:, 0].any())


def test_cache_record_round_trip_with_restricted_torch_load(tmp_path):
    split_root = tmp_path / "validation"
    split_root.mkdir()
    record = make_record()
    torch.save(record, split_root / "case_000000.pt")
    (split_root / "records.jsonl").write_text(
        '{"index":0,"record":"case_000000.pt","video_name":"test"}\n'
    )
    loaded = TrajectoryDataset(tmp_path, "validation")[0]
    assert loaded["source"]["video_name"] == "test"
    assert torch.equal(loaded["slots"], record["slots"])
