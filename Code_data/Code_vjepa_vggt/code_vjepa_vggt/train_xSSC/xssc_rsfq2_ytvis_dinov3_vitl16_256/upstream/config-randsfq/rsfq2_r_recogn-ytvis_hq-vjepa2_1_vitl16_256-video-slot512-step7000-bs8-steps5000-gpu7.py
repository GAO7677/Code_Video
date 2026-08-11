"""Paper-protocol downstream probe for the noncausal V-JEPA xSSC step-7000."""

from pathlib import Path as _Path

from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_r_recogn-ytvis_hq-vjepa2_1_vitl16_256-video-"
        "slot512-step7000-pilot.py"
    ),
    name="_vjepa_step7000_downstream_pilot",
)
globals().update(_base)

# Runtime placement and experiment identity.
gpu_ids = [7]
expected_world_size = 1
data_dir = _Path("/data/gaoya/dataset")
save_dir = _Path(
    "/data/gaoya/agent-data/checkpoints/"
    "xssc_vjepa2_1_downstream_recognition_step7000_bs8_steps5000_gpu7"
)
variant_name = "vjepa2_1_vitl16_video_slot512_step7000_recognition_bs8_steps5000"
wandb_project = "xssc_vjepa2_1_downstream"
wandb_mode = "online"

# Match the official xSSC downstream optimization budget. Increasing workers
# changes only input throughput.
batch_size_t = 8
batch_size_v = 1
total_step = 5000
max_step = total_step
val_interval = 125
checkpoint_interval = 1000
checkpoint_keep_steps = list(
    range(checkpoint_interval, total_step + 1, checkpoint_interval)
)
num_work = 16

del _base, _importlib_cfg, _Path
