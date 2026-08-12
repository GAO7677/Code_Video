"""Transfer the YTVIS clip-2 branch at step 16k to noncausal MOVi-C."""

from pathlib import Path as _Path

from object_centric_bench.learn import CbLinearCosineRestart
from object_centric_bench.util import importlib_cfg as _importlib_cfg


_base = _importlib_cfg(
    _Path(__file__).with_name(
        "rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer20000.py"
    ),
    name="_xssc_vjepa_movic_10f_transfer20000_base_for_clip2_16k",
)
globals().update(_base)

variant_name = (
    "vjepa2_1_vitl16_video_256_movi_c_10f_slot512_"
    "transfer16000_clip2_noncausal"
)
source_variant_name = (
    "vjepa2_1_vitl16_video_ytvis_hq_10f_ar_slot512_transfer10000_bs64"
)

# The source is the latest complete model checkpoint from the YTVIS clip=2.0
# fork. This is deliberately a model-only domain transfer: Adam moments,
# scheduler state, RNG state, and the YTVIS sampler are not carried to MOVi-C.
start_step = 16000
total_step = 50000
max_step = total_step
transfer_expected_source_variant = source_variant_name
transfer_expected_source_step = start_step

# Preserve the clip=2.0 intervention while changing only the training domain.
gradient_clip_norm = 2.0
gclip["max_norm"] = gradient_clip_norm

checkpoint_interval = 1000
checkpoint_keep_steps = list(
    range(start_step + checkpoint_interval, total_step + 1, checkpoint_interval)
)
val_interval = 500

# Restart the learning-rate phase for the new domain. Global step numbering is
# retained for lineage, but the 34k MOVi-C phase gets its own warmup + cosine.
phase_total_steps = total_step - start_step
phase_warmup_steps = int(phase_total_steps * warmup_fraction)
before_step[1] = dict(
    type=CbLinearCosineRestart,
    assigns=["optimiz.param_groups[0]['lr']=value"],
    start_step=start_step,
    nlin=phase_warmup_steps,
    ntotal=phase_total_steps,
    vstart=0,
    vbase=lr,
    vfinal=lr * final_lr_ratio,
)

transfer_description = (
    "YTVIS 10f noncausal step-16000 clip2 model-only transfer to MOVi-C; "
    "MOVi-C bbox initializer and optimizer schedule restart"
)

del _base, _Path, _importlib_cfg
