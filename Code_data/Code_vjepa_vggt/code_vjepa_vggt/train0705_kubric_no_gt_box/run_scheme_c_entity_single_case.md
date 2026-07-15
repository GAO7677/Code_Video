# Scheme-C single-case inference

This launcher reproduces the formal Scheme-C configuration for one JSON-native
PhysicIQ case. It calls the Scheme-C wrapper directly and does not modify the
inference implementation.

## Default run

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_scheme_c_entity_single_case.sh
```

The default run uses GPU 6, checkpoint `step-003500`, and object residual scale
`1.5`. It reproduces the settings of the selected formal result.

## Input and output paths

| Item | Default path |
|---|---|
| Input JSON | `/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json` |
| Source video resolved by JSON | `/data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center-trimmed/physicIQ_0002_clip_2p5s_3p5s.mp4` |
| Checkpoint | `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_raw49f_scheme_c_entity_caption_physical_fresh_20260714T174707Z/checkpoints/step-003500` |
| Scheme-C wrapper | `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_scheme_c_entity_caption_physical_v2v.py` |
| Output base | `/data/gaoya/agent-data/outputs/scheme_c_entity_single_case` |
| Temporary list/cache | `/data/gaoya/agent-data/cache/t/scheme_c_entity_single_case` |

The default run directory is:

```text
/data/gaoya/agent-data/outputs/scheme_c_entity_single_case/
  step-003500/object_residual_1p5x/
  physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed/
```

It contains `run_config.txt`, `inference.log`, and a `results/` directory. The
`results/` directory contains the generated MP4, per-case JSON/debug artifacts,
context visualization produced by the native pipeline, and the aggregate
`result.json`.

## Explicit parameters

| Environment variable | Default | Meaning |
|---|---:|---|
| `GPU_ID` | `6` | Physical GPU exposed through `CUDA_VISIBLE_DEVICES` |
| `STEP` | `step-003500` | Checkpoint directory name under `CHECKPOINT_ROOT` |
| `CHECKPOINT_ROOT` | Scheme-C fresh run checkpoint root | Parent checkpoint directory |
| `CHECKPOINT` | `${CHECKPOINT_ROOT}/${STEP}` | Full checkpoint override |
| `INPUT_JSON` | selected PhysicIQ case | One JSON-native v2v sample |
| `OUTPUT_ROOT` | `/data/gaoya/agent-data/outputs/scheme_c_entity_single_case` | Output base directory |
| `TMP_ROOT` | `/data/gaoya/agent-data/cache/t/scheme_c_entity_single_case` | Temporary/cache base |
| `HEIGHT` | `512` | Generated frame height |
| `WIDTH` | `896` | Generated frame width |
| `CONTEXT_FRAMES` | `8` | Prefix context frames read from the input video |
| `OUTPUT_FRAMES` | `49` | Total generated video frames |
| `NUM_INFERENCE_STEPS` | `40` | Diffusion sampling steps |
| `CFG_SCALE` | `5.0` | Text classifier-free guidance scale |
| `SEED` | `42` | Sampling seed |
| `FPS` | `30` | Output MP4 frame rate |
| `OBJECT_BRANCH_RESIDUAL_SCALE` | `1.5` | Scale applied to the gated object-branch residual |
| `FORCE` | `1` | Re-run and overwrite an existing result; set to `0` to allow native skip behavior |

Input frames use the native `cover_crop` path to reach `896x512`. Context
sampling is prefix mode. The negative prompt is intentionally not supplied, so
the recorded value is `null`, matching the formal Scheme-C run.

## Scheme-C implicit settings

The Scheme-C wrapper audits the checkpoint and automatically injects the
training-matched entity-caption settings:

```text
grounding_text_prompt=""
grounding_enable_caption_terms=true
grounding_caption_prompt_mode=physical_noun_phrases
grounding_caption_max_phrases=4
grounding_caption_min_score=4.0
compact_object_context_slots=true
object_adapter_mlp_residual_max_ratio=3.0
object_branch_ratio_guard_max_ratio=0.30
object_branch_ratio_guard_max_block_id=-1
```

For the default case, caption grounding resolves `brown tennis ball` and
`orange block`; the expected binding is slot 0 to the ball and slot 1 to the
block. The residual scale multiplies the gated object-branch residual, not the
raw object-context token.

The wrapper also requires Scheme-C entity-binding tensors in the checkpoint and
rejects an untrained all-zero `entity_text_up.weight`. After generation, the
launcher verifies one successful result, exact input/checkpoint identity, a
decodable 49-frame `896x512` video, enabled entity binding, nonempty unique
entity IDs, no unmatched slots, and zero ID collisions.

## Override example

```bash
GPU_ID=7 \
STEP=step-002500 \
OBJECT_BRANCH_RESIDUAL_SCALE=1.0 \
INPUT_JSON=/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/scheme_c_entity_single_case_compare \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_scheme_c_entity_single_case.sh
```

The temporary one-line text file is intentional: the JSON-native v2v entrypoint
accepts a list file, so the launcher adapts a single `INPUT_JSON` to that API.
