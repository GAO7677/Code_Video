"""Train Wan self-attention LoRA with a Frozen Motion Probe objective.

Unlike the older Scheme-B design, the second Student pass is not trainable.
Both GT x0 and reconstructed Student x0 are corrupted with exactly the same
fixed-level noise and passed through one separately loaded, parameter-frozen,
LoRA-free Wan2.2-TI2V-5B baseline DiT.  Its latest3350 PCK Top100 Q/K maps are
the measurement instrument, not an optimization target that can move itself.

The main Student also starts from the official Wan2.2-TI2V baseline.  No
historical OpenVid/preset LoRA is accepted; only the new zero-initialized
self-attention adapter created by the existing experiment core is trainable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

EXPERIMENT_ROOT = Path(__file__).resolve().parent.parent
REPOSITORY_ROOT = EXPERIMENT_ROOT.parents[2]
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
for _path in (EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, REPOSITORY_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch
import torch.nn as nn
from einops import rearrange
from torch.utils.checkpoint import checkpoint

from diffsynth.core import ModelConfig

import code_vjepa_vggt.context_wan_v_newtrain as context_wan
import train_xssc_object_self_attn_lora as core
from attention_trajectory_distillation_project.frozen_motion_probe import (
    TopHeadQKCollector,
    aggregate_head_probabilities,
    assert_no_lora_modules,
    blend_with_fixed_probe_noise,
    capture_wan_self_attention_qk,
    load_pck_head_weights,
    pck_weighted_teacher_student_head_kl,
    query_rows_from_mask,
    query_rows_from_points,
    student_teacher_heatmap_kl,
    teacher_student_head_kl,
    trajectory_huber_loss,
)


DEFAULT_WAN22_BASELINE_ROOT = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
DEFAULT_TOP100_CONFIG = str(
    EXPERIMENT_ROOT / "configs/physiciq67_pck32_s039_latest3350_top100_heads.json"
)
DEFAULT_TOP100_SUBSET = "T_physiciq67_pck32_s039_latest3350_top100"
DEFAULT_TOP100_SUBTYPE = "physiciq67_pck32_s039_latest3350"


def _baseline_dit_shards(root: str | Path) -> tuple[str, ...]:
    root = Path(root).expanduser().resolve()
    expected = tuple(
        root / f"diffusion_pytorch_model-{index:05d}-of-00003.safetensors"
        for index in range(1, 4)
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Wan2.2 baseline DiT shards are missing: {missing}")
    return tuple(map(str, expected))


def _probe_grid(dit: nn.Module, latents: torch.Tensor) -> tuple[int, int, int]:
    patch_size = tuple(map(int, dit.patch_size))
    latent_shape = tuple(map(int, latents.shape[-3:]))
    if len(patch_size) != 3 or any(size % patch for size, patch in zip(latent_shape, patch_size)):
        raise RuntimeError(
            f"latent shape {latent_shape} is incompatible with Wan patch_size={patch_size}"
        )
    return tuple(size // patch for size, patch in zip(latent_shape, patch_size))


def _unwrap_query_payload(raw_sample: dict, key: str):
    if key in raw_sample:
        return raw_sample[key]
    metadata = raw_sample.get("metadata")
    if isinstance(metadata, dict) and key in metadata:
        return metadata[key]
    return None


def _infer_image_size(raw_sample: dict) -> tuple[int, int] | None:
    explicit = _unwrap_query_payload(raw_sample, "object_query_image_size")
    if explicit is not None:
        values = torch.as_tensor(explicit).flatten()
        if values.numel() != 2:
            raise ValueError("object_query_image_size must contain [height,width]")
        return int(values[0]), int(values[1])
    video = raw_sample.get("video")
    if isinstance(video, torch.Tensor) and video.ndim >= 4:
        return int(video.shape[-2]), int(video.shape[-1])
    return None


def resolve_gt_fixed_query_rows(
    raw_sample: dict,
    *,
    grid: tuple[int, int, int],
    token_key: str,
    mask_key: str,
    points_key: str,
    query_latent_frame: int,
    object_index: int,
) -> tuple[torch.Tensor, str]:
    """Resolve one GT-defined row set; never inspect Student predictions."""
    if not isinstance(raw_sample, dict):
        raise TypeError(
            "Frozen Motion Probe requires raw_sample metadata with GT query tokens, "
            "an object mask, or tracking points"
        )
    exact = _unwrap_query_payload(raw_sample, token_key)
    if exact is not None:
        rows = torch.as_tensor(exact, dtype=torch.long).flatten()
        if rows.numel() == 0:
            raise ValueError(f"{token_key} is empty")
        sequence = int(grid[0] * grid[1] * grid[2])
        if int(rows.min()) < 0 or int(rows.max()) >= sequence:
            raise IndexError(
                f"{token_key} rows [{int(rows.min())},{int(rows.max())}] "
                f"outside probe sequence length {sequence}"
            )
        spatial = int(grid[1] * grid[2])
        frames = torch.div(rows, spatial, rounding_mode="floor")
        if not bool((frames == int(query_latent_frame)).all()):
            raise ValueError(
                f"{token_key} must contain only fixed latent-frame "
                f"{query_latent_frame} rows, got frames={torch.unique(frames).tolist()}"
            )
        return torch.unique(rows, sorted=True), "precomputed_token_indices"

    mask = _unwrap_query_payload(raw_sample, mask_key)
    if mask is not None:
        return (
            query_rows_from_mask(
                torch.as_tensor(mask),
                grid=grid,
                query_latent_frame=query_latent_frame,
                object_index=object_index,
            ),
            "gt_object_mask",
        )

    points = _unwrap_query_payload(raw_sample, points_key)
    if points is not None:
        return (
            query_rows_from_points(
                torch.as_tensor(points),
                grid=grid,
                query_latent_frame=query_latent_frame,
                image_size=_infer_image_size(raw_sample),
                object_index=object_index,
            ),
            "gt_tracking_points",
        )

    raise KeyError(
        "Frozen Motion Probe found no GT query source. Add one of "
        f"{token_key!r}, {mask_key!r}, or {points_key!r} to each raw sample. "
        "There is intentionally no Student-derived fallback."
    )


class FrozenMotionProbeWanModule(core.DINOv3XSSCContextSlotsWanModule):
    """LoRA Student plus a shared, unregistered, frozen baseline Wan probe."""

    def _initialize_frozen_dit(self, dit: nn.Module) -> list[str]:
        # Called by the existing core before it injects the new trainable adapter.
        assert_no_lora_modules(dit, label="main Wan2.2 Student baseline")
        self.base_dit_initialization = "official Wan2.2-TI2V-5B baseline (no loaded LoRA)"
        return []

    def __init__(
        self,
        *args,
        motion_probe_wan_root: str,
        motion_probe_head_config: str,
        motion_probe_head_subset_id: str,
        motion_probe_head_feature_subtype: str,
        motion_probe_timestep: float,
        motion_probe_noise_level: float,
        motion_probe_heatmap_weight: float,
        motion_probe_trajectory_weight: float,
        motion_probe_trajectory_huber_delta: float,
        motion_probe_query_latent_frame: int,
        motion_probe_query_object_index: int,
        motion_probe_query_token_key: str,
        motion_probe_query_mask_key: str,
        motion_probe_query_points_key: str,
        motion_probe_expected_latent_frames: int,
        motion_probe_gradient_checkpointing_offload: bool,
        motion_probe_gradient_diagnostics_every_n_forwards: int,
        motion_probe_device: str | torch.device,
        **kwargs,
    ) -> None:
        kwargs["enable_object_branch"] = False
        super().__init__(*args, **kwargs)
        if self.enable_object_branch:
            raise RuntimeError("Frozen Motion Probe entry requires --disable_object_branch")
        if self.self_attn_adaptation_mode not in ("full_sa", "t_head"):
            raise ValueError(
                "Frozen Motion Probe currently supports full_sa or t_head Student adapters"
            )

        self.motion_probe_timestep = float(motion_probe_timestep)
        self.motion_probe_noise_level = float(motion_probe_noise_level)
        self.motion_probe_heatmap_weight = float(motion_probe_heatmap_weight)
        self.motion_probe_trajectory_weight = float(motion_probe_trajectory_weight)
        self.motion_probe_trajectory_huber_delta = float(
            motion_probe_trajectory_huber_delta
        )
        self.motion_probe_query_latent_frame = int(motion_probe_query_latent_frame)
        self.motion_probe_query_object_index = int(motion_probe_query_object_index)
        self.motion_probe_query_token_key = str(motion_probe_query_token_key)
        self.motion_probe_query_mask_key = str(motion_probe_query_mask_key)
        self.motion_probe_query_points_key = str(motion_probe_query_points_key)
        self.motion_probe_expected_latent_frames = int(
            motion_probe_expected_latent_frames
        )
        self.motion_probe_gradient_checkpointing_offload = bool(
            motion_probe_gradient_checkpointing_offload
        )
        self.motion_probe_gradient_diagnostics_every_n_forwards = int(
            motion_probe_gradient_diagnostics_every_n_forwards
        )
        self._motion_probe_forward_count = 0

        if not 0.0 <= self.motion_probe_timestep <= 1000.0:
            raise ValueError("motion_probe_timestep must be in [0,1000]")
        if not 0.0 <= self.motion_probe_noise_level <= 1.0:
            raise ValueError("motion_probe_noise_level must be in [0,1]")
        if self.motion_probe_heatmap_weight < 0.0:
            raise ValueError("motion_probe_heatmap_weight must be non-negative")
        if self.motion_probe_trajectory_weight < 0.0:
            raise ValueError("motion_probe_trajectory_weight must be non-negative")
        if self.motion_probe_heatmap_weight + self.motion_probe_trajectory_weight <= 0.0:
            raise ValueError("at least one Frozen Motion Probe loss weight must be positive")
        if self.motion_probe_trajectory_huber_delta <= 0.0:
            raise ValueError("motion_probe_trajectory_huber_delta must be positive")
        if self.motion_probe_expected_latent_frames <= 0:
            raise ValueError("motion_probe_expected_latent_frames must be positive")
        if self.motion_probe_gradient_diagnostics_every_n_forwards <= 0:
            raise ValueError(
                "motion_probe_gradient_diagnostics_every_n_forwards must be positive"
            )

        (
            probe_heads_by_block,
            probe_head_metadata,
        ) = core.load_head_selection_config(
            motion_probe_head_config,
            expected_subset_id=motion_probe_head_subset_id,
            expected_role="T",
            expected_feature_subtype=motion_probe_head_feature_subtype,
            expected_num_heads=100,
            num_blocks=30,
            num_heads=24,
        )
        self.motion_probe_selected_heads_by_block = probe_heads_by_block
        self.motion_probe_head_metadata = probe_head_metadata
        pck_weights, pck_audit = load_pck_head_weights(
            probe_head_metadata,
            probe_heads_by_block,
        )
        self.register_buffer(
            "motion_probe_pck_weights",
            pck_weights,
            persistent=True,
        )
        self.register_buffer(
            "motion_probe_pck_head_identity",
            torch.tensor(pck_audit["head_pairs"], dtype=torch.int32),
            persistent=True,
        )
        self.motion_probe_pck_audit = pck_audit

        probe_device = torch.device(motion_probe_device)
        probe_pipe = context_wan.ContextAwareWanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=probe_device,
            model_configs=[ModelConfig(path=list(_baseline_dit_shards(motion_probe_wan_root)))],
            tokenizer_config=None,
            audio_processor_config=None,
            redirect_common_files=False,
        )
        probe_dit = probe_pipe.dit
        if probe_dit is None:
            raise RuntimeError("failed to load Wan2.2 baseline DiT for Frozen Motion Probe")
        assert_no_lora_modules(probe_dit, label="Frozen Motion Probe Wan2.2 baseline")
        probe_dit.requires_grad_(False)
        probe_dit.eval()
        if any(parameter.requires_grad for parameter in probe_dit.parameters()):
            raise RuntimeError("Frozen Motion Probe contains trainable parameters")
        # Bypass nn.Module registration: the 5B probe must not enter the optimizer,
        # DDP state, or Student checkpoints.  It still participates in autograd
        # with respect to its tensor input.
        object.__setattr__(self, "_motion_probe_dit", probe_dit)
        probe_pipe.dit = None
        del probe_pipe

    def train(self, mode: bool = True):
        super().train(mode)
        probe = getattr(self, "_motion_probe_dit", None)
        if probe is not None:
            probe.eval()
        return self

    def forward(self, data, inputs=None):
        # The parent bypasses _compute_object_losses when the xSSC object branch
        # is disabled.  This entry intentionally disables that branch but still
        # needs the preserved flow loss plus Frozen Motion Probe auxiliaries.
        if isinstance(data, list):
            return super().forward(data, inputs=inputs)
        if inputs is None:
            inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(
            inputs,
            self.pipe.device,
            self.pipe.torch_dtype,
        )
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss, metrics = self._compute_object_losses(
            self.pipe,
            inputs[0],
            inputs[1],
        )
        self.last_train_metrics = metrics
        return loss

    @staticmethod
    def _restore_condition_latents(
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        captured_inputs: dict,
    ) -> torch.Tensor:
        context_indices = context_wan.resolve_context_latent_indices_from_frames(
            raw_frame_indices=captured_inputs.get("context_frame_indices"),
            raw_num_frames=captured_inputs.get("num_frames"),
            latent_length=int(target_x0.shape[2]),
        )
        if context_indices:
            return context_wan.apply_clean_latents_at_indices(
                pred_x0,
                target_x0,
                context_indices,
            )
        prefix_length = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=captured_inputs.get("clean_prefix_latents"),
            num_clean_prefix_latents=captured_inputs.get("num_clean_prefix_latents"),
        )
        if prefix_length > 0:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, :prefix_length] = target_x0[:, :, :prefix_length]
        elif "first_frame_latents" in captured_inputs:
            pred_x0 = pred_x0.clone()
            pred_x0[:, :, :1] = target_x0[:, :, :1]
        return pred_x0

    def _run_frozen_probe(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        captured_inputs: dict,
        query_rows: torch.Tensor,
        grid: tuple[int, int, int],
        retain_input_gradient: bool,
        fixed_query_by_block: dict[int, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        probe_dit = self._motion_probe_dit
        probe_dit.eval()
        if any(parameter.requires_grad for parameter in probe_dit.parameters()):
            raise RuntimeError("Frozen Motion Probe parameter unexpectedly became trainable")
        selected_blocks = set(self.motion_probe_selected_heads_by_block)
        if fixed_query_by_block is not None:
            fixed_blocks = set(fixed_query_by_block)
            if fixed_blocks != selected_blocks:
                raise ValueError(
                    "fixed GT-Q block mismatch: "
                    f"got={sorted(fixed_blocks)}, expected={sorted(selected_blocks)}"
                )
        # This is the standard Wan2.2 TI2V baseline path up to the final
        # transformer block.  Q/K maps are explicit checkpoint outputs rather
        # than Python side effects, so non-reentrant recomputation is exact.
        clean_prefix_latents = captured_inputs.get("clean_prefix_latents")
        clean_prefix_len = context_wan.resolve_num_clean_prefix_latents(
            clean_prefix_latents=clean_prefix_latents,
            num_clean_prefix_latents=captured_inputs.get(
                "num_clean_prefix_latents"
            ),
        )
        if clean_prefix_len > 0:
            latents = context_wan.apply_clean_prefix_to_latents(
                latents,
                clean_prefix_latents,
            )

        if captured_inputs.get("reference_latents") is not None:
            raise ValueError(
                "Frozen Motion Probe does not support reference_latents because "
                "they change the fixed 13-frame query-token indexing"
            )
        fuse_vae = bool(captured_inputs.get("fuse_vae_embedding_in_latents", False))
        if probe_dit.seperated_timestep and (fuse_vae or clean_prefix_len > 0):
            clean_steps = clean_prefix_len if clean_prefix_len > 0 else 1
            tokens_per_latent = int(grid[1] * grid[2])
            token_timestep = torch.cat(
                [
                    torch.zeros(
                        (clean_steps * tokens_per_latent,),
                        dtype=latents.dtype,
                        device=latents.device,
                    ),
                    torch.ones(
                        ((latents.shape[2] - clean_steps) * tokens_per_latent,),
                        dtype=latents.dtype,
                        device=latents.device,
                    )
                    * timestep.flatten()[0],
                ]
            )
            time_embedding = probe_dit.time_embedding(
                context_wan.sinusoidal_embedding_1d(
                    probe_dit.freq_dim,
                    token_timestep,
                ).unsqueeze(0)
            )
            t_mod = probe_dit.time_projection(time_embedding).unflatten(
                2,
                (6, probe_dit.dim),
            )
        else:
            time_embedding = probe_dit.time_embedding(
                context_wan.sinusoidal_embedding_1d(
                    probe_dit.freq_dim,
                    timestep,
                )
            )
            t_mod = probe_dit.time_projection(time_embedding).unflatten(
                1,
                (6, probe_dit.dim),
            )

        text_context = captured_inputs.get("context")
        if not isinstance(text_context, torch.Tensor):
            raise RuntimeError("Frozen Motion Probe requires encoded text context")
        text_context = probe_dit.text_embedding(text_context)
        x = latents
        if x.shape[0] != text_context.shape[0]:
            x = torch.cat([x] * text_context.shape[0], dim=0)
        y = captured_inputs.get("y")
        if y is not None and probe_dit.require_vae_embedding:
            x = torch.cat([x, y], dim=1)
        clip_feature = captured_inputs.get("clip_feature")
        if clip_feature is not None and probe_dit.require_clip_embedding:
            clip_embedding = probe_dit.img_emb(clip_feature)
            text_context = torch.cat([clip_embedding, text_context], dim=1)
        x = probe_dit.patchify(
            x,
            captured_inputs.get("control_camera_latents_input"),
        )
        actual_grid = tuple(map(int, x.shape[2:]))
        if actual_grid != tuple(grid):
            raise RuntimeError(
                f"probe patch grid changed from expected {grid} to {actual_grid}"
            )
        frames, height, width = actual_grid
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        freqs = torch.cat(
            [
                probe_dit.freqs[0][:frames]
                .view(frames, 1, 1, -1)
                .expand(frames, height, width, -1),
                probe_dit.freqs[1][:height]
                .view(1, height, 1, -1)
                .expand(frames, height, width, -1),
                probe_dit.freqs[2][:width]
                .view(1, 1, width, -1)
                .expand(frames, height, width, -1),
            ],
            dim=-1,
        ).reshape(frames * height * width, 1, -1).to(x.device)

        head_probabilities = []
        query_representations_by_block: dict[int, torch.Tensor] = {}
        for block_id, block in enumerate(probe_dit.blocks):
            selected_heads = self.motion_probe_selected_heads_by_block.get(
                block_id,
                (),
            )
            if selected_heads:

                def selected_block_forward(
                    block_x,
                    block_context,
                    block_t_mod,
                    block_freqs,
                    *,
                    _block=block,
                    _block_id=block_id,
                    _selected_heads=selected_heads,
                ):
                    collector = TopHeadQKCollector(
                        selected_heads_by_block={_block_id: _selected_heads},
                        query_rows=query_rows,
                        grid=grid,
                        expected_num_heads=24,
                        fixed_query_by_block=(
                            None
                            if fixed_query_by_block is None
                            else {
                                _block_id: fixed_query_by_block[_block_id],
                            }
                        ),
                    )
                    with capture_wan_self_attention_qk(probe_dit, collector):
                        block_output = _block(
                            block_x,
                            block_context,
                            block_t_mod,
                            block_freqs,
                        )
                    return (
                        block_output,
                        collector.finalize_head_probabilities(),
                        collector.finalize_query_representations(),
                    )

                if retain_input_gradient:
                    if self.motion_probe_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x, block_probabilities, block_query = checkpoint(
                                selected_block_forward,
                                x,
                                text_context,
                                t_mod,
                                freqs,
                                use_reentrant=False,
                            )
                    else:
                        x, block_probabilities, block_query = checkpoint(
                            selected_block_forward,
                            x,
                            text_context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x, block_probabilities, block_query = selected_block_forward(
                        x,
                        text_context,
                        t_mod,
                        freqs,
                    )
                head_probabilities.append(block_probabilities)
                query_representations_by_block[block_id] = block_query
            elif retain_input_gradient:
                if self.motion_probe_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = checkpoint(
                            block,
                            x,
                            text_context,
                            t_mod,
                            freqs,
                            use_reentrant=False,
                        )
                else:
                    x = checkpoint(
                        block,
                        x,
                        text_context,
                        t_mod,
                        freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, text_context, t_mod, freqs)

        if not head_probabilities:
            raise RuntimeError("Frozen Motion Probe captured no latest3350 heads")
        captured_blocks = set(query_representations_by_block)
        if captured_blocks != selected_blocks:
            raise RuntimeError(
                "Frozen Motion Probe query capture block mismatch: "
                f"captured={sorted(captured_blocks)}, expected={sorted(selected_blocks)}"
            )
        physical_heads = torch.cat(head_probabilities, dim=1)
        if physical_heads.shape[1] != 100:
            raise RuntimeError(
                f"Frozen Motion Probe captured {physical_heads.shape[1]} heads, expected 100"
            )
        heatmap = aggregate_head_probabilities(
            physical_heads,
            grid=grid,
            head_weights=self.motion_probe_pck_weights,
        )
        return (
            heatmap,
            physical_heads,
            query_representations_by_block,
        )

    def _compute_object_losses(self, pipe, inputs_shared, inputs_posi):
        captured: list[dict] = []
        original_model_fn = pipe.model_fn

        def capture_main_model_fn(*args, **kwargs):
            output = original_model_fn(*args, **kwargs)
            captured.append(
                {
                    "model_output": output,
                    "latents": kwargs.get("latents"),
                    "timestep": kwargs.get("timestep"),
                    "inputs": kwargs,
                }
            )
            return output

        pipe.model_fn = capture_main_model_fn
        try:
            flow_loss, metrics = super()._compute_object_losses(
                pipe,
                inputs_shared,
                inputs_posi,
            )
        finally:
            pipe.model_fn = original_model_fn
        if len(captured) != 1:
            raise RuntimeError(
                f"expected exactly one main Student DiT forward, captured {len(captured)}"
            )
        record = captured[0]
        latent_xt = record["latents"]
        model_output = record["model_output"]
        if not isinstance(latent_xt, torch.Tensor) or not isinstance(model_output, torch.Tensor):
            raise RuntimeError("main Student capture did not contain latent x_t and v_pred")

        sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            record["timestep"],
        ).to(device=latent_xt.device, dtype=latent_xt.dtype)
        while sigma.ndim < latent_xt.ndim:
            sigma = sigma.unsqueeze(-1)
        pred_x0 = latent_xt - sigma * model_output
        target_x0 = inputs_shared["input_latents"].detach()
        pred_x0 = self._restore_condition_latents(pred_x0, target_x0, record["inputs"])

        grid = _probe_grid(self._motion_probe_dit, target_x0)
        if grid[0] != self.motion_probe_expected_latent_frames:
            raise RuntimeError(
                f"Frozen Motion Probe expected {self.motion_probe_expected_latent_frames} "
                f"latent frames, got grid={grid}"
            )
        raw_sample = inputs_shared.get("raw_sample")
        query_rows, query_source = resolve_gt_fixed_query_rows(
            raw_sample,
            grid=grid,
            token_key=self.motion_probe_query_token_key,
            mask_key=self.motion_probe_query_mask_key,
            points_key=self.motion_probe_query_points_key,
            query_latent_frame=self.motion_probe_query_latent_frame,
            object_index=self.motion_probe_query_object_index,
        )

        epsilon_p = torch.randn_like(target_x0)
        teacher_probe_input = blend_with_fixed_probe_noise(
            target_x0,
            epsilon_p,
            noise_level=self.motion_probe_noise_level,
        )
        student_probe_input = blend_with_fixed_probe_noise(
            pred_x0,
            epsilon_p,
            noise_level=self.motion_probe_noise_level,
        )
        # Preserve the exact same clean conditioning frames in both branches.
        teacher_probe_input = self._restore_condition_latents(
            teacher_probe_input,
            target_x0,
            record["inputs"],
        )
        student_probe_input = self._restore_condition_latents(
            student_probe_input,
            target_x0,
            record["inputs"],
        )
        probe_timestep = torch.full(
            (target_x0.shape[0],),
            self.motion_probe_timestep,
            device=target_x0.device,
            dtype=pipe.torch_dtype,
        )

        with torch.no_grad():
            teacher_heatmap, teacher_head_maps, gt_query_by_block = self._run_frozen_probe(
                latents=teacher_probe_input,
                timestep=probe_timestep,
                captured_inputs=record["inputs"],
                query_rows=query_rows,
                grid=grid,
                retain_input_gradient=False,
                fixed_query_by_block=None,
            )
            teacher_heatmap = teacher_heatmap.detach()
            teacher_head_maps = teacher_head_maps.detach()
            gt_query_by_block = {
                block_id: query.detach()
                for block_id, query in gt_query_by_block.items()
            }
        student_heatmap, student_head_maps, _ = self._run_frozen_probe(
            latents=student_probe_input,
            timestep=probe_timestep,
            captured_inputs=record["inputs"],
            query_rows=query_rows,
            grid=grid,
            retain_input_gradient=True,
            fixed_query_by_block=gt_query_by_block,
        )
        if teacher_heatmap.requires_grad:
            raise RuntimeError("teacher Frozen Motion Probe branch must be stop-gradient")
        if not student_heatmap.requires_grad:
            raise RuntimeError(
                "student Frozen Motion Probe heatmap lost its gradient to x0_pred"
            )

        heatmap_loss, per_head_kl = pck_weighted_teacher_student_head_kl(
            student_head_maps,
            teacher_head_maps,
            self.motion_probe_pck_weights,
        )
        equal_teacher_heatmap = aggregate_head_probabilities(
            teacher_head_maps,
            grid=grid,
        )
        equal_student_heatmap = aggregate_head_probabilities(
            student_head_maps,
            grid=grid,
        )
        equal_head_kl = teacher_student_head_kl(
            student_head_maps,
            teacher_head_maps,
        ).mean()
        legacy_aggregate_kl = student_teacher_heatmap_kl(
            equal_student_heatmap,
            equal_teacher_heatmap,
        )
        trajectory_loss, student_trajectory, teacher_trajectory = trajectory_huber_loss(
            student_heatmap,
            teacher_heatmap,
            delta=self.motion_probe_trajectory_huber_delta,
        )
        weighted_heatmap = self.motion_probe_heatmap_weight * heatmap_loss
        weighted_trajectory = self.motion_probe_trajectory_weight * trajectory_loss
        auxiliary_loss = weighted_heatmap + weighted_trajectory

        self._motion_probe_forward_count += 1
        metrics["train/motion_probe_grad_diag_applied"] = 0.0
        if (
            self._motion_probe_forward_count
            % self.motion_probe_gradient_diagnostics_every_n_forwards
            == 0
        ):
            auxiliary_gradient = torch.autograd.grad(
                auxiliary_loss,
                model_output,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[0]
            if auxiliary_gradient is None:
                raise RuntimeError(
                    "Frozen Motion Probe loss has no gradient to first-pass v_pred"
                )
            metrics["train/motion_probe_grad_diag_applied"] = 1.0
            metrics["train/motion_probe_grad_v_norm"] = float(
                auxiliary_gradient.detach().float().norm().item()
            )

        total = flow_loss + auxiliary_loss
        trajectory_distance = torch.linalg.vector_norm(
            student_trajectory.detach() - teacher_trajectory.detach(),
            dim=-1,
        ).mean()
        scheduler_sigma = context_wan._diffsynth_sigma_for_timestep(
            pipe.scheduler,
            probe_timestep,
        )
        metrics.update(
            {
                "train/loss_flow": float(flow_loss.detach().item()),
                "train/loss_motion_probe_heatmap_kl_student_teacher": float(
                    legacy_aggregate_kl.detach().item()
                ),
                "train/loss_motion_probe_equal_head_kl_teacher_student": float(
                    equal_head_kl.detach().item()
                ),
                "train/loss_motion_probe_pck_head_kl_teacher_student": float(
                    heatmap_loss.detach().item()
                ),
                "train/loss_motion_probe_trajectory_huber": float(
                    trajectory_loss.detach().item()
                ),
                "train/motion_probe_weighted_heatmap": float(
                    weighted_heatmap.detach().item()
                ),
                "train/motion_probe_weighted_trajectory": float(
                    weighted_trajectory.detach().item()
                ),
                "train/motion_probe_trajectory_distance_normalized": float(
                    trajectory_distance.item()
                ),
                "train/motion_probe_query_token_count": float(query_rows.numel()),
                "train/motion_probe_uses_teacher_q_for_student_map": 1.0,
                "train/motion_probe_query_source_precomputed": float(
                    query_source == "precomputed_token_indices"
                ),
                "train/motion_probe_query_source_mask": float(
                    query_source == "gt_object_mask"
                ),
                "train/motion_probe_query_source_points": float(
                    query_source == "gt_tracking_points"
                ),
                "train/motion_probe_timestep": self.motion_probe_timestep,
                "train/motion_probe_noise_level": self.motion_probe_noise_level,
                "train/motion_probe_scheduler_sigma": float(
                    scheduler_sigma.detach().float().item()
                ),
                "train/motion_probe_pck_score_min": float(
                    self.motion_probe_pck_audit["score_min"]
                ),
                "train/motion_probe_pck_score_max": float(
                    self.motion_probe_pck_audit["score_max"]
                ),
                "train/motion_probe_pck_weight_min": float(
                    self.motion_probe_pck_audit["weight_min"]
                ),
                "train/motion_probe_pck_weight_max": float(
                    self.motion_probe_pck_audit["weight_max"]
                ),
                "train/motion_probe_per_head_kl_min": float(
                    per_head_kl.detach().min().item()
                ),
                "train/motion_probe_per_head_kl_max": float(
                    per_head_kl.detach().max().item()
                ),
                "train/motion_probe_trainable_params": 0.0,
                "train/loss_total": float(total.detach().item()),
            }
        )
        return total, metrics


def build_parser() -> argparse.ArgumentParser:
    parser = core.build_parser()
    parser.description += (
        " Adds shared LoRA-free Wan2.2 Frozen Motion Probe heatmap/trajectory loss."
    )
    group = parser.add_argument_group("frozen_motion_probe")
    group.add_argument(
        "--motion_probe_wan_root",
        default=DEFAULT_WAN22_BASELINE_ROOT,
    )
    group.add_argument(
        "--motion_probe_head_config",
        default=DEFAULT_TOP100_CONFIG,
    )
    group.add_argument(
        "--motion_probe_head_subset_id",
        default=DEFAULT_TOP100_SUBSET,
    )
    group.add_argument(
        "--motion_probe_head_feature_subtype",
        default=DEFAULT_TOP100_SUBTYPE,
    )
    group.add_argument("--probe_timestep", type=float, required=True)
    group.add_argument("--probe_noise_level", type=float, required=True)
    group.add_argument("--motion_probe_heatmap_weight", type=float, required=True)
    group.add_argument("--motion_probe_trajectory_weight", type=float, required=True)
    group.add_argument(
        "--motion_probe_trajectory_huber_delta",
        type=float,
        default=0.05,
    )
    group.add_argument(
        "--motion_probe_query_latent_frame",
        type=int,
        default=1,
        help="Fixed F04/latent-1 query frame used by the established overlay protocol.",
    )
    group.add_argument("--motion_probe_query_object_index", type=int, default=0)
    group.add_argument(
        "--motion_probe_query_token_key",
        default="object_query_token_indices",
    )
    group.add_argument(
        "--motion_probe_query_mask_key",
        default="object_query_mask",
    )
    group.add_argument(
        "--motion_probe_query_points_key",
        default="object_query_points",
    )
    group.add_argument("--motion_probe_expected_latent_frames", type=int, default=13)
    group.add_argument(
        "--motion_probe_gradient_diagnostics_every_n_forwards",
        type=int,
        default=400,
    )
    group.add_argument(
        "--disable_motion_probe_gradient_checkpointing_offload",
        action="store_true",
        help="Disable save-on-CPU non-reentrant checkpointing inside Student probe.",
    )
    return parser


def _reject_loaded_lora(args: argparse.Namespace) -> None:
    forbidden = {
        "lora_checkpoint": args.lora_checkpoint,
        "preset_lora_path": args.preset_lora_path,
        "preset_lora_model": args.preset_lora_model,
    }
    populated = {key: value for key, value in forbidden.items() if value not in (None, "")}
    if populated:
        raise ValueError(
            "Frozen Motion Probe baseline entry does not load historical/preset LoRA: "
            f"{populated}"
        )
    if args.lora_base_model not in (None, ""):
        raise ValueError(
            "Do not pass --lora_base_model here; the new Student self-attention "
            "adapter is injected only by self_attn_adaptation_mode"
        )
    if args.trainable_models not in (None, ""):
        raise ValueError("Full-model training is disabled; train only the new Student adapter")
    resume_options = {
        "head_resume_from": getattr(args, "head_resume_from", None),
        "stage2_resume_from": getattr(args, "stage2_resume_from", None),
    }
    populated_resume = {
        key: value
        for key, value in resume_options.items()
        if value not in (None, "")
    }
    if populated_resume:
        raise ValueError(
            "This baseline entry starts a new adapter and does not load resume LoRA: "
            f"{populated_resume}"
        )


def _flatten_model_paths(value) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, (list, tuple)):
        flattened: list[str] = []
        for item in value:
            flattened.extend(_flatten_model_paths(item))
        return flattened
    if value is None:
        return []
    return [str(value)]


def _assert_main_student_uses_same_baseline(args: argparse.Namespace) -> None:
    if args.model_id_with_origin_paths not in (None, ""):
        raise ValueError(
            "Use local Wan2.2 baseline files; model_id_with_origin_paths is disabled"
        )
    expected = {
        str(Path(path).resolve())
        for path in _baseline_dit_shards(args.motion_probe_wan_root)
    }
    supplied = {
        str(Path(path).expanduser().resolve())
        for path in _flatten_model_paths(args.model_paths)
    }
    supplied_dit = {
        path
        for path in supplied
        if Path(path).name.startswith("diffusion_pytorch_model")
        and Path(path).suffix == ".safetensors"
    }
    if supplied_dit != expected:
        raise ValueError(
            "Main Student must load the same three Wan2.2-TI2V baseline DiT "
            f"shards as the frozen probe: supplied={sorted(supplied_dit)}, "
            f"expected={sorted(expected)}"
        )


def build_model(args: argparse.Namespace, accelerator):
    _reject_loaded_lora(args)
    _assert_main_student_uses_same_baseline(args)
    if not args.disable_object_branch:
        raise ValueError("Frozen Motion Probe entry requires --disable_object_branch")
    if int(args.train_batch_size) != 1:
        raise ValueError(
            "Frozen Motion Probe currently requires --train_batch_size 1 so each "
            "sample has one auditable fixed GT query set"
        )
    return core.build_model(
        args,
        accelerator,
        model_class=FrozenMotionProbeWanModule,
        extra_model_kwargs={
            "motion_probe_wan_root": args.motion_probe_wan_root,
            "motion_probe_head_config": args.motion_probe_head_config,
            "motion_probe_head_subset_id": args.motion_probe_head_subset_id,
            "motion_probe_head_feature_subtype": (
                args.motion_probe_head_feature_subtype
            ),
            "motion_probe_timestep": args.probe_timestep,
            "motion_probe_noise_level": args.probe_noise_level,
            "motion_probe_heatmap_weight": args.motion_probe_heatmap_weight,
            "motion_probe_trajectory_weight": args.motion_probe_trajectory_weight,
            "motion_probe_trajectory_huber_delta": (
                args.motion_probe_trajectory_huber_delta
            ),
            "motion_probe_query_latent_frame": (
                args.motion_probe_query_latent_frame
            ),
            "motion_probe_query_object_index": (
                args.motion_probe_query_object_index
            ),
            "motion_probe_query_token_key": args.motion_probe_query_token_key,
            "motion_probe_query_mask_key": args.motion_probe_query_mask_key,
            "motion_probe_query_points_key": args.motion_probe_query_points_key,
            "motion_probe_expected_latent_frames": (
                args.motion_probe_expected_latent_frames
            ),
            "motion_probe_gradient_checkpointing_offload": (
                not args.disable_motion_probe_gradient_checkpointing_offload
            ),
            "motion_probe_gradient_diagnostics_every_n_forwards": (
                args.motion_probe_gradient_diagnostics_every_n_forwards
            ),
            "motion_probe_device": accelerator.device,
        },
    )


def log_stage_summary(accelerator, model, args: argparse.Namespace) -> None:
    core._log_stage_summary(accelerator, model, args)
    if accelerator.is_main_process:
        accelerator.print(
            "Frozen Motion Probe: shared official Wan2.2-TI2V-5B baseline; "
            "loaded LoRA=none; trainable probe params=0; "
            f"latest3350 heads=100; probe_timestep={args.probe_timestep:g}; "
            f"probe_noise_level={args.probe_noise_level:g}; "
            f"lambda_heatmap={args.motion_probe_heatmap_weight:g}; "
            f"lambda_trajectory={args.motion_probe_trajectory_weight:g}; "
            "heatmap loss=sum_h normalized(PCK_h)*KL(teacher_h||student_h); "
            "trajectory=PCK-weighted aggregate 13-frame normalized soft-argmax Huber"
        )


def main() -> None:
    core.main(
        build_parser_fn=build_parser,
        build_model_fn=build_model,
        log_stage_summary_fn=log_stage_summary,
        require_pretrained_lora=False,
    )


if __name__ == "__main__":
    main()
