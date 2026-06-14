from __future__ import annotations

import torch


class WanFlowMatchScheduler:
    """Minimal local copy of the Wan Flow Matching schedule used by DiffSynth-Studio."""

    def __init__(self, num_train_timesteps: int = 1000, shift: float = 5.0) -> None:
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.sigmas = None
        self.timesteps = None
        self.linear_timesteps_weights = None
        self.set_timesteps(self.num_train_timesteps, training=True)

    def _timestep_index(self, timestep: torch.Tensor) -> int:
        if self.timesteps is None:
            raise RuntimeError("scheduler timesteps are not initialized")
        timestep_cpu = timestep.detach().cpu().to(dtype=self.timesteps.dtype)
        return int(torch.argmin((self.timesteps - timestep_cpu).abs()).item())

    def set_timesteps(self, num_inference_steps: int = 1000, training: bool = False) -> None:
        sigma_min = 0.0
        sigma_max = 1.0
        sigmas = torch.linspace(sigma_max, sigma_min, int(num_inference_steps) + 1)[:-1]
        sigmas = self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)
        self.sigmas = sigmas
        self.timesteps = sigmas * float(self.num_train_timesteps)
        if training:
            self._set_training_weight()

    def _set_training_weight(self) -> None:
        assert self.timesteps is not None
        steps = float(self.num_train_timesteps)
        x = self.timesteps
        y = torch.exp(-2.0 * ((x - steps / 2.0) / steps) ** 2)
        y_shifted = y - y.min()
        weights = y_shifted * (steps / y_shifted.sum().clamp_min(1e-6))
        if len(self.timesteps) != self.num_train_timesteps:
            weights = weights * (len(self.timesteps) / steps)
            weights = weights + weights[1]
        self.linear_timesteps_weights = weights

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        timestep_id = self._timestep_index(timestep)
        sigma = self.sigmas[timestep_id].to(device=original_samples.device, dtype=original_samples.dtype)
        return (1.0 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

    def training_weight(self, timestep: torch.Tensor, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        timestep_id = self._timestep_index(timestep)
        weight = self.linear_timesteps_weights[timestep_id]
        return weight.to(device=device, dtype=dtype)
