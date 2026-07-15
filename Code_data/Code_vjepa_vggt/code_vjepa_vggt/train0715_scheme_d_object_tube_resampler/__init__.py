"""Scheme-D learned object-tube conditioning for Wan video generation."""

from .models import ObjectTubeResampler, prune_object_cross_attention_blocks

__all__ = ["ObjectTubeResampler", "prune_object_cross_attention_blocks"]
