"""Aspect-ratio-aware distributed batching."""

import math

import torch
from torch.utils.data import Sampler

from .transform import choose_aspect_ratio_bucket


class DistributedAspectRatioBatchSampler(Sampler):
    """Build shape-homogeneous batches with identical bucket order on all ranks."""

    def __init__(
        self,
        dataset,
        buckets,
        batch_size,
        num_replicas,
        rank,
        seed=0,
        drop_last=False,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0 <= rank < num_replicas:
            raise ValueError(f"Invalid rank {rank} for world size {num_replicas}")
        if not hasattr(dataset, "get_spatial_shapes"):
            raise TypeError("Aspect-ratio batching requires dataset.get_spatial_shapes()")

        self.buckets = [tuple(int(value) for value in bucket) for bucket in buckets]
        self.batch_size = int(batch_size)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.global_batch_size = self.batch_size * self.num_replicas

        grouped = {bucket: [] for bucket in self.buckets}
        spatial_shapes = dataset.get_spatial_shapes()
        if len(spatial_shapes) != len(dataset):
            raise RuntimeError(
                f"Spatial shape count {len(spatial_shapes)} != dataset size {len(dataset)}"
            )
        for index, (height, width) in enumerate(spatial_shapes):
            bucket = choose_aspect_ratio_bucket(height, width, self.buckets)
            grouped[bucket].append(index)
        self.grouped_indices = grouped
        self.group_batch_counts = {
            bucket: self._batch_count(len(indices))
            for bucket, indices in self.grouped_indices.items()
        }

    def _batch_count(self, sample_count):
        if self.drop_last:
            return sample_count // self.global_batch_size
        return math.ceil(sample_count / self.global_batch_size)

    def __len__(self):
        return sum(self.group_batch_counts.values())

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_batches = []
        for bucket, source_indices in self.grouped_indices.items():
            if not source_indices:
                continue
            order = torch.randperm(len(source_indices), generator=generator).tolist()
            indices = [source_indices[position] for position in order]
            batch_count = self.group_batch_counts[bucket]
            target_count = batch_count * self.global_batch_size
            if self.drop_last:
                indices = indices[:target_count]
            elif len(indices) < target_count:
                repeats = math.ceil((target_count - len(indices)) / len(indices))
                indices += (indices * repeats)[: target_count - len(indices)]
            for start in range(0, len(indices), self.global_batch_size):
                global_batches.append((bucket, indices[start : start + self.global_batch_size]))

        batch_order = torch.randperm(len(global_batches), generator=generator).tolist()
        rank_start = self.rank * self.batch_size
        rank_end = rank_start + self.batch_size
        for position in batch_order:
            _, global_indices = global_batches[position]
            local_indices = global_indices[rank_start:rank_end]
            if len(local_indices) != self.batch_size:
                raise RuntimeError(
                    f"Rank {self.rank} received incomplete batch of {len(local_indices)}"
                )
            yield local_indices

    def summary(self):
        groups = {}
        for bucket, indices in self.grouped_indices.items():
            batches = self.group_batch_counts[bucket]
            padded = batches * self.global_batch_size
            groups[f"{bucket[0]}x{bucket[1]}"] = {
                "source_samples": len(indices),
                "global_batches": batches,
                "samples_after_padding": padded,
                "padding_samples": max(0, padded - len(indices)),
            }
        return {
            "batch_size_per_rank": self.batch_size,
            "world_size": self.num_replicas,
            "global_batch_size": self.global_batch_size,
            "drop_last": self.drop_last,
            "batches_per_rank": len(self),
            "groups": groups,
        }
