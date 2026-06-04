from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2

from .paths import PROXY_CKPT, PROXY_REPO, REPO_ROOT, TMP_ROOT
from .records import stable_path_id


class ProxyRunner:
    def __init__(self, *, device: str = "cuda") -> None:
        self.device = device
        self._scorer = None
        self._video_utils = None

    def _lazy_imports(self) -> None:
        if self._scorer is not None:
            return
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from rerank_video.scorers import JEPAPredictiveScorer
        from rerank_video.schemas import JEPAScoreConfig
        from rerank_video.video_utils import ensure_dir, load_video_frames, uniform_subsample_frames

        self._ensure_dir = ensure_dir
        self._load_video_frames = load_video_frames
        self._uniform_subsample_frames = uniform_subsample_frames
        self._scorer = JEPAPredictiveScorer(
            JEPAScoreConfig(
                backend="vjepa2",
                device=self.device,
                max_frames=32,
                context_frames=8,
                future_frames=16,
                context_repeat_frames=8,
                crop_size=384,
                vjepa_checkpoint=PROXY_CKPT,
                vjepa_repo_root=PROXY_REPO,
                vjepa_model_name="vjepa2_1_vit_large_384",
            )
        )

    def _write_clip(self, path: Path, frames: list[Any], fps: int) -> None:
        self._ensure_dir(path.parent)
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(int(fps), 1), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {path}")
        try:
            for frame in frames:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def score(self, video_path: Path, *, context_video_path: Path | None = None) -> dict[str, Any] | None:
        self._lazy_imports()
        candidate_frames_all = self._load_video_frames(video_path)
        total = len(candidate_frames_all)
        if total < 30:
            return None
        context_source_path = context_video_path or video_path
        context_frames_all = self._load_video_frames(context_source_path)
        if len(context_frames_all) < 2:
            return None

        # For explicit conditioning clips such as PhyGenBench first-frame ctx08,
        # use the whole provided context video instead of halving it again.
        if context_video_path is not None:
            context_split = len(context_frames_all)
        else:
            context_split = min(60, len(context_frames_all) // 2)
        future_split = min(60, total // 2)
        context_frames = self._uniform_subsample_frames(context_frames_all[:context_split], 8)
        future_frames = self._uniform_subsample_frames(candidate_frames_all[future_split:], 16)
        if len(context_frames) < 2 or len(future_frames) < 2:
            return None

        tmp_root = self._ensure_dir(TMP_ROOT / "proxy" / stable_path_id(video_path))
        context_path = tmp_root / "context.mp4"
        future_path = tmp_root / "future.mp4"
        self._write_clip(context_path, context_frames, fps=16)
        self._write_clip(future_path, future_frames, fps=16)

        score, details = self._scorer.score(context_video_path=context_path, candidate_video_path=future_path)
        return {
            "score": float(score),
            "context_frames": len(context_frames),
            "future_frames": len(future_frames),
            "context_video": str(context_source_path),
            "details": details,
        }
