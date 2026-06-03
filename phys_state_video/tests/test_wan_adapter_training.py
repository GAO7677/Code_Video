from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch
from phys_state_video.wan_adapter_training import (
    REQUIRED_STATE_ADAPTER_KEYS_TI2V,
    align_wan_frame_num,
    build_first_frame_mask,
    build_ti2v_timestep_tensor,
    build_ti2v_training_video,
    discover_state_condition_bundles,
    is_ti2v_state_adapter_checkpoint,
)

torch = require_torch()


class WanAdapterTrainingTests(unittest.TestCase):
    def test_align_wan_frame_num(self):
        self.assertEqual(align_wan_frame_num(9), 9)
        self.assertEqual(align_wan_frame_num(10), 13)

    def test_build_ti2v_training_video_uses_first_context_and_pads_last_frame(self):
        context = torch.arange(4 * 3 * 2 * 2, dtype=torch.float32).view(4, 3, 2, 2)
        future = 1000 + torch.arange(6 * 3 * 2 * 2, dtype=torch.float32).view(6, 3, 2, 2)

        video = build_ti2v_training_video(context, future)

        self.assertEqual(tuple(video.shape), (9, 3, 2, 2))
        self.assertTrue(torch.equal(video[0], context[0]))
        self.assertTrue(torch.equal(video[1:7], future))
        self.assertTrue(torch.equal(video[7], future[-1]))
        self.assertTrue(torch.equal(video[8], future[-1]))

    def test_build_first_frame_mask_and_timestep_tensor(self):
        latent = torch.zeros(16, 3, 8, 8)
        mask = build_first_frame_mask(latent)
        timestep = torch.tensor([500.0], dtype=torch.float32)
        timestep_tokens = build_ti2v_timestep_tensor(mask, timestep=timestep, seq_len=48)

        self.assertEqual(tuple(mask.shape), (16, 3, 8, 8))
        self.assertTrue(torch.equal(mask[:, 0], torch.zeros_like(mask[:, 0])))
        self.assertTrue(torch.equal(mask[:, 1:], torch.ones_like(mask[:, 1:])))
        self.assertEqual(tuple(timestep_tokens.shape), (1, 48))
        self.assertEqual(float(timestep_tokens[0, 0].item()), 0.0)
        self.assertEqual(float(timestep_tokens[0, -1].item()), 500.0)

    def test_discover_state_condition_bundles_from_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_dir = root / "episode_0000"
            bundle_dir.mkdir()
            episode_path = root / "episode_0000.npz"
            np.savez_compressed(
                episode_path,
                context_frames=np.zeros((4, 3, 8, 8), dtype=np.float32),
                future_frames=np.zeros((6, 3, 8, 8), dtype=np.float32),
            )
            np.savez_compressed(bundle_dir / "state_condition.npz", state_tokens=np.zeros((2, 128), dtype=np.float32))
            (bundle_dir / "input_image.png").write_bytes(b"not-used")
            (bundle_dir / "prompt.txt").write_text("test prompt", encoding="utf-8")
            (bundle_dir / "meta.json").write_text(
                json.dumps(
                    {
                        "sample_id": "episode_0000",
                        "episode_path": str(episode_path),
                        "prompt": "test prompt",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "sample_id": "episode_0000",
                        "state_condition_path": str(bundle_dir / "state_condition.npz"),
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            records = discover_state_condition_bundles(root)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sample_id, "episode_0000")
            self.assertEqual(records[0].prompt, "test prompt")
            self.assertEqual(records[0].episode_path, episode_path.resolve())

    def test_is_ti2v_state_adapter_checkpoint(self):
        good = {key: object() for key in REQUIRED_STATE_ADAPTER_KEYS_TI2V}
        bad = {"state_adapter": object()}
        self.assertTrue(is_ti2v_state_adapter_checkpoint(good))
        self.assertFalse(is_ti2v_state_adapter_checkpoint(bad))


if __name__ == "__main__":
    unittest.main()
