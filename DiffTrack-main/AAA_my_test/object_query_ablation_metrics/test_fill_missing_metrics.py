from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from AAA_my_test.object_query_ablation_metrics.fill_missing_metrics import (
    build_dynamic_inventory,
    complete25_report_complete,
    select_balanced_shard,
)


class FillMissingMetricsTest(unittest.TestCase):
    def test_dynamic_inventory_has_baseline_and_unique_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline.mp4"
            candidate = root / "candidate.mp4"
            baseline.write_bytes(b"baseline")
            candidate.write_bytes(b"candidate")
            rows = [
                {
                    "variant_id": "object_A__self_future__top100",
                    "target_scope": "single_object",
                    "region": "object_A",
                    "mask_mode": "self_future",
                    "head_scope": "top100",
                    "path": str(candidate),
                }
            ]
            with patch(
                "AAA_my_test.object_query_ablation_metrics.fill_missing_metrics.locate_baseline",
                return_value=baseline,
            ):
                path, payload = build_dynamic_inventory(
                    root, "case", 7, rows, write=True
                )
            self.assertTrue(path.is_file())
            self.assertEqual(payload["video_count"], 2)
            self.assertEqual([row["id"] for row in payload["videos"]], ["baseline", rows[0]["variant_id"]])

    def test_complete25_accepts_dynamic_ablation_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identifier in ("a", "b"):
                assets = {
                    key: f"{identifier}_{key}.mp4"
                    for key in ("trajectory", "mask", "pixel", "raft")
                }
                for value in assets.values():
                    (root / value).write_bytes(b"video")
                assets["perceptual"] = {}
                for object_name in ("object_A", "object_B"):
                    assets["perceptual"][object_name] = {}
                    for reference in ("baseline", "source_gt_video"):
                        relative = f"{identifier}_{object_name}_{reference}.jpg"
                        (root / relative).write_bytes(b"image")
                        assets["perceptual"][object_name][reference] = relative
                records.append(
                    {
                        "id": identifier,
                        "assets": assets,
                        "vbench": {
                            f"metric_{index}": {"score": 0.5} for index in range(7)
                        },
                    }
                )
            report = {
                "video_count": 3,
                "ablation_count": 2,
                "metric_definitions": [{} for _ in range(25)],
                "baseline": {
                    "vbench": {
                        f"metric_{index}": {"score": 0.5} for index in range(7)
                    }
                },
                "records": records,
            }
            path = root / "report.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertTrue(complete25_report_complete(path, {"a", "b"}, True))

    def test_complete25_rejects_missing_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "report.json"
            path.write_text(
                json.dumps(
                    {
                        "video_count": 2,
                        "ablation_count": 1,
                        "metric_definitions": [{} for _ in range(25)],
                        "records": [{"id": "a", "assets": {}, "vbench": {}}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(complete25_report_complete(path, {"a"}, False))

    def test_balanced_shards_are_disjoint_and_complete(self) -> None:
        entries = [
            {"case": f"c{index}", "seed": index, "candidates": [{}] * weight}
            for index, weight in enumerate((9, 8, 7, 6, 5, 4, 3))
        ]
        selected = [select_balanced_shard(entries, 3, index)[0] for index in range(3)]
        identities = [
            (row["case"], row["seed"])
            for shard in selected
            for row in shard
        ]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(set(identities), {(row["case"], row["seed"]) for row in entries})


if __name__ == "__main__":
    unittest.main()
