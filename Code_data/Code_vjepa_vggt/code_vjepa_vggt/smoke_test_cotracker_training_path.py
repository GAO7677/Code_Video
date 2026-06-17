from __future__ import annotations

import json
from pathlib import Path

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config


def main() -> None:
    cfg = load_yaml_config(
        "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_probe_gpu67.yaml"
    )
    cfg["model"]["track_source"] = "cotracker"
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1
    trainer = ContextVideoTrainer(cfg, build_optimizer=False)
    debug = trainer.inspect_one_batch()
    out_path = Path("/data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/smoke/smoke_train_shapes_cotracker.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, ensure_ascii=False)
    print(out_path)


if __name__ == "__main__":
    main()
