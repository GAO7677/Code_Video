"""Visualize the actual replay-mixture training data through the Stage1A middleware."""
from __future__ import annotations

import sys
from pathlib import Path

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_actual_train_forward_aux_overlay as implementation,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as base_train,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve as replay_train,
)


replay_train.ContextOnlyNoGTBoxWanModule = replay_train.ReplayPreserveNoGTBoxWanModule
replay_train.prepare_jepa_context_video = base_train.prepare_jepa_context_video
replay_train.load_vggt_cache = base_train.load_vggt_cache
implementation.trainmod = replay_train

_original_inspect_one = implementation._inspect_one


def _inspect_one_with_embedded_video_fallback(*, sample, output_dir, inspect_fps, **kwargs):
    original_video_path = str(sample["video_path"])
    dataset_source = str(sample.get("metadata", {}).get("dataset_source", "unknown"))
    effective_sample = sample
    if not Path(original_video_path).is_file():
        materialized_path = output_dir / "embedded_training_clip.mp4"
        implementation.inspectmod._write_tensor_video(
            materialized_path,
            sample["video"],
            fps=int(inspect_fps),
        )
        effective_sample = dict(sample)
        effective_sample["video_path"] = str(materialized_path)

    result = _original_inspect_one(
        sample=effective_sample,
        output_dir=output_dir,
        inspect_fps=inspect_fps,
        **kwargs,
    )
    result["video_path"] = original_video_path
    result["dataset_source"] = dataset_source
    result["caption"] = f"[dataset_source={dataset_source}] {result['caption']}"
    result["sample_key"] = f"{dataset_source}:{result.get('sample_key', '')}"
    return result


implementation._inspect_one = _inspect_one_with_embedded_video_fallback


def _argument_value(flag: str) -> str | None:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    return sys.argv[index + 1] if index + 1 < len(sys.argv) else None


def _rename_gallery_titles(output_dir: Path) -> None:
    for html_path in output_dir.rglob("index.html"):
        text = html_path.read_text(encoding="utf-8")
        text = text.replace("Kubric No-GT-Box", "Replay Mix No-GT-Box")
        text = text.replace("current Kubric no-GT-box training dataset", "current replay-mixture training dataset")
        html_path.write_text(text, encoding="utf-8")


def main() -> None:
    implementation.main()
    output_dir = _argument_value("--inspect_output_dir")
    if output_dir:
        _rename_gallery_titles(Path(output_dir).expanduser().resolve())


if __name__ == "__main__":
    main()
