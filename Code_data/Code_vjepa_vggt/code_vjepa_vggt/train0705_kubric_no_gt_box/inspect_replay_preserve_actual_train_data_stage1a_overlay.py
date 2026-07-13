"""Visualize the exact replay-preserve data contract and Stage1A middleware."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import code_vjepa_vggt.train0705_kubric_no_gt_box.inspect_kubric_actual_train_forward_aux_overlay as inspect
import code_vjepa_vggt.train0705_kubric_no_gt_box.train_stage1b_no_gt_box_replay_preserve as replay


def _install_replay_training_entry() -> None:
    # Keep the established renderer but construct data/model from the trainer
    # that is currently running.
    base_train = inspect.trainmod
    inspect.trainmod = SimpleNamespace(
        ContextOnlyNoGTBoxWanModule=base_train.ContextOnlyNoGTBoxWanModule,
        build_parser=replay.build_parser,
        build_dataset=replay.build_dataset,
        build_model=replay.build_model,
        load_vggt_cache=base_train.load_vggt_cache,
        prepare_jepa_context_video=base_train.prepare_jepa_context_video,
        tvn=base_train.tvn,
    )

    original_export = inspect.inspectmod._export_browser_video

    def export_browser_video(source_path: Path, output_path: Path) -> Path:
        if source_path.is_file():
            return original_export(source_path, output_path)
        train_clip = output_path.parent / "train_clip_full.mp4"
        if not train_clip.is_file():
            raise FileNotFoundError(
                f"Neither source video nor decoded training clip exists: {source_path}"
            )
        return original_export(train_clip, output_path)

    inspect.inspectmod._export_browser_video = export_browser_video


def _rewrite_gallery_labels(output_dir: Path) -> None:
    replacements = {
        "Kubric Actual Train Object-Branch Middleware Overlay Gallery":
            "Replay-Mix Actual Train Stage1A/Object-Branch Overlay Gallery",
        "Kubric Actual Train Object-Branch Middleware Overlay":
            "Replay-Mix Actual Train Stage1A/Object-Branch Overlay",
        "69-frame train clip": "49-frame train clip",
        "current Kubric no-GT-box training dataset configuration":
            "current PyBullet/Kubric/OpenVid replay-preserve training dataset configuration",
    }
    for html_path in output_dir.rglob("*.html"):
        text = html_path.read_text(encoding="utf-8")
        for source, target in replacements.items():
            text = text.replace(source, target)
        html_path.write_text(text, encoding="utf-8")


def main() -> None:
    _install_replay_training_entry()
    inspect.main()

    import sys

    if "--inspect_output_dir" in sys.argv:
        output_dir = Path(sys.argv[sys.argv.index("--inspect_output_dir") + 1]).resolve()
        _rewrite_gallery_labels(output_dir)


if __name__ == "__main__":
    main()
