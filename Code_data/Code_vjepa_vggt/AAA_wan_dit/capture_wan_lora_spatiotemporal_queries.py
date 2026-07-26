#!/usr/bin/env python3
"""Run Wan+LoRA and record exact per-query same/cross-frame attention."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

from self_attention_matrix import (
    DiffSynthAttentionScope,
    MatrixCaptureConfig,
    install_diffsynth_block_recorder,
    parse_step_numbers,
)
from spatiotemporal_query_attention import ExactSpatiotemporalQueryRecorder


def _extract_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--attention-output-root", type=Path, required=True)
    parser.add_argument("--attention-block", type=int, default=17)
    parser.add_argument("--attention-steps", default="5,15,25,35")
    parser.add_argument("--attention-query-chunk", type=int, default=128)
    parser.add_argument("--seed-map-json", type=Path, required=True)
    return parser.parse_known_args(argv)


def _cli_path(argv: list[str], name: str) -> Path:
    for index, token in enumerate(argv):
        if token == name:
            return Path(argv[index + 1]).expanduser().resolve()
        if token.startswith(f"{name}="):
            return Path(token.split("=", 1)[1]).expanduser().resolve()
    raise ValueError(f"missing required base option {name}")


def _load_schedule(
    input_list: Path, seed_map_path: Path
) -> dict[Path, deque[dict]]:
    seed_map = json.loads(seed_map_path.read_text(encoding="utf-8"))
    queues: dict[Path, deque[dict]] = defaultdict(deque)
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        input_json = Path(line.strip()).expanduser().resolve()
        payload = json.loads(input_json.read_text(encoding="utf-8"))
        stem = input_json.stem
        if stem not in seed_map:
            raise KeyError(f"{stem} is missing from {seed_map_path}")
        entry = dict(seed_map[stem])
        entry["input_json"] = str(input_json)
        context = Path(payload["input_video"]).expanduser().resolve()
        queues[context].append(entry)
    return queues


def main() -> None:
    custom, remaining = _extract_args(sys.argv[1:])
    input_list = _cli_path(remaining, "--input-json-list-path")
    schedule = _load_schedule(
        input_list, custom.seed_map_json.expanduser().resolve()
    )
    config = MatrixCaptureConfig(
        block_id=int(custom.attention_block),
        step_numbers=parse_step_numbers(custom.attention_steps),
        output_bins=1,
        query_chunk=int(custom.attention_query_chunk),
    )

    from code_vjepa_vggt.AAAinfer import wan_openvid_0613pybullet_lorav2v as base

    original_generate = base.core.generate_one_video

    def generate_with_attention(*args, **kwargs):
        pipe = kwargs["pipe"]
        context_path = Path(kwargs["context_path"]).expanduser().resolve()
        if context_path not in schedule or not schedule[context_path]:
            raise RuntimeError(f"no scheduled run remains for {context_path}")
        entry = schedule[context_path].popleft()
        seed = int(entry["seed"])
        kwargs["seed"] = seed
        recorder = ExactSpatiotemporalQueryRecorder(
            config=config,
            model_label="wan_lora",
            output_root=custom.attention_output_root,
        )
        recorder.begin_case(
            str(entry["case_key"]),
            metadata={
                **entry,
                "input_video": str(context_path),
            },
        )
        restore_block = install_diffsynth_block_recorder(pipe.dit, recorder)
        scope = DiffSynthAttentionScope(
            pipe=pipe,
            recorder=recorder,
            cfg_scale=float(kwargs["cfg_scale"]),
        )
        scope.install()
        try:
            result = original_generate(*args, **kwargs)
        finally:
            scope.restore()
            restore_block()
        recorder.finalize_case()
        return result

    base.core.generate_one_video = generate_with_attention
    sys.argv = [sys.argv[0], *remaining]
    base.main()


if __name__ == "__main__":
    main()
