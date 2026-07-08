from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_actual_train_forward_aux_overlay as actualinspect,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)


def _parse_ctx_values(raw_value: str) -> list[int]:
    values: list[int] = []
    for item in str(raw_value).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"context length must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("ctx-values is empty")
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _build_ctx_sample(raw_sample: dict[str, Any], *, ctx_num_frames: int) -> dict[str, Any]:
    raw_video = raw_sample["video"]
    total_frames = int(raw_video.shape[1])
    if int(ctx_num_frames) > int(total_frames):
        raise ValueError(
            f"ctx_num_frames={ctx_num_frames} exceeds available train clip frames={total_frames}"
        )
    context_indices = torch.arange(int(ctx_num_frames), dtype=torch.long)
    sample = dict(raw_sample)
    sample["context_video"] = raw_video[:, context_indices].contiguous()
    sample["context_frame_indices"] = context_indices
    sample["num_context_frames"] = int(ctx_num_frames)
    sample["ctx_max_length"] = int(ctx_num_frames - 1)
    sample["sampled_ctx_last_index"] = int(ctx_num_frames - 1)
    sample["sampled_ctx_num_frames"] = int(ctx_num_frames)
    return sample


def _make_row(
    *,
    raw_sample: dict[str, Any],
    sample: dict[str, Any],
    debug: dict[str, Any],
) -> dict[str, Any]:
    sampled_source_indices = list(raw_sample.get("metadata", {}).get("sampled_frame_indices", []))
    context_local_indices = [int(v) for v in sample["context_frame_indices"].tolist()]
    context_source_frame_indices = [
        int(sampled_source_indices[idx])
        for idx in context_local_indices
        if 0 <= int(idx) < len(sampled_source_indices)
    ]
    jepa_time_idx = [int(v) for v in debug["jepa_time_idx"].detach().cpu().tolist()]
    latent_time_idx = [int(v) for v in debug["latent_time_idx"].detach().cpu().tolist()]
    jepa_input_frames = int(debug["jepa_input_video"].shape[2])
    jepa_padding_frames = int(debug["jepa_ctx_fix"].get("padded_context_frames", 0))
    latent_frames = int(debug["context_latents"].shape[2])
    jepa_token_frames = int(debug["jepa_out"].patch_tokens.shape[1])
    row = {
        "ctx_num_frames": int(sample["num_context_frames"]),
        "sampled_ctx_last_index": int(sample["sampled_ctx_last_index"]),
        "context_source_frame_indices": context_source_frame_indices,
        "latent_frames": latent_frames,
        "jepa_input_frames": jepa_input_frames,
        "jepa_padding_frames": jepa_padding_frames,
        "jepa_token_frames": jepa_token_frames,
        "jepa_time_idx": jepa_time_idx,
        "latent_time_idx": latent_time_idx,
        "jepa_input_source_indices": actualinspect._expand_with_last_index(
            context_source_frame_indices,
            jepa_input_frames,
        ),
        "jepa_time_source_indices": actualinspect._source_indices_from_time_idx(
            context_source_frame_indices,
            jepa_time_idx,
        ),
        "latent_input_source_indices": actualinspect._source_indices_from_time_idx(
            context_source_frame_indices,
            latent_time_idx,
        ),
        "metrics": debug["metrics"],
    }
    return row


def _write_summary_html(output_dir: Path, payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    table_rows = []
    for row in rows:
        table_rows.append(
            f"""
<tr>
  <td>{int(row["ctx_num_frames"])}</td>
  <td>{int(row["latent_frames"])}</td>
  <td>{int(row["jepa_input_frames"])}</td>
  <td>{int(row["jepa_padding_frames"])}</td>
  <td>{int(row["jepa_token_frames"])}</td>
  <td><code>{html.escape(str(row["latent_time_idx"]))}</code></td>
  <td><code>{html.escape(str(row["jepa_time_idx"]))}</code></td>
  <td><code>{html.escape(str(row["latent_input_source_indices"]))}</code></td>
  <td><code>{html.escape(str(row["jepa_time_source_indices"]))}</code></td>
  <td>{html.escape(json.dumps(row["metrics"], ensure_ascii=False))}</td>
</tr>
"""
        )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Actual Train Forward Ctx Sweep</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: #1f1f1f;
      background: #f5f1e8;
    }}
    .page {{ max-width: 1800px; margin: 0 auto; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #fffdf8;
    }}
    th, td {{
      border: 1px solid #d9d0c2;
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #efe6d6;
      position: sticky;
      top: 0;
    }}
    code {{ white-space: pre-wrap; }}
    pre {{
      padding: 14px;
      background: #fffdf8;
      border: 1px solid #d9d0c2;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Kubric Actual Train Forward Ctx Sweep</h1>
    <p><b>Sample key:</b> {html.escape(str(payload["sample_key"]))}</p>
    <p><b>Dataset index:</b> {int(payload["inspect_index"])}</p>
    <p><b>Video path:</b> {html.escape(str(payload["video_path"]))}</p>
    <p><b>ctx values:</b> {html.escape(str(payload["ctx_values"]))}</p>
    <table>
      <thead>
        <tr>
          <th>ctx</th>
          <th>latent_frames</th>
          <th>jepa_input_frames</th>
          <th>jepa_padding_frames</th>
          <th>jepa_token_frames</th>
          <th>latent_time_idx</th>
          <th>jepa_time_idx</th>
          <th>latent_input_source_indices</th>
          <th>jepa_time_source_indices</th>
          <th>metrics</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
    <h2>Payload</h2>
    <pre>{html.escape(json.dumps(payload, ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = trainmod.build_parser()
    parser.set_defaults(
        fixed_num_context_frames=20,
        ctx_max_length=20,
        min_context_frames=0,
        max_context_ratio=1.0,
        context_frame_choices=None,
        context_length_sampling="short_biased",
        no_context_ratio=0.0,
    )
    parser.add_argument("--inspect_index", type=int, default=59726)
    parser.add_argument("--ctx_values", type=str, default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21")
    parser.add_argument(
        "--inspect_output_dir",
        type=str,
        default="/data/gaoya/agent-data/outputs/kubric_actual_train_forward_ctx_sweep_20260708",
    )
    return parser.parse_args()


def main() -> None:
    args = trainmod.tvn.prepare_args(parse_args())
    ctx_values = _parse_ctx_values(args.ctx_values)
    output_dir = Path(args.inspect_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disabled_vggt_cache_root = output_dir / "_disabled_vggt_cache"
    disabled_vggt_cache_root.mkdir(parents=True, exist_ok=True)
    args.vggt_cache_root = str(disabled_vggt_cache_root)

    dataset = trainmod.build_dataset(args)
    raw_sample = dataset[int(args.inspect_index)]

    accelerator = SimpleNamespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = trainmod.build_model(args, accelerator)
    target_device = torch.device(model.pipe.device)
    inspectmod._move_optional_module(model.object_pooler, target_device)
    inspectmod._move_optional_module(model.object_aux_heads, target_device)
    inspectmod._move_optional_module(model.object_adapter, target_device)
    inspectmod._move_optional_module(model.vggt_adapter, target_device)

    if args.stage1a_init_from is not None:
        trainmod.tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
    inspectmod._load_optional_stage2_weights(model, args.stage2_resume_from)
    actualinspect._offload_unused_pipe_modules(model)
    torch.nn.Module.train(model, False)

    rows: list[dict[str, Any]] = []
    per_case_dir = output_dir / "cases"
    per_case_dir.mkdir(parents=True, exist_ok=True)
    for ctx_num_frames in ctx_values:
        sample = _build_ctx_sample(raw_sample, ctx_num_frames=int(ctx_num_frames))
        case_dir = per_case_dir / f"ctx_{int(ctx_num_frames):02d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            debug = actualinspect._run_forward_debug(model, sample)
        row = _make_row(raw_sample=raw_sample, sample=sample, debug=debug)
        row.update(
            {
                "sample_key": str(raw_sample.get("metadata", {}).get("sample_key", "")),
                "video_path": str(raw_sample["video_path"]),
                "case_dir": str(case_dir.relative_to(output_dir)),
            }
        )
        (case_dir / "result.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        rows.append(row)

    payload = {
        "output_dir": str(output_dir),
        "inspect_index": int(args.inspect_index),
        "sample_key": str(raw_sample.get("metadata", {}).get("sample_key", "")),
        "video_path": str(raw_sample["video_path"]),
        "ctx_values": ctx_values,
        "case_count": len(rows),
        "rows": rows,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_summary_html(output_dir, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
