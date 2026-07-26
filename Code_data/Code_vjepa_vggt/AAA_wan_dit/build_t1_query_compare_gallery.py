#!/usr/bin/env python3
"""Build moving/fixed/context-t1 query comparisons for classified heads."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from build_moving_query_head_overlay_gallery import (
    MODEL_LABELS,
    MODELS,
    _render_fixed_query_head,
    _render_head,
    _render_qk_matrix,
)
from motion_query_map import _center_crop_resize, _read_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maps-root", type=Path, required=True)
    parser.add_argument("--fixed-query-root", type=Path)
    parser.add_argument("--full-matrix-root", type=Path, required=True)
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--query-map", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--block", type=int, default=17)
    parser.add_argument("--step", type=int, default=35)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _model_rows(classification: dict, model: str) -> list[dict]:
    return [row for row in classification["heads"] if row["model"] == model]


def _classification_table(rows: list[dict]) -> str:
    cells = []
    for row in sorted(rows, key=lambda item: int(item["head"])):
        class_name = str(row["class"])
        cells.append(
            f"""<tr class="{class_name}">
<td>H{int(row['head']):02d}</td><td>{html.escape(class_name)}</td>
<td>{float(row['same_frame_mass']):.4f}</td>
<td>{float(row['outside_frame_mass']):.4f}</td>
<td>{float(row['same_vs_outside_density_log2']):+.3f}</td>
<td>{float(row['past_frame_mass']):.4f}</td>
<td>{float(row['future_frame_mass']):.4f}</td></tr>"""
        )
    return f"""<details><summary>All 24 heads: context-t1 inside/outside classification</summary>
<div class="table-scroll"><table><thead><tr>
<th>Head</th><th>Class</th><th>Same mass</th><th>Outside mass</th>
<th>log2 density ratio</th><th>Past mass</th><th>Future mass</th>
</tr></thead><tbody>{''.join(cells)}</tbody></table></div></details>"""


def main() -> None:
    args = parse_args()
    maps_root = args.maps_root.expanduser().resolve()
    matrix_root = args.full_matrix_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    classification = json.loads(
        args.classification.expanduser().resolve().read_text(encoding="utf-8")
    )
    query_map = json.loads(
        args.query_map.expanduser().resolve().read_text(encoding="utf-8")
    )["cases"]
    item = query_map[args.case]
    source_frames = [
        _center_crop_resize(frame)
        for frame in _read_video(Path(item["source_video"]))
    ]
    manifest = {
        "case": args.case,
        "block": args.block,
        "step": args.step,
        "context_raw_frames": 8,
        "context_latent_times": [0, 1],
        "query_time": 1,
        "models": {},
    }
    sections = []
    for model in MODELS:
        moving_summary_path = (
            maps_root
            / f"block{args.block:02d}"
            / "matrices"
            / model
            / args.case
            / "summary.json"
        )
        moving_summary = json.loads(
            moving_summary_path.read_text(encoding="utf-8")
        )
        moving_entry = next(
            entry
            for entry in moving_summary["steps"]
            if int(entry["step_number_one_based"]) == args.step
        )
        with np.load(
            moving_summary_path.parent
            / moving_entry["directory"]
            / moving_entry["maps_npz"]
        ) as arrays:
            moving_maps = arrays["attention"].astype(np.float32)
            selected_heads = arrays["selected_heads"].astype(int)
            moving_coords = arrays["query_coords"].astype(int)
        if selected_heads.tolist() != list(range(24)):
            raise ValueError(f"{model} moving maps do not contain all heads")
        t1_coords = moving_coords[moving_coords[:, 0] == 1]
        fixed_coords = moving_coords[moving_coords[:, 0] == 2]
        if len(fixed_coords) == 0:
            raise ValueError(f"{model} moving maps do not contain latent t2 queries")
        fixed_maps = moving_maps[:, 2]
        if "full_matrix_npz" in moving_entry:
            matrix_npz = (
                moving_summary_path.parent
                / moving_entry["directory"]
                / moving_entry["full_matrix_npz"]
            )
        else:
            matrix_npz = (
                matrix_root
                / model
                / args.case
                / f"step_{args.step:02d}"
                / f"block{args.block:02d}_all_heads_token_matrix.npz"
            )
        cards = []
        manifest["models"][model] = {}
        representatives = classification["representatives"][model]
        selected = [
            ("most_in_frame", rank, int(head))
            for rank, head in enumerate(representatives["most_in_frame"], 1)
        ] + [
            ("most_out_frame_leaning", rank, int(head))
            for rank, head in enumerate(
                representatives["most_out_frame_leaning"], 1
            )
        ]
        row_index = {
            int(row["head"]): row for row in _model_rows(classification, model)
        }
        for class_name, rank, head in selected:
            actual_class = str(row_index[head]["class"])
            label = (
                f"{class_name.replace('_', ' ')} rank {rank} "
                f"(class: {actual_class})"
            )
            stem = assets / f"{model}_{class_name}{rank}_head{head:02d}"
            moving_video, moving_sheet = _render_head(
                source_frames=source_frames,
                attention=moving_maps[head],
                query_coords=moving_coords,
                model=model,
                role=label,
                head=head,
                block=args.block,
                step=args.step,
                output_stem=stem.with_name(stem.name + "_moving"),
            )
            fixed_video, fixed_sheet = _render_fixed_query_head(
                source_frames=source_frames,
                attention=fixed_maps[head],
                fixed_query_coords=fixed_coords,
                role=label,
                head=head,
                block=args.block,
                step=args.step,
                output_stem=stem.with_name(stem.name + "_fixed_frame08"),
            )
            t1_video, t1_sheet = _render_fixed_query_head(
                source_frames=source_frames,
                attention=moving_maps[head, 1],
                fixed_query_coords=t1_coords,
                role=label,
                head=head,
                block=args.block,
                step=args.step,
                output_stem=stem.with_name(stem.name + "_context_t1"),
                query_frame_index=4,
                reference_title="Context query: latent t1 (frame 4 reference)",
                attention_title="Context-t1 attention A(q_t1, k_t)",
                query_description="context object q=t1",
            )
            qk_path = stem.with_name(stem.name + "_full_qk.png")
            _render_qk_matrix(
                matrix_npz=matrix_npz,
                head=head,
                query_coords=moving_coords,
                fixed_query_coords=fixed_coords,
                context_t1_coords=t1_coords,
                model=model,
                block=args.block,
                step=args.step,
                output_path=qk_path,
            )
            stats = row_index[head]
            manifest["models"][model][f"{class_name}_{rank}"] = {
                "head": head,
                "selection_group": class_name,
                "class": actual_class,
                "same_frame_mass": stats["same_frame_mass"],
                "outside_frame_mass": stats["outside_frame_mass"],
                "same_vs_outside_density_log2": stats[
                    "same_vs_outside_density_log2"
                ],
            }
            cards.append(
                f"""<article>
<h3>{html.escape(label)} · Head {head}</h3>
<p class="metrics">same mass {float(stats['same_frame_mass']):.4f} ·
outside mass {float(stats['outside_frame_mass']):.4f} ·
log2 density ratio {float(stats['same_vs_outside_density_log2']):+.3f}</p>
<div class="videos">
<figure><video controls loop muted preload="metadata" src="assets/{moving_video.name}"></video>
<figcaption>Moving object query at every latent time</figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{fixed_video.name}"></video>
<figcaption>Fixed frame-8 object query at latent t2</figcaption></figure>
<figure><video controls loop muted preload="metadata" src="assets/{t1_video.name}"></video>
<figcaption>Context object query fixed at latent t1</figcaption></figure>
</div>
<div class="images">
<figure><a href="assets/{moving_sheet.name}"><img loading="lazy" src="assets/{moving_sheet.name}"></a><figcaption>Moving Q</figcaption></figure>
<figure><a href="assets/{fixed_sheet.name}"><img loading="lazy" src="assets/{fixed_sheet.name}"></a><figcaption>Fixed frame-8 Q</figcaption></figure>
<figure><a href="assets/{t1_sheet.name}"><img loading="lazy" src="assets/{t1_sheet.name}"></a><figcaption>Context t1 Q</figcaption></figure>
<figure><a href="assets/{qk_path.name}"><img loading="lazy" src="assets/{qk_path.name}"></a><figcaption>Full pooled QK matrix</figcaption></figure>
</div></article>"""
            )
        sections.append(
            f"""<section><h2>{html.escape(MODEL_LABELS[model])}</h2>
<p class="count">Classification count: in-frame
{int(representatives['class_counts']['in_frame'])}/24 · out-frame
{int(representatives['class_counts']['out_frame'])}/24.</p>
{_classification_table(_model_rows(classification, model))}
<div class="cards">{''.join(cards)}</div></section>"""
        )
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Context-t1 object-query head classification</title><style>
body{{margin:0;background:#f1f3f0;color:#202421;font:14px Arial,sans-serif}}
header,main{{max-width:1880px;margin:auto;padding:18px 24px}}header{{max-width:none;background:#202421;color:#fff}}
h1,h2,h3{{letter-spacing:0}}h1{{margin:0 0 8px}}header p{{margin:5px 0;color:#d1d7d2}}
section{{border-top:2px solid #222;margin:28px 0 46px;padding-top:12px}}
details{{background:#fff;border:1px solid #ccd0cc;padding:10px;margin-bottom:14px}}summary{{cursor:pointer;font-weight:bold}}
.table-scroll{{overflow:auto;max-height:430px;margin-top:10px}}table{{border-collapse:collapse;width:100%;font-size:12px}}
th,td{{padding:6px 8px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}
tr.in_frame{{background:#edf9f0}}tr.out_frame{{background:#fff2ec}}
.cards{{display:grid;grid-template-columns:1fr;gap:18px}}article{{background:#fff;border:1px solid #c9cec9;border-radius:4px;padding:10px}}
h3{{margin:0 0 4px}}.metrics{{margin:0 0 9px;color:#555}}.videos{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}}
.count{{font-weight:bold}}
.images{{display:grid;grid-template-columns:repeat(3,1.15fr) 0.85fr;gap:10px;margin-top:10px}}
figure{{margin:0}}video,img{{display:block;width:100%;height:auto;background:#111}}figcaption{{padding:6px 2px;color:#555;font-size:12px}}
@media(max-width:1100px){{.videos,.images{{grid-template-columns:1fr}}header,main{{padding:14px}}}}
</style></head><body><header><h1>Context-t1 object-query head classification</h1>
<p>{html.escape(args.case)} · Block {args.block} · denoise step {args.step}</p>
<p>Eight raw context frames produce two DiT latent times, t0 and t1. Classification uses the object Q at t1.
Inside/outside compares attention density, not raw mass: score =
log2((same_mass/(1/13))/(outside_mass/(12/13))). Positive is in-frame; negative is out-frame.</p>
<p>QK markers: green = moving object Q/K, blue square = fixed frame-8/t2 Q,
yellow diamond = context-t1 Q.</p></header><main>{''.join(sections)}</main></body></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    print(output / "index.html")


if __name__ == "__main__":
    main()
