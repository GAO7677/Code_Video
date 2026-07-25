#!/usr/bin/env python3
"""Build a static browser for per-head self-attention matrix heatmaps."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from self_attention_matrix import _frame_boundaries, _render_contact_sheet


MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--rerender-contacts", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    output = (args.output or (root / "_gallery")).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for summary_path in sorted(root.glob("*/*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["_summary_path"] = summary_path
        records.append(payload)

    cases = sorted({str(record["case"]) for record in records})
    models = ["wan_lora", "xssc", "physrvg"]
    index = {(str(record["case"]), str(record["model"])): record for record in records}
    case_sections: list[str] = []
    for case in cases:
        method_cards: list[str] = []
        for model in models:
            record = index.get((case, model))
            if record is None:
                method_cards.append(
                    f"<article><h3>{html.escape(MODEL_LABELS[model])}</h3><p>pending</p></article>"
                )
                continue
            summary_path = Path(record["_summary_path"])
            step_cards = []
            for step in record["steps"]:
                step_dir = summary_path.parent / str(step["directory"])
                contact = step_dir / str(step["contact_sheet"])
                if args.rerender_contacts:
                    matrix_path = step_dir / str(step["matrix_npz"])
                    with np.load(matrix_path) as arrays:
                        matrices = arrays["key_mass"]
                    matrix_metadata = step["matrix_metadata"]
                    boundaries = _frame_boundaries(
                        grid=tuple(int(value) for value in record["latent_grid"]),
                        token_count=int(matrix_metadata["token_count"]),
                        bins=int(matrix_metadata["output_bins"]),
                    )
                    _render_contact_sheet(
                        matrices=matrices,
                        output_path=contact,
                        title=(
                            f"{record['model']} | Block {record['block_id']} | "
                            f"denoise step {int(step['step_number_one_based'])}"
                        ),
                        boundaries=boundaries,
                    )
                relative = Path("assets") / model / case / step["directory"] / contact.name
                target = output / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(contact)
                for image_name in step["head_images"]:
                    image_source = step_dir / str(image_name)
                    image_target = target.parent / str(image_name)
                    if image_target.exists() or image_target.is_symlink():
                        image_target.unlink()
                    image_target.symlink_to(image_source)
                case_index = Path("assets") / model / case / "index.html"
                index_target = output / case_index
                index_target.parent.mkdir(parents=True, exist_ok=True)
                if index_target.exists() or index_target.is_symlink():
                    index_target.unlink()
                index_target.symlink_to(summary_path.parent / "index.html")
                step_cards.append(
                    "<figure>"
                    f"<a href='{html.escape(str(case_index))}'>"
                    f"<img loading='lazy' src='{html.escape(str(relative))}'></a>"
                    f"<figcaption>step {int(step['step_number_one_based'])}</figcaption>"
                    "</figure>"
                )
            method_cards.append(
                f"<article data-model='{html.escape(model)}'>"
                f"<h3>{html.escape(MODEL_LABELS[model])}</h3>"
                f"<div class='steps'>{''.join(step_cards)}</div></article>"
            )
        case_sections.append(
            f"<section><h2>{html.escape(case)}</h2>"
            f"<div class='methods'>{''.join(method_cards)}</div></section>"
        )

    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Block 17 self-attention matrices</title>
<style>
:root{{--blue:#2667a8;--green:#22865f;--red:#b34d42}}
body{{margin:0;background:#f3f4f1;color:#20231f;font:14px Arial,sans-serif}}
header{{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #cfd3cc;padding:14px 20px}}
main{{max-width:1900px;margin:auto;padding:0 20px 30px}}
h1,h2,h3{{letter-spacing:0}} section{{border-top:1px solid #c7cbc4;padding-top:14px;margin-top:22px}}
.methods{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}}
article{{background:#fff;border:1px solid #d4d8d1;border-top:4px solid #777;padding:10px;min-width:0}}
article[data-model=wan_lora]{{border-top-color:var(--blue)}} article[data-model=xssc]{{border-top-color:var(--green)}}
article[data-model=physrvg]{{border-top-color:var(--red)}} .steps{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}}
figure{{margin:0;min-width:0}} img{{width:100%;height:auto;display:block;background:#111}} figcaption{{padding-top:4px}}
@media(max-width:1100px){{.methods{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>Block 17 · full-token self-attention matrices</h1>
<div>Steps 5, 15, 25, 35 · all query/key tokens · per-head heatmaps</div></header>
<main>{''.join(case_sections)}</main></body></html>"""
    index_path = output / "index.html"
    index_path.write_text(page, encoding="utf-8")
    print(json.dumps({"index": str(index_path), "records": len(records), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
