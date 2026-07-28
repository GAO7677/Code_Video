#!/usr/bin/env python3
"""Build an HTML gallery for common-S score-extreme all-token QK maps."""

from __future__ import annotations

import argparse
import csv
import html
import os
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")
MODEL_NAMES = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--heatmap-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=851)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ranking = args.ranking.expanduser().resolve()
    heatmaps = args.heatmap_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "figures"
    if assets.is_symlink() and assets.resolve() != heatmaps:
        assets.unlink()
    if not assets.exists():
        assets.symlink_to(heatmaps, target_is_directory=True)
    elif not assets.is_symlink():
        raise RuntimeError(f"refusing to replace non-symlink path: {assets}")

    with ranking.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    sections = []
    for group, title in (("top", "score_S 前 10"), ("bottom", "score_S 后 10")):
        table_rows = []
        for row in (item for item in rows if item["score_group"] == group):
            rank = int(row["group_rank"])
            block = int(row["block"])
            head = int(row["head"])
            role = f"S_{group}{rank:02d}"
            cells = []
            for model in MODELS:
                figure = (
                    heatmaps
                    / model
                    / next(heatmaps.joinpath(model).iterdir()).name
                    / f"{role}_block{block:02d}_head{head:02d}.png"
                )
                if figure.is_file():
                    relative = Path("figures") / figure.relative_to(heatmaps)
                    cells.append(
                        f"<td><a href='{html.escape(relative.as_posix())}'>"
                        f"<img loading='lazy' src='{html.escape(relative.as_posix())}'></a>"
                        f"<div>{MODEL_NAMES[model]} score "
                        f"{float(row[f'{model}_score_S']):.4f}</div></td>"
                    )
                else:
                    cells.append(f"<td class='missing'>{MODEL_NAMES[model]}：等待生成</td>")
            table_rows.append(
                "<tr>"
                f"<th>#{rank}<br>B{block:02d} H{head:02d}<br>"
                f"mean {float(row['mean_score_S']):.4f}</th>"
                + "".join(cells)
                + "</tr>"
            )
        sections.append(
            f"<section><h2>{title}</h2><table><thead><tr><th>公共 S-head</th>"
            + "".join(f"<th>{MODEL_NAMES[model]}</th>" for model in MODELS)
            + "</tr></thead><tbody>"
            + "".join(table_rows)
            + "</tbody></table></section>"
        )

    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Common S score extremes Q@K</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;background:#111416;color:#f4f5f6;font:14px/1.45 system-ui,sans-serif;letter-spacing:0}}
header,main{{padding:16px 20px}}header{{border-bottom:1px solid #3b4147}}h1,h2,p{{margin:0 0 9px}}h1{{font-size:22px}}h2{{font-size:18px;margin-top:18px}}
.note{{color:#b9c0c7}}table{{width:100%;border-collapse:collapse;table-layout:fixed}}th,td{{border:1px solid #3b4147;padding:7px;vertical-align:top}}
th{{background:#202529}}th:first-child{{width:110px}}img{{display:block;width:100%;height:auto;background:#080a0b}}
.missing{{color:#9ba2a8;text-align:center;vertical-align:middle}}a{{color:inherit}}
</style></head><body>
<header><h1>公共 S-head：score_S 前 10 / 后 10 的 all-token Q@K</h1>
<p>Seed {args.seed}；去噪步 5、15、25、35；5,824 个 query/key tokens 的精确 softmax，显示时池化为 512×512。</p>
<p class="note">每张图左列为 raw QK/√d，右列为 log10 softmax attention mass；白线标记 13 个 latent 时间段。</p></header>
<main>{''.join(sections)}</main></body></html>"""
    temporary = output / f".index.html.{os.getpid()}.tmp"
    temporary.write_text(document, encoding="utf-8")
    temporary.replace(output / "index.html")
    print(output / "index.html")


if __name__ == "__main__":
    main()
