#!/usr/bin/env python3
"""Compare xSSC decoder static/dynamic memory sensitivity on 0526dp cases."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from analyze_xssc_0526dp_slot_sensitivity import (  # noqa: E402
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT,
    MODEL_SPECS,
    build_model,
    cosine_distance_curve,
    cosine_rdm,
    descriptor,
    discover_cases,
    group_definitions,
    l2_normalize,
    linear_cka,
    load_embedding,
    safe_key,
    set_seed,
)


DEFAULT_OUTPUT = DEFAULT_OUTPUT / "decoder_static_dynamic"
PARTITIONS = ("static", "dynamic")
OBJECTS = ("ball", "block")
STRICT_EXCLUDED = {
    "jepa_sensitivity/nomiss",
    "jepa_sensitivity/rev_035",
}
MODEL_COLORS = {
    "official_dinov2": "#2563eb",
    "dinov3_step036000": "#dc2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--parent-output",
        type=Path,
        default=DEFAULT_OUTPUT.parent,
        help="Directory produced by analyze_xssc_0526dp_slot_sensitivity.py",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--stage", choices=("project", "report", "all"), default="all")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    required = [args.data_root, args.parent_output / "canonical_boxes.json"]
    for model_name, spec in MODEL_SPECS.items():
        required.extend(
            [
                Path(spec["config"]),
                Path(spec["checkpoint"]),
                args.parent_output / "embeddings" / model_name,
            ]
        )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing inputs:\n" + "\n".join(missing))


def project_slots(
    decoder,
    slots: np.ndarray,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    tensor = torch.from_numpy(slots).to(device)
    time_steps, num_slots, slot_dim = tensor.shape
    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=amp_dtype, enabled=device.type == "cuda"
    ):
        projected = decoder.project2(tensor.reshape(time_steps, num_slots, slot_dim))
    projected = projected.detach().float().cpu().numpy()
    decoder_dim = int(projected.shape[-1])
    dynamic_dim = int(decoder_dim * float(decoder.rd))
    static_dim = decoder_dim - dynamic_dim
    return (
        projected[..., :static_dim].astype(np.float32),
        projected[..., static_dim:].astype(np.float32),
        {
            "slot_dim": slot_dim,
            "decoder_dim": decoder_dim,
            "static_dim": static_dim,
            "dynamic_dim": dynamic_dim,
            "dynamic_ratio": float(decoder.rd),
        },
    )


def project_all(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    device: torch.device,
) -> dict[str, Any]:
    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[args.amp_dtype]
    dimensions = {}
    for model_name, spec in MODEL_SPECS.items():
        print(f"[model] loading {model_name}", flush=True)
        model = build_model(spec, device)
        decoder = model.m.decode
        for position, case in enumerate(cases, start=1):
            output = (
                args.output_dir
                / "features"
                / model_name
                / f"{safe_key(case['key'])}.npz"
            )
            if output.is_file() and not args.force:
                print(
                    f"[{model_name}] {position}/{len(cases)} cached {case['key']}",
                    flush=True,
                )
                continue
            item = load_embedding(args.parent_output, model_name, case["key"])
            static, dynamic, dims = project_slots(
                decoder, item["slotz"], device, amp_dtype
            )
            dimensions[model_name] = dims
            output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output,
                static=static,
                dynamic=dynamic,
                semantic_slots=item["semantic_slots"],
                frame_indices=item["frame_indices"],
            )
            print(
                f"[{model_name}] {position}/{len(cases)} {case['key']} "
                f"static={static.shape} dynamic={dynamic.shape}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if dimensions:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "dimensions.json").write_text(
            json.dumps(dimensions, indent=2) + "\n"
        )
    elif (args.output_dir / "dimensions.json").is_file():
        dimensions = json.loads((args.output_dir / "dimensions.json").read_text())
    return dimensions


def load_projected(
    output_dir: Path, model_name: str, key: str
) -> dict[str, np.ndarray]:
    path = output_dir / "features" / model_name / f"{safe_key(key)}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as item:
        return {name: item[name] for name in item.files}


def adjacent_similarity(features: np.ndarray) -> np.ndarray:
    normalized = l2_normalize(features)
    return np.sum(normalized[:-1] * normalized[1:], axis=-1)


def projected_descriptor(item: dict[str, np.ndarray], partition: str) -> np.ndarray:
    ids = item["semantic_slots"].astype(int)
    features = item[partition]
    return np.concatenate(
        [l2_normalize(features[:, slot_id]).mean(axis=0) for slot_id in ids]
    )


def nearest_group(
    key: str, groups: list[dict[str, Any]]
) -> dict[str, Any] | None:
    for group in groups:
        if key in group["cases"]:
            return group
    return None


def save_case_plot(
    key: str,
    baseline: str,
    curves: dict[str, dict[str, dict[str, np.ndarray]]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=True, sharey="row")
    for row, partition in enumerate(PARTITIONS):
        for column, object_name in enumerate(OBJECTS):
            axis = axes[row, column]
            for model_name in MODEL_SPECS:
                axis.plot(
                    1.0
                    - curves[model_name][partition][f"{object_name}_distance"],
                    color=MODEL_COLORS[model_name],
                    linewidth=1.5,
                    label=MODEL_SPECS[model_name]["label"],
                )
            axis.set_title(f"{partition.capitalize()} {object_name}")
            axis.set_ylabel("Cosine similarity to group baseline")
            axis.set_ylim(-0.02, 1.02)
            axis.grid(alpha=0.22)
    axes[1, 0].set_xlabel("Frame")
    axes[1, 1].set_xlabel("Frame")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"{key} vs {baseline}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_summary_bars(
    family_summary: list[dict[str, Any]], output: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    labels = [
        "DINOv2\nstatic",
        "DINOv2\ndynamic",
        "DINOv3\nstatic",
        "DINOv3\ndynamic",
    ]
    for axis, family in zip(axes, ("physics", "appearance")):
        rows = [row for row in family_summary if row["family"] == family]
        values_ball = []
        values_block = []
        for model_name in MODEL_SPECS:
            for partition in PARTITIONS:
                row = next(
                    item
                    for item in rows
                    if item["model"] == model_name
                    and item["partition"] == partition
                )
                values_ball.append(row["ball_distance"])
                values_block.append(row["block_distance"])
        x = np.arange(len(labels))
        width = 0.36
        axis.bar(x - width / 2, values_ball, width, label="ball", color="#ef4444")
        axis.bar(x + width / 2, values_block, width, label="block", color="#3b82f6")
        axis.set_xticks(x, labels)
        axis.set_title(f"{family.capitalize()} changes")
        axis.set_ylabel("Cosine distance to baseline")
        axis.grid(axis="y", alpha=0.22)
    axes[0].legend()
    fig.suptitle("Decoder partition sensitivity (higher distance = stronger response)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_rdm_grid(
    keys: list[str],
    rdms: dict[str, dict[str, np.ndarray]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 12))
    vmax = max(
        float(matrix.max())
        for model_rdms in rdms.values()
        for matrix in model_rdms.values()
    )
    labels = [Path(key).name for key in keys]
    for row, partition in enumerate(PARTITIONS):
        for column, model_name in enumerate(MODEL_SPECS):
            axis = axes[row, column]
            image = axis.imshow(
                rdms[model_name][partition],
                cmap="magma",
                vmin=0,
                vmax=max(vmax, 1e-6),
            )
            axis.set_title(
                f"{MODEL_SPECS[model_name]['label']}\n{partition} memory"
            )
            axis.set_xticks(range(len(keys)), labels, rotation=90, fontsize=5)
            axis.set_yticks(range(len(keys)), labels, fontsize=5)
            fig.colorbar(image, ax=axis, fraction=0.046)
    fig.suptitle("Decoder static/dynamic representational distance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_family(
    rows: list[dict[str, Any]], groups: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    group_family = {
        group["name"]: group.get("family", "physics") for group in groups
    }
    output = []
    for family in ("physics", "appearance"):
        for model_name in MODEL_SPECS:
            for partition in PARTITIONS:
                selected = [
                    row
                    for row in rows
                    if row["model"] == model_name
                    and row["partition"] == partition
                    and group_family.get(row["group"]) == family
                    and row["case"] != row["baseline"]
                ]
                output.append(
                    {
                        "family": family,
                        "model": model_name,
                        "partition": partition,
                        "n": len(selected),
                        "ball_distance": float(
                            np.mean([row["ball_distance"] for row in selected])
                        ),
                        "block_distance": float(
                            np.mean([row["block_distance"] for row in selected])
                        ),
                        "ball_similarity": float(
                            np.mean([row["ball_similarity"] for row in selected])
                        ),
                        "block_similarity": float(
                            np.mean([row["block_similarity"] for row in selected])
                        ),
                        "ball_adjacent_similarity": float(
                            np.mean(
                                [row["ball_adjacent_similarity"] for row in selected]
                            )
                        ),
                        "block_adjacent_similarity": float(
                            np.mean(
                                [row["block_adjacent_similarity"] for row in selected]
                            )
                        ),
                    }
                )
    return output


def compute_report(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    dimensions: dict[str, Any],
) -> dict[str, Any]:
    case_by_key = {case["key"]: case for case in cases}
    groups = group_definitions(cases)
    rows = []
    case_payload = {}
    for case in cases:
        key = case["key"]
        group = nearest_group(key, groups)
        baseline_key = group["baseline"] if group else key
        curves = {}
        for model_name in MODEL_SPECS:
            current = load_projected(args.output_dir, model_name, key)
            baseline = load_projected(args.output_dir, model_name, baseline_key)
            current_ids = current["semantic_slots"].astype(int)
            baseline_ids = baseline["semantic_slots"].astype(int)
            curves[model_name] = {}
            for partition in PARTITIONS:
                partition_curves = {}
                for object_id, object_name in enumerate(OBJECTS):
                    distance = cosine_distance_curve(
                        baseline[partition][:, baseline_ids[object_id]],
                        current[partition][:, current_ids[object_id]],
                    )
                    adjacent = adjacent_similarity(
                        current[partition][:, current_ids[object_id]]
                    )
                    partition_curves[f"{object_name}_distance"] = distance
                    rows.append(
                        {
                            "case": key,
                            "group": group["name"] if group else "ungrouped",
                            "baseline": baseline_key,
                            "model": model_name,
                            "partition": partition,
                            "object": object_name,
                            "distance": float(distance.mean()),
                            "similarity": float(1.0 - distance.mean()),
                            "adjacent_similarity": float(adjacent.mean()),
                        }
                    )
                curves[model_name][partition] = partition_curves
        chart = args.output_dir / "assets/cases" / f"{safe_key(key)}.png"
        save_case_plot(key, baseline_key, curves, chart)
        case_payload[key] = {
            "group": group["name"] if group else "ungrouped",
            "baseline": baseline_key,
            "chart": chart.relative_to(args.output_dir).as_posix(),
            "source_video": (
                Path("..") / "media" / f"{safe_key(key)}.mp4"
            ).as_posix(),
        }

    wide_rows = []
    index = {}
    for row in rows:
        key = (
            row["case"],
            row["group"],
            row["baseline"],
            row["model"],
            row["partition"],
        )
        if key not in index:
            index[key] = {
                "case": row["case"],
                "group": row["group"],
                "baseline": row["baseline"],
                "model": row["model"],
                "partition": row["partition"],
            }
            wide_rows.append(index[key])
        prefix = row["object"]
        index[key][f"{prefix}_distance"] = row["distance"]
        index[key][f"{prefix}_similarity"] = row["similarity"]
        index[key][f"{prefix}_adjacent_similarity"] = row[
            "adjacent_similarity"
        ]

    family_summary = aggregate_family(wide_rows, groups)
    strict_keys = sorted(set(case_by_key) - STRICT_EXCLUDED)
    features = {}
    rdms = {}
    observations = {}
    for model_name in MODEL_SPECS:
        features[model_name] = {}
        rdms[model_name] = {}
        observations[model_name] = {}
        for partition in PARTITIONS:
            features[model_name][partition] = np.stack(
                [
                    projected_descriptor(
                        load_projected(args.output_dir, model_name, key),
                        partition,
                    )
                    for key in strict_keys
                ]
            )
            rdms[model_name][partition] = cosine_rdm(
                features[model_name][partition]
            )
            observation_rows = []
            for key in strict_keys:
                item = load_projected(args.output_dir, model_name, key)
                for slot_id in item["semantic_slots"].astype(int):
                    observation_rows.append(
                        l2_normalize(item[partition][:, slot_id])
                    )
            observations[model_name][partition] = np.concatenate(
                observation_rows, axis=0
            )

    cross_model = {}
    upper = np.triu_indices(len(strict_keys), k=1)
    for partition in PARTITIONS:
        cross_model[partition] = {
            "rdm_spearman": float(
                spearmanr(
                    rdms["official_dinov2"][partition][upper],
                    rdms["dinov3_step036000"][partition][upper],
                ).statistic
            ),
            "linear_cka": linear_cka(
                observations["official_dinov2"][partition],
                observations["dinov3_step036000"][partition],
            ),
        }

    save_summary_bars(
        family_summary, args.output_dir / "assets/summary_sensitivity.png"
    )
    save_rdm_grid(strict_keys, rdms, args.output_dir / "assets/summary_rdm.png")
    write_csv(args.output_dir / "metrics.csv", wide_rows)
    write_csv(args.output_dir / "family_summary.csv", family_summary)
    summary = {
        "case_count": len(cases),
        "strict_case_count": len(strict_keys),
        "strict_excluded_cases": sorted(STRICT_EXCLUDED & set(case_by_key)),
        "dimensions": dimensions,
        "method": (
            "Each cached slotz is passed through the checkpoint's actual "
            "decoder.project2. The leading (1-rd) channels are static memory "
            "and the trailing rd channels are dynamic memory. Similarity is "
            "measured before the transformer decoder mixes the partitions."
        ),
        "cross_model": cross_model,
        "family_summary": family_summary,
        "cases": case_payload,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_html(args.output_dir, summary)
    write_readme(args.output_dir, summary)
    add_parent_link(args.parent_output)
    return summary


def write_html(output_dir: Path, summary: dict[str, Any]) -> None:
    cases_json = json.dumps(summary["cases"], ensure_ascii=True)
    first_case = next(iter(summary["cases"]))
    options = "\n".join(
        f"<option value='{html.escape(key)}'>{html.escape(key)}</option>"
        for key in summary["cases"]
    )
    dimensions = summary["dimensions"]
    family_rows = "\n".join(
        "<tr>"
        f"<td>{row['family']}</td><td>{row['model']}</td>"
        f"<td>{row['partition']}</td>"
        f"<td>{row['ball_similarity']:.5f}</td>"
        f"<td>{row['block_similarity']:.5f}</td>"
        f"<td>{row['ball_adjacent_similarity']:.5f}</td>"
        f"<td>{row['block_adjacent_similarity']:.5f}</td>"
        "</tr>"
        for row in summary["family_summary"]
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC decoder static/dynamic comparison</title>
<style>
:root{{--bg:#f5f7fa;--ink:#17202a;--muted:#65717e;--line:#ccd3db}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
header{{background:#17202a;color:#fff;padding:18px 24px}}header h1{{font-size:22px;margin:0 0 5px}}header p{{margin:0;color:#cbd5e1}}
main{{max-width:1450px;margin:auto;padding:20px 24px 40px}}h2{{font-size:17px;margin:25px 0 10px}}
.metrics{{display:grid;grid-template-columns:repeat(4,minmax(160px,1fr));border:1px solid var(--line);background:#fff}}
.metric{{padding:14px;border-right:1px solid var(--line)}}.metric:last-child{{border:0}}.metric b{{display:block;font-size:21px}}.metric span,.note{{color:var(--muted)}}
select{{width:100%;padding:8px;border:1px solid #98a3af;background:#fff}}video{{width:100%;max-height:500px;background:#111;margin-top:12px}}
img{{display:block;width:100%;border:1px solid var(--line);background:#fff}}table{{width:100%;border-collapse:collapse;background:#fff;font-variant-numeric:tabular-nums}}
th,td{{padding:7px 9px;border:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
a{{color:#2563eb}}@media(max-width:800px){{.metrics{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<header><h1>Decoder static/dynamic similarity</h1><p>Actual project2 memory split before transformer mixing. <a href="../" style="color:#93c5fd">Back to slot report</a></p></header>
<main>
<div class="metrics">
 <div class="metric"><span>Official split</span><b>{dimensions['official_dinov2']['static_dim']} + {dimensions['official_dinov2']['dynamic_dim']}</b></div>
 <div class="metric"><span>DINOv3 split</span><b>{dimensions['dinov3_step036000']['static_dim']} + {dimensions['dinov3_step036000']['dynamic_dim']}</b></div>
 <div class="metric"><span>Static RDM rho / CKA</span><b>{summary['cross_model']['static']['rdm_spearman']:.3f} / {summary['cross_model']['static']['linear_cka']:.3f}</b></div>
 <div class="metric"><span>Dynamic RDM rho / CKA</span><b>{summary['cross_model']['dynamic']['rdm_spearman']:.3f} / {summary['cross_model']['dynamic']['linear_cka']:.3f}</b></div>
</div>
<p class="note">Similarity closer to 1 means more invariant. Distance in the summary plot is 1-similarity, so a larger bar means a stronger response to parameter changes.</p>
<h2>Aggregate sensitivity</h2><img src="assets/summary_sensitivity.png" alt="decoder sensitivity bars">
<h2>Case inspection</h2><select id="case-select">{options}</select><p id="case-note" class="note"></p>
<video id="source-video" controls preload="metadata"></video><img id="case-chart" alt="case similarity curves">
<h2>Representational distance</h2><img src="assets/summary_rdm.png" alt="static and dynamic RDM">
<h2>Mean similarities</h2>
<table><thead><tr><th>Family</th><th>Model</th><th>Partition</th><th>Ball baseline</th><th>Block baseline</th><th>Ball adjacent</th><th>Block adjacent</th></tr></thead><tbody>{family_rows}</tbody></table>
</main><script>
const cases={cases_json};const select=document.getElementById("case-select");
const video=document.getElementById("source-video");const chart=document.getElementById("case-chart");const note=document.getElementById("case-note");
function show(key){{const item=cases[key];video.src=item.source_video;chart.src=item.chart;note.textContent=`Group: ${{item.group}} | baseline: ${{item.baseline}}`;}}
select.value={json.dumps(first_case)};select.addEventListener("change",()=>show(select.value));show(select.value);
</script></body></html>"""
    (output_dir / "index.html").write_text(page)


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    text = f"""# xSSC decoder static/dynamic comparison

This report reads the cached xSSC `slotz` tensors and applies each checkpoint's
actual `decoder.project2`. It then splits decoder memory at `rd=0.25`:

- Official DINOv2: {summary['dimensions']['official_dinov2']['static_dim']} static
  + {summary['dimensions']['official_dinov2']['dynamic_dim']} dynamic channels.
- DINOv3: {summary['dimensions']['dinov3_step036000']['static_dim']} static
  + {summary['dimensions']['dinov3_step036000']['dynamic_dim']} dynamic channels.

The split is analyzed before the transformer decoder mixes the channels.
`metrics.csv` contains per-case similarities to group baselines and adjacent
frame similarities. `family_summary.csv` contains physics/appearance means.
"""
    (output_dir / "README.md").write_text(text)


def add_parent_link(parent_output: Path) -> None:
    path = parent_output / "index.html"
    if not path.is_file():
        return
    marker = "<!-- DECODER_PARTITION_LINK -->"
    if marker in path.read_text():
        return
    text = path.read_text()
    insertion = (
        marker
        + "<p style='margin:12px 0 0'><a href='decoder_static_dynamic/' "
        + "style='color:#93c5fd'>Open decoder static/dynamic comparison</a></p>"
    )
    text = text.replace("</header>", insertion + "</header>", 1)
    path.write_text(text)


def main() -> None:
    args = parse_args()
    validate(args)
    set_seed(args.seed)
    cases = discover_cases(args.data_root, args.max_cases)
    device = torch.device(args.device)
    dimensions = {}
    if args.stage in ("project", "all"):
        dimensions = project_all(args, cases, device)
    if args.stage in ("report", "all"):
        if not dimensions:
            dimensions = json.loads(
                (args.output_dir / "dimensions.json").read_text()
            )
        summary = compute_report(args, cases, dimensions)
        print(
            "[report] "
            f"static rho={summary['cross_model']['static']['rdm_spearman']:.5f} "
            f"dynamic rho={summary['cross_model']['dynamic']['rdm_spearman']:.5f}",
            flush=True,
        )
        print(f"[report] {args.output_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
