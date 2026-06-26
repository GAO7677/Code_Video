from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean


def load_summaries(root: Path) -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(root.rglob("summary.json"))]


def aggregate(summaries: list[dict]) -> dict:
    by_layer: dict[str, dict[str, list[float]]] = {}
    for item in summaries:
        for layer_name, layer_stats in item["layers"].items():
            bucket = by_layer.setdefault(
                layer_name,
                {
                    "motion_saliency_mean": [],
                    "adjacent_affinity_mean": [],
                    "token_std": [],
                },
            )
            for key in bucket:
                bucket[key].append(float(layer_stats[key]))

    out = {"num_videos": len(summaries), "layers": {}}
    for layer_name, metrics in by_layer.items():
        out["layers"][layer_name] = {key: mean(vals) for key, vals in metrics.items() if vals}
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate extracted V-JEPA layer summaries.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = load_summaries(args.input_root.expanduser().resolve())
    report = aggregate(summaries)
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(str(args.output_json))


if __name__ == "__main__":
    main()
