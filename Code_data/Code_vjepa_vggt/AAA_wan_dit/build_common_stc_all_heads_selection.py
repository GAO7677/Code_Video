#!/usr/bin/env python3
"""Select every cross-model common stable S/T/C head for one QK replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")
ROLES = ("S", "T", "C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=851)
    return parser.parse_args()


def load_case(input_list: Path) -> tuple[str, str]:
    lines = [
        line.strip()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one input case, found {len(lines)}")
    path = Path(lines[0]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.stem, str(path)


def main() -> None:
    args = parse_args()
    report = args.report.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case, input_json = load_case(input_list)

    rows_by_head: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    with report.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows_by_head[(int(row["block"]), int(row["head"]))][row["model"]] = row

    common: dict[str, list[dict]] = {role: [] for role in ROLES}
    for (block, head), model_rows in sorted(rows_by_head.items()):
        if set(model_rows) != set(MODELS):
            continue
        roles = {model_rows[model]["role"] for model in MODELS}
        if len(roles) != 1:
            continue
        role = roles.pop()
        if role not in ROLES:
            continue
        model_values = {}
        for model in MODELS:
            row = model_rows[model]
            model_values[model] = {
                "score": float(row[f"score_{role}"]),
                "margin": float(row["margin"]),
                "support": float(row["support"]),
                "support_ci95_low": float(row["support_ci95_low"]),
                "support_ci95_high": float(row["support_ci95_high"]),
            }
        common[role].append(
            {
                "role": role,
                "block": block,
                "head": head,
                "models": model_values,
                "cross_model_mean_score": sum(
                    item["score"] for item in model_values.values()
                )
                / len(MODELS),
            }
        )

    samples = {}
    for model in MODELS:
        role_map = {}
        for role in ROLES:
            for rank, item in enumerate(common[role], start=1):
                label = f"{role}_{rank:03d}_B{item['block']:02d}H{item['head']:02d}"
                role_map[label] = {
                    "role": role,
                    "block": item["block"],
                    "head": item["head"],
                    "aggregate_score": item["models"][model]["score"],
                    "cross_model_mean_score": item["cross_model_mean_score"],
                    "margin": item["models"][model]["margin"],
                    "support": item["models"][model]["support"],
                    "representative_seed": args.seed,
                }
        samples[model] = {case: {"roles": role_map}}

    selection = {
        "schema_version": 1,
        "source": str(report),
        "policy": (
            "All heads whose aggregate role is identically S, T, or C in "
            "wan_lora, xssc, and physrvg; the same (block, head) list is replayed "
            "once per model on one fixed case and seed."
        ),
        "representative_seed": args.seed,
        "case": case,
        "input_json": input_json,
        "role_counts": {role: len(common[role]) for role in ROLES},
        "samples": samples,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "role",
        "role_index",
        "block",
        "head",
        "cross_model_mean_score",
    ]
    for model in MODELS:
        fieldnames.extend(
            (
                f"{model}_score",
                f"{model}_margin",
                f"{model}_support",
                f"{model}_support_ci95_low",
                f"{model}_support_ci95_high",
            )
        )
    with (output / "heads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for role in ROLES:
            for role_index, item in enumerate(common[role], start=1):
                row = {
                    "role": role,
                    "role_index": role_index,
                    "block": item["block"],
                    "head": item["head"],
                    "cross_model_mean_score": item["cross_model_mean_score"],
                }
                for model in MODELS:
                    for field, value in item["models"][model].items():
                        row[f"{model}_{field}"] = value
                writer.writerow(row)

    print(
        "[common-stc-selection] "
        + " ".join(f"{role}={len(common[role])}" for role in ROLES)
    )
    print(output / "selection.json")
    print(output / "heads.csv")


if __name__ == "__main__":
    main()
