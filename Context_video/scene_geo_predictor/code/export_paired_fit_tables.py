"""Export compact CSV tables from the bounded paired-fit JSON report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write(path, rows):
    if not rows:
        Path(path).write_text("")
        return
    fields = sorted({k for row in rows for k in row})
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = json.loads(args.report.read_text())
    case_rows, pair_rows, resample_case_rows, resample_pair_rows, grad_rows = [], [], [], [], []
    for arm, result in report["results"].items():
        for row in result["base_eval"]["case_rows"]:
            case_rows.append({"arm": arm, "seed": "base", **row})
        for row in result["base_eval"]["pairs"]:
            pair_rows.append({"arm": arm, "seed": "base", **row})
        for step, diag in result["diagnostics"].items():
            for module in sorted(diag["gradient_norm"]):
                grad_rows.append({"arm": arm, "step": int(step), "module": module,
                                  "gradient_norm": diag["gradient_norm"][module],
                                  "update_norm": diag["update_norm"][module]})
        for seed, evaluation in result["resampling"].items():
            for row in evaluation["case_rows"]:
                resample_case_rows.append({"arm": arm, "seed": seed, **row,
                                           "prediction_vs_base_mean_l2_m": evaluation["prediction_vs_base_mean_l2_m"],
                                           "prediction_vs_base_max_l2_m": evaluation["prediction_vs_base_max_l2_m"]})
            for row in evaluation["pairs"]:
                resample_pair_rows.append({"arm": arm, "seed": seed, **row})
    write(args.output / "paired_fit_base_case_metrics.csv", case_rows)
    write(args.output / "paired_fit_base_pair_metrics.csv", pair_rows)
    write(args.output / "paired_fit_resample_case_metrics.csv", resample_case_rows)
    write(args.output / "paired_fit_resample_pair_metrics.csv", resample_pair_rows)
    write(args.output / "paired_fit_gradient_update_metrics.csv", grad_rows)
    print(json.dumps({"status": "EXECUTED", "files": sorted(p.name for p in args.output.glob("*.csv"))}, indent=2))


if __name__ == "__main__":
    main()
