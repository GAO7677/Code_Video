from __future__ import annotations

"""
Batch metric backfill wrapper for baseline result folders listed in baseline.txt.

This script delegates the actual metric computation/backfill to the canonical
AAAinfer bench.py, while handling a txt file that lists one result folder per line.

Examples:

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py \
  --metric wmreward

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/bench.py \
  --metric physics_iq \
  --baseline-list /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/baseline.txt
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
DEFAULT_BASELINE_LIST = SCRIPT_DIR / "baseline.txt"
AAA_BENCH = PROJECT_ROOT / "code_vjepa_vggt" / "AAAinfer" / "bench.py"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Read result folders from baseline.txt and delegate one metric at a time "
            "to code_vjepa_vggt/AAAinfer/bench.py for metric backfill."
        )
    )
    parser.add_argument("--metric", required=True, help="Metric name accepted by AAAinfer/bench.py.")
    parser.add_argument("--baseline-list", type=Path, default=DEFAULT_BASELINE_LIST)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_known_args()


def read_result_roots(list_path: Path) -> list[Path]:
    result_roots: list[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            result_roots.append(Path(line).expanduser().resolve())
    return result_roots


def main() -> None:
    args, passthrough = parse_args()

    baseline_list = args.baseline_list.expanduser().resolve()
    if not baseline_list.is_file():
        raise FileNotFoundError(f"baseline list not found: {baseline_list}")
    if not AAA_BENCH.is_file():
        raise FileNotFoundError(f"delegate bench.py not found: {AAA_BENCH}")

    result_roots = read_result_roots(baseline_list)
    if args.limit is not None:
        result_roots = result_roots[: max(0, int(args.limit))]

    if not result_roots:
        raise RuntimeError(f"no result roots found in {baseline_list}")

    env = os.environ.copy()
    pythonpath_entries = [str(PROJECT_ROOT), str(TRY0526_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    print(f"[baseline-bench] python={args.python_bin}")
    print(f"[baseline-bench] metric={args.metric}")
    print(f"[baseline-bench] baseline_list={baseline_list}")
    print(f"[baseline-bench] num_result_roots={len(result_roots)}")

    num_success = 0
    num_failed = 0
    failures: list[tuple[Path, int]] = []

    for index, result_root in enumerate(result_roots, start=1):
        if not result_root.exists():
            print(f"[baseline-bench] skip missing ({index}/{len(result_roots)}): {result_root}")
            num_failed += 1
            failures.append((result_root, 404))
            if args.stop_on_error:
                raise SystemExit(404)
            continue

        cmd = [
            str(args.python_bin),
            str(AAA_BENCH),
            "--metric",
            str(args.metric),
            "--result-root",
            str(result_root),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        if args.dry_run:
            cmd.append("--dry-run")
        cmd.extend(passthrough)

        print(f"[baseline-bench] start ({index}/{len(result_roots)}): {result_root}")
        completed = subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT))
        if completed.returncode == 0:
            num_success += 1
            print(f"[baseline-bench] done  ({index}/{len(result_roots)}): {result_root}")
            continue

        num_failed += 1
        failures.append((result_root, int(completed.returncode)))
        print(
            f"[baseline-bench] fail  ({index}/{len(result_roots)}): "
            f"{result_root} returncode={completed.returncode}"
        )
        if args.stop_on_error:
            raise SystemExit(int(completed.returncode))

    print(
        f"[baseline-bench] summary metric={args.metric} "
        f"success={num_success} failed={num_failed} total={len(result_roots)}"
    )
    if failures:
        for result_root, returncode in failures:
            print(f"[baseline-bench] failure result_root={result_root} returncode={returncode}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
