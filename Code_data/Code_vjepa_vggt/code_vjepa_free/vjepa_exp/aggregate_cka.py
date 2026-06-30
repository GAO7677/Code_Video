"""
Aggregate per-case [9,30] CKA .npy files into a mean matrix.

Usage:
    python aggregate_cka.py \
        --case-dir /data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0613pybullet \
        --out /data/gaoya/agent-data/outputs/phys_compare/cka_pybullet_full/cka_mean.npz
"""
import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    case_dir = Path(args.case_dir)
    files = sorted(case_dir.glob("*.npy"))
    if not files:
        print("No .npy files found.")
        return

    accum = None
    bad = 0
    for f in files:
        try:
            grid = np.load(str(f)).astype(np.float32)
            if grid.shape != (9, 30):
                print(f"  SKIP {f.name}: unexpected shape {grid.shape}")
                bad += 1
                continue
            if accum is None:
                accum = grid.copy()
            else:
                accum += grid
        except Exception as e:
            print(f"  WARN {f.name}: {e}")
            bad += 1

    n = len(files) - bad
    mean = accum / n
    print(f"Aggregated {n} cases (skipped {bad}), shape {mean.shape}")
    print(f"  CKA range: [{mean.min():.4f}, {mean.max():.4f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), mean=mean, count=np.array(n))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
