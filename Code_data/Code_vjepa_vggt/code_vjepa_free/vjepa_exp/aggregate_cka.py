"""
Aggregate per-case [9,30] or [9,30,T] CKA .npy files into a mean matrix.

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
    files = sorted(f for f in case_dir.glob("*.npy") if f.name != "timesteps.npy")
    if not files:
        print("No .npy files found.")
        return

    accum = None
    ref_shape = None
    bad = 0
    for f in files:
        try:
            grid = np.load(str(f)).astype(np.float32)
            if grid.ndim == 2:
                grid = grid[:, :, np.newaxis]   # legacy [9,30] → [9,30,1]
            if grid.shape[:2] != (9, 30):
                print(f"  SKIP {f.name}: unexpected shape {grid.shape}")
                bad += 1
                continue
            if ref_shape is None:
                ref_shape = grid.shape
            if grid.shape != ref_shape:
                print(f"  SKIP {f.name}: shape {grid.shape} != ref {ref_shape}")
                bad += 1
                continue
            if accum is None:
                accum = grid.copy()
            else:
                accum += grid
        except Exception as e:
            print(f"  WARN {f.name}: {e}")
            bad += 1

    if accum is None:
        print("No valid files to aggregate.")
        return

    n = len(files) - bad
    mean = accum / n
    print(f"Aggregated {n} cases (skipped {bad}), shape {mean.shape}")
    print(f"  CKA range: [{mean.min():.4f}, {mean.max():.4f}]")

    # load timesteps if available
    ts_file = case_dir / "timesteps.npy"
    timesteps = np.load(str(ts_file)) if ts_file.exists() else np.array([500])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out), mean=mean, count=np.array(n), timesteps=timesteps)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
