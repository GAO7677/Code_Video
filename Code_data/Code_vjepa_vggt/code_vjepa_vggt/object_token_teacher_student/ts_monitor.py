#!/usr/bin/env python3
"""Training health monitor + best-checkpoint picker for the teacher-student pipeline.

Usage:
  python3 ts_monitor.py health  <train_log>          # parse latest step/loss, scan for NaN/errors
  python3 ts_monitor.py best    <ckpt_dir> <log>     # pick lowest-EMA-loss step that has a saved ckpt
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

STEP_RE = re.compile(r"(\d+)/\d+ \[[^\]]*?loss=(nan|inf|-inf|[0-9.eE+-]+)(?:, pmax=([0-9.eE+-]+))?")
ERR_RE = re.compile(r"Traceback|RuntimeError|OutOfMemory|ChildFailedError|non-finite|CUDA error", re.IGNORECASE)
BENIGN = re.compile(r"FutureWarning|deprecated|elastic/error|errors\.html|expandable_segments", re.IGNORECASE)


def _to_float(s: str):
    try:
        return float(s)
    except ValueError:
        return float("nan")


def parse_steps(log: Path):
    pts = []
    if not log.is_file():
        return pts
    text = log.read_text(errors="ignore")
    for m in STEP_RE.finditer(text):
        step = int(m.group(1))
        loss = _to_float(m.group(2))
        pmax = float(m.group(3)) if m.group(3) else None
        pts.append((step, loss, pmax))
    return pts


def health(log_path: str):
    log = Path(log_path)
    pts = parse_steps(log)
    text = log.read_text(errors="ignore") if log.is_file() else ""
    errs = [ln for ln in text.splitlines() if ERR_RE.search(ln) and not BENIGN.search(ln)]
    if not pts:
        print(f"[health] no step lines yet in {log.name}; errors={len(errs)}")
        if errs:
            print("  last err:", errs[-1][:200])
        return
    last_step, last_loss, last_pmax = pts[-1]
    window = [l for _, l, _ in pts[-50:]]
    ema = window[0]
    for l in window[1:]:
        ema = 0.9 * ema + 0.1 * l
    finite = all(l == l and abs(l) != float("inf") for _, l, _ in pts[-20:])
    print(f"[health] {log.name}: step={last_step} loss={last_loss:.4f} pmax={last_pmax} "
          f"ema50={ema:.4f} finite_last20={finite} err_lines={len(errs)}")
    if errs:
        print("  last err:", errs[-1][:200])


def best(ckpt_dir: str, log_path: str):
    pts = parse_steps(Path(log_path))
    saved = {}
    for p in Path(ckpt_dir).glob("step_*.pt"):
        m = re.search(r"step_(\d+)\.pt", p.name)
        if m:
            saved[int(m.group(1))] = p
    if not saved:
        print(f"[best] no checkpoints in {ckpt_dir}")
        return
    # EMA loss per step
    ema = None
    ema_at = {}
    for step, loss, _ in pts:
        ema = loss if ema is None else 0.9 * ema + 0.1 * loss
        ema_at[step] = ema
    # choose saved step with min EMA (nearest recorded step <= ckpt step)
    rec_steps = sorted(ema_at)
    best_step, best_val = None, None
    for s in sorted(saved):
        near = [r for r in rec_steps if r <= s]
        val = ema_at[near[-1]] if near else None
        if val is not None and (best_val is None or val < best_val):
            best_val, best_step = val, s
    if best_step is None:
        best_step = max(saved)
        print(f"[best] no loss match; falling back to latest: {saved[best_step]}")
    else:
        print(f"[best] step={best_step} ema_loss={best_val:.4f} -> {saved[best_step]}")
    print(str(saved[best_step]))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "health":
        health(sys.argv[2])
    elif cmd == "best":
        best(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
        sys.exit(1)
