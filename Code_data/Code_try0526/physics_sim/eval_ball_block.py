#!/usr/bin/env python3
"""Batch report for ball_block videos.

Single-case scoring now lives in `physv_eval.single_case.ball_block`.
This script keeps only the dataset loop, metadata persistence, HTML report
generation, and optional local HTTP serving.

用法: conda run -n wan python eval_ball_block.py [--skip-pdi] [--skip-jepa] [--port 18703]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from physv_eval.single_case.ball_block import score_case as score_ball_block_case

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = DATA_DIR / "videos" / "ball_block"
REPORT_DIR = DATA_DIR / "eval_report"


def gen_html(results: list[dict]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        n = r["name"]
        p = r.get("pdi") or {}
        j = r.get("jepa") or {}
        a = r.get("parameters") or {}
        def fv(v): return f"{v:.4f}" if isinstance(v, (int,float)) else str(v)
        rows.append(f"<tr><td><a href='../videos/ball_block/{n}.mp4'>{n}</a></td>"
            f"<td>{a.get('restitution','-')}</td><td>{a.get('lateral_friction','-')}</td><td>{a.get('ball_mass_kg','-')}</td>"
            f"<td class='n'>{fv(p.get('pdi_score','-'))}</td>"
            f"<td class='n'>{fv(p.get('scale_error','-'))}</td>"
            f"<td class='n'>{fv(p.get('traj_error','-'))}</td>"
            f"<td class='n'>{fv(p.get('rigidity_error','-'))}</td>"
            f"<td>{p.get('grade','-')}</td>"
            f"<td class='n'>{fv(j.get('jepa_score','-'))}</td></tr>")

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Ball-Block Eval</title><style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1200px;margin:0 auto;padding:24px}}h1{{margin:0 0 6px}}.sub{{color:var(--muted);margin:0 0 20px;font-size:14px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:12px;overflow:hidden}}
th,td{{padding:10px 14px;border-bottom:1px solid var(--line)}}th{{background:rgba(255,255,255,0.05);font-size:12px;text-transform:uppercase}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}a{{color:var(--accent);text-decoration:none}}a:hover{{text-decoration:underline}}
</style></head><body><div class="page">
<h1>Ball-Block Physics — Evaluation</h1>
<p class="sub">PDI-Bench ↓ (geometric consistency, lower=better) &nbsp;|&nbsp; V-JEPA2 ↑ (predictive plausibility, higher=better)</p>
<table><thead><tr>
<th>Scenario</th><th>e</th><th>μ</th><th>m</th>
<th>PDI↓</th><th>Scale↓</th><th>Traj↓</th><th>Rigid↓</th><th>Grade</th>
<th>JEPA↑</th>
</tr></thead><tbody>{''.join(rows)}</tbody></table>
</div></body></html>"""
    p = REPORT_DIR / "index.html"
    p.write_text(html)
    return p


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pdi", action="store_true")
    ap.add_argument("--skip-jepa", action="store_true")
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--port", type=int, default=18703)
    args = ap.parse_args()
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    videos = sorted(VIDEO_DIR.glob("*.mp4"))
    print(f"{len(videos)} videos\n")

    results = []
    for vp in videos:
        name = vp.stem
        jp = VIDEO_DIR / f"{name}.json"
        meta = json.loads(jp.read_text()) if jp.exists() else {}
        print(f"[{name}]")

        if not args.skip_pdi or not args.skip_jepa:
            result = score_ball_block_case(
                vp,
                caption="ball",
                skip_pdi=args.skip_pdi,
                skip_jepa=args.skip_jepa,
            )
            if not args.skip_pdi:
                print("  PDI...", end=" ", flush=True)
                pdi = result.get("pdi")
                if pdi:
                    meta["pdi"] = pdi
                    print(f"PDI={pdi['pdi_score']:.4f} grade={pdi['grade']}")
                else:
                    print("FAILED")
            if not args.skip_jepa:
                print("  JEPA...", end=" ", flush=True)
                jepa = result.get("jepa")
                if jepa:
                    meta["jepa"] = jepa
                    print(f"JEPA={jepa['jepa_score']:.4f}")
                else:
                    print("FAILED")

        meta["name"] = name
        jp.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        results.append(meta)

    hp = gen_html(results)
    # Symlink videos for serving
    link = REPORT_DIR / "videos"
    if not link.exists():
        os.symlink(VIDEO_DIR.parent, link)
    print(f"\nReport: {hp}")
    print(f"http://127.0.0.1:{args.port}/index.html")
    subprocess.run([sys.executable, "-m", "http.server", str(args.port), "--directory", str(REPORT_DIR)])


if __name__ == "__main__":
    main()
