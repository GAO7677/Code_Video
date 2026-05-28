#!/usr/bin/env python3
"""对 ball_block 视频跑 PDI + JEPA 评分，回填 JSON，可视化

用法: conda run -n wan python eval_ball_block.py [--skip-pdi] [--skip-jepa] [--port 18703]
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
VIDEO_DIR = DATA_DIR / "videos" / "ball_block"
PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
TMP_DIR = DATA_DIR / "tmp_eval"
REPORT_DIR = DATA_DIR / "eval_report"


def run_pdi(video_path: Path, output_dir: Path) -> dict | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem
    report_path = output_dir / f"{stem}_pdi_report.txt"

    if not report_path.exists():
        # Write a wrapper that patches flash_attn before importing PDI
        wrapper = TMP_DIR / "_pdi_wrapper.py"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "import sys, os\n"
            "for mod in ['flash_attn.bert_padding', 'flash_attn.flash_attn_interface', 'flash_attn_2_cuda']:\n"
            "    if mod not in sys.modules:\n"
            "        m = type(sys)(mod)\n"
            "        sys.modules[mod] = m\n"
            "        if 'bert_padding' in mod:\n"
            "            def _fail(*a,**k): raise ImportError('flash_attn not available')\n"
            "            m.index_first_axis = m.pad_input = m.unpad_input = _fail\n"
            "sys.modules.setdefault('flash_attn', type(sys)('flash_attn'))\n"
            "sys.modules['flash_attn'].__path__ = []\n"
            "sys.argv = [sys.argv[0], '--input', sys.argv[1], '--text', sys.argv[2], '--output_dir', sys.argv[3]]\n"
            "import runpy\n"
            "runpy.run_path('evaluation/main.py', run_name='__main__')\n"
        )
        cmd = [
            sys.executable, "-u", str(wrapper),
            str(video_path), "ball", str(output_dir),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PDI_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PDI_FLORENCE_MODEL_ID"] = "/data/gaoya/ckpt/microsoft-Florence-2-base"
        r = subprocess.run(cmd, cwd=PDI_ROOT, env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    FAILED: {r.stderr[-200:]}")
            return None

    text = report_path.read_text()

    def ex(pat, cast=None):
        m = re.search(pat, text)
        return cast(m.group(1)) if (m and cast) else (m.group(1) if m else None)

    return {
        "pdi_score": ex(r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": ex(r"OVERALL GRADE:\s*([A-Z+-]+)"),
        "scale_error": ex(r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_error": ex(r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "rigidity_error": ex(r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "vp_error": ex(r"VP Component .*?:\s*([0-9.]+)", float),
    }


def run_jepa(video_path: Path) -> dict | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rerank_video"))
    from scorers import JEPAPredictiveScorer
    from schemas import JEPAScoreConfig
    from video_utils import load_video_frames, uniform_subsample_frames, ensure_dir
    import cv2, numpy as np

    frames = load_video_frames(video_path)
    total = len(frames)
    if total < 30:
        return None
    split = min(60, total // 2)
    ctx = uniform_subsample_frames(frames[:split], 8)
    fut = uniform_subsample_frames(frames[split:], 16)

    tmp = ensure_dir(TMP_DIR / "jepa" / video_path.stem)
    ctx_p = tmp / "context.mp4"
    fut_p = tmp / "future.mp4"

    def wv(p, frs):
        h, w = frs[0].shape[:2]
        out = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 16, (w, h))
        for f in frs: out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
    wv(ctx_p, ctx)
    wv(fut_p, fut)

    scorer = JEPAPredictiveScorer(JEPAScoreConfig(
        backend="vjepa2", device="cuda", max_frames=32,
        context_frames=8, future_frames=16, context_repeat_frames=8, crop_size=384,
        vjepa_checkpoint=Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt"),
        vjepa_repo_root=Path("/home/gaoya/Code_Video/vjepa2-main"),
        vjepa_model_name="vjepa2_1_vit_large_384",
    ))
    score, details = scorer.score(context_video_path=ctx_p, candidate_video_path=fut_p)
    return {"jepa_score": float(score)}


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

        if not args.skip_pdi:
            print("  PDI...", end=" ", flush=True)
            pdi = run_pdi(vp, TMP_DIR / "pdi" / name)
            if pdi:
                meta["pdi"] = pdi
                print(f"PDI={pdi['pdi_score']:.4f} grade={pdi['grade']}")
            else:
                print("FAILED")

        if not args.skip_jepa:
            print("  JEPA...", end=" ", flush=True)
            try:
                jepa = run_jepa(vp)
                if jepa:
                    meta["jepa"] = jepa
                    print(f"JEPA={jepa['jepa_score']:.4f}")
                else:
                    print("FAILED")
            except Exception as e:
                print(f"FAILED: {e}")

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
