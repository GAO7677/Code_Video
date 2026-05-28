#!/usr/bin/env python3
"""WMReward JEPA 完整对比 — PDI-Bench方法 + 仿真视频"""

import json, os, csv, sys, subprocess
from pathlib import Path

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
PDI_OUTPUT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output")
PDI_METRICS = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/result/metrics.csv")
REPORT_DIR = DATA_DIR / "wmreward_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def fv(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)

# ---- Section 1: PDI-Bench method-level comparison ----
method_stats = {}
for jp in PDI_OUTPUT.rglob("*.json"):
    d = json.loads(jp.read_text())
    wmr = d.get("wmreward_jepa", {}).get("similarity")
    if wmr is None: continue
    method = jp.parent.parent.name
    if method not in method_stats:
        method_stats[method] = []
    method_stats[method].append(wmr)

pdi_stats = {}
with open(PDI_METRICS) as f:
    for row in csv.DictReader(f):
        pdi_stats[row["provider"]] = float(row["mean_pdi_score"])

# Method comparison table
method_rows = ""
for m in ["GT", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08", "wan22-5B-TI2V"]:
    scores = method_stats.get(m, [])
    mean_w = sum(scores)/len(scores) if scores else 0
    mean_p = pdi_stats.get(m, 0)
    method_rows += f"<tr><td>{m}</td><td class='n'>{len(scores)}</td><td class='n pdi'>{mean_p:.4f}</td><td class='n jepa'>{mean_w:.4f}</td><td class='n'>{min(scores):.4f}</td><td class='n'>{max(scores):.4f}</td></tr>"

# ---- Section 2: Per-case detail ----
case_rows = ""
for method in ["GT", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08", "wan22-5B-TI2V"]:
    mdir = PDI_OUTPUT / method
    if not mdir.exists(): continue
    for cat_dir in sorted(mdir.iterdir()):
        if not cat_dir.is_dir(): continue
        for vp in sorted(cat_dir.glob("*.mp4")):
            jp = vp.with_suffix(".json")
            d = json.loads(jp.read_text()) if jp.exists() else {}
            wmr = d.get("wmreward_jepa", {}).get("similarity")
            rel = str(vp.relative_to(PDI_OUTPUT.parent))
            cat = cat_dir.name
            case = vp.stem
            case_rows += f"<tr><td>{method}</td><td>{cat}</td><td>{case}</td><td class='n'>{fv(wmr)}</td></tr>"

# ---- Section 3: Simulation videos summary ----
sim_dirs = [
    ("ball_block", DATA_DIR / "videos" / "ball_block"),
    ("jepa_sensitivity", DATA_DIR / "videos" / "jepa_sensitivity"),
]
sim_rows = ""
for dname, vdir in sim_dirs:
    if not vdir.exists(): continue
    scores = []
    for jp in sorted(vdir.glob("*.json")):
        d = json.loads(jp.read_text())
        wmr = d.get("wmreward_jepa", {}).get("similarity")
        if wmr: scores.append(wmr)
    if scores:
        sim_rows += f"<tr><td>{dname}</td><td class='n'>{len(scores)}</td><td class='n jepa'>{sum(scores)/len(scores):.4f}</td><td class='n'>{min(scores):.4f}</td><td class='n'>{max(scores):.4f}</td></tr>"

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>WMReward JEPA — Full Comparison</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d;--blue:#6ba4d1}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1400px;margin:0 auto;padding:24px}}h1{{margin:0 0 4px}}h2{{font-size:16px;margin:24px 0 8px}}.sub{{color:var(--muted);margin:0 0 16px;font-size:14px}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:10px;overflow:hidden;margin-bottom:16px}}
th,td{{padding:8px 12px;border-bottom:1px solid var(--line);font-size:13px}}
th{{background:rgba(255,255,255,0.05);font-size:11px;text-transform:uppercase;white-space:nowrap}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}.pdi{{color:var(--blue)}}.jepa{{color:var(--accent)}}
.finding{{padding:14px 18px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:10px;margin:0 0 20px;font-size:13px;line-height:1.7}}
.finding strong{{color:var(--accent)}}
.finding .inv{{color:var(--red)}}
.summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:24px}}
.stat{{border:1px solid var(--line);border-radius:10px;padding:14px;text-align:center;background:var(--panel)}}
.stat .val{{font-size:28px;font-weight:700}}.stat .lbl{{font-size:11px;color:var(--muted);margin-top:2px}}
.cor{{font-size:12px;color:var(--muted);margin-top:4px}}
</style></head><body><div class="page">
<h1>WMReward JEPA — Full Evaluation</h1>
<p class="sub">PDI-Bench videos (GT + 3 methods × 15 cases) &amp; simulation videos (ball_block + jepa_sensitivity)</p>

<div class="finding">
<strong class="inv">Key finding: PDI and JEPA rankings are INVERTED for real/gen videos.</strong><br>
GT (real) → <strong>best PDI (0.144)</strong> but <strong>worst JEPA (0.412)</strong> — real videos are geometrically accurate but visually unpredictable.<br>
Wan (generated) → <strong>worst PDI (0.878)</strong> but <strong>best JEPA (0.435)</strong> — generated videos are geometrically flawed but visually bland/predictable.<br>
Simulation → JEPA ~0.47 (highest similarity, most predictable) — synthetic rigid-body motion is trivial for V-JEPA2.<br>
<strong>PDI measures geometric correctness. JEPA measures visual predictability. They are complementary, not redundant.</strong>
</div>

<div class="summary">
  <div class="stat"><div class="val" style="color:var(--accent)">88</div><div class="lbl">videos evaluated (WMReward)</div></div>
  <div class="stat"><div class="val" style="color:var(--red)">Inverted</div><div class="lbl">PDI vs JEPA ranking</div><div class="cor">GT: PDI best, JEPA worst</div></div>
  <div class="stat"><div class="val" style="color:var(--green)">0.38~0.47</div><div class="lbl">WMReward range across all</div><div class="cor">Real < Gen < Sim</div></div>
</div>

<h2>PDI-Bench Methods — WMReward JEPA vs PDI</h2>
<table><thead><tr><th>Method</th><th>N</th><th class="pdi">mean PDI↓</th><th class="jepa">mean WMR↑</th><th>min WMR</th><th>max WMR</th></tr></thead>
<tbody>{method_rows}</tbody></table>

<h2>Simulation Videos — WMReward Summary</h2>
<table><thead><tr><th>Dataset</th><th>N</th><th class="jepa">mean WMR↑</th><th>min</th><th>max</th></tr></thead>
<tbody>{sim_rows}</tbody></table>

<h2>Per-Case Details (PDI-Bench)</h2>
<table><thead><tr><th>Method</th><th>Category</th><th>Case</th><th>WMR Sim↑</th></tr></thead>
<tbody>{case_rows}</tbody></table>
</div></body></html>"""
(REPORT_DIR / "index.html").write_text(html)

link = REPORT_DIR / "videos"
if not link.exists():
    link.symlink_to(DATA_DIR / "videos")

subprocess.run([sys.executable, "-m", "http.server", "18706", "--directory", str(REPORT_DIR)])
