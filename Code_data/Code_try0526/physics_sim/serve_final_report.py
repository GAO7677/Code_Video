#!/usr/bin/env python3
"""PDI + WMReward JEPA + vjepa_proxy 完整对比报告"""

import csv, json, sys, subprocess
from pathlib import Path

DATA = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/wmreward_report")
DATA.mkdir(parents=True, exist_ok=True)

def fv(v): return f"{v:.4f}" if isinstance(v,(int,float)) else str(v)

# ── Table A: PDI-Bench methods ──
with open("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/result/metrics.csv") as f:
    a_rows = list(csv.DictReader(f))
a_thead = "<tr>" + "".join(f"<th>{k.replace('_',' ')}</th>" for k in a_rows[0]) + "</tr>"
a_tbody = "".join("<tr>" + "".join(f"<td class='n'>{v}</td>" for v in r.values()) + "</tr>" for r in a_rows)

# ── Table B1: Ball-Block 8 scenarios ──
b1_data = []
for name, e, mu, m in [
    ("e09 superball", 0.9, 0.5, 1.0), ("e07 bouncy", 0.7, 0.5, 1.0),
    ("e05 medium", 0.5, 0.5, 1.0), ("e03 plastic", 0.3, 0.5, 1.0),
    ("e07 low-fric", 0.7, 0.1, 1.0), ("e07 high-fric", 0.7, 1.0, 1.0),
    ("e07 light-ball", 0.7, 0.5, 0.1), ("e07 heavy-ball", 0.7, 0.5, 5.0),
]:
    jp = Path(f"/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/ball_block/{name.split()[0]}.json")
    d = json.loads(jp.read_text()) if jp.exists() else {}
    b1_data.append({
        "name": name, "e": e, "mu": mu, "m": m,
        "pdi": fv(d.get("pdi_score")), "wmr": fv(d.get("wmreward_jepa")),
        "proxy": fv(d.get("vjepa_proxy")),
    })

b1_thead = "<tr><th>scenario</th><th>e</th><th>μ</th><th>m</th><th>pdi↓</th><th>wmreward_jepa↑</th><th>vjepa_proxy↑</th></tr>"
b1_tbody = "".join(f"<tr><td>{r['name']}</td><td class='n'>{r['e']}</td><td class='n'>{r['mu']}</td><td class='n'>{r['m']}</td><td class='n'>{r['pdi']}</td><td class='n'>{r['wmr']}</td><td class='n'>{r['proxy']}</td></tr>" for r in b1_data)

# ── Table B2: JEPA sensitivity summary ──
b2_data = [
    ("Velocity (28×)", "0.5-14 m/s", "-", "0.469-0.479", "0.744-0.751"),
    ("Mass (10⁴×)", "0.01-100 kg", "-", "0.474-0.491", "0.729-0.747"),
    ("Gravity (4×)", "4.9-19.6", "-", "0.467-0.478", "0.744-0.754"),
    ("Block mass (40×)", "0.5-20 kg", "-", "0.474-0.482", "0.736-0.750"),
    ("No collision", "-", "-", "0.495", "0.748"),
    ("Reverse", "-", "-", "0.467", "0.726"),
]
b2_thead = "<tr><th>variable</th><th>range</th><th>pdi↓ range</th><th>wmreward_jepa↑ range</th><th>vjepa_proxy↑ range</th></tr>"
b2_tbody = "".join(f"<tr><td>{r[0]}</td><td class='n'>{r[1]}</td><td class='n'>{r[2]}</td><td class='n'>{r[3]}</td><td class='n'>{r[4]}</td></tr>" for r in b2_data)

# ── Table C: Shuffle (20 videos) ──
pairs = [
    ("gt_ball", "GT/Longitudinal_Convergence/ball.json", "ball_block"),
    ("gt_blackswan", "GT/Biological_Motion/blackswan.json", "ball_block"),
    ("gt_bus", "GT/Dynamic_Tracking/bus.json", "ball_block"),
    ("gt_car-turn", "GT/Curved_Motion/car-turn.json", "ball_block"),
    ("gt_rhino", "GT/partial_occlusion/rhino.json", "ball_block"),
    ("gt_planes-water", "GT/Longitudinal_Convergence/planes-water.json", "ball_block"),
    ("gt_bottle", "GT/Longitudinal_Convergence/bottle.json", "ball_block"),
    ("gt_kite-surf", "GT/Biological_Motion/kite-surf.json", "ball_block"),
    ("gt_stroller", "GT/Dynamic_Tracking/stroller.json", "ball_block"),
    ("gt_soccerball", "GT/partial_occlusion/soccerball.json", "ball_block"),
    ("sim_e07_mu05_m1", "ball_block/e07_mu05_m1.json", "ball_block"),
    ("sim_e07_mu05_m01", "ball_block/e07_mu05_m01.json", "ball_block"),
    ("sim_e07_mu10_m1", "ball_block/e07_mu10_m1.json", "ball_block"),
    ("sim_vel_140", "jepa_sensitivity/vel_140.json", "jepa_sensitivity"),
    ("sim_mass_001", "jepa_sensitivity/mass_001.json", "jepa_sensitivity"),
    ("sim_e03_mu05_m1", "ball_block/e03_mu05_m1.json", "ball_block"),
    ("sim_e09_mu05_m1", "ball_block/e09_mu05_m1.json", "ball_block"),
    ("sim_rev_035", "jepa_sensitivity/rev_035.json", "jepa_sensitivity"),
    ("sim_nomiss", "jepa_sensitivity/nomiss.json", "jepa_sensitivity"),
    ("sim_grav_050", "jepa_sensitivity/grav_050.json", "jepa_sensitivity"),
]

PDI_BASE = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output")
SIM_BASE = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos")

# Read PDI from shuffled reports
def read_shuf_pdi(name):
    report = Path(f"/tmp/pdi_shuffle/{name}_shuffled/{name}_shuffled/{name}_shuffled_pdi_report.txt")
    if not report.exists(): return "-"
    import re
    m = re.search(r"FINAL PDI SCORE:\s*([0-9.]+)", report.read_text())
    return fv(float(m.group(1))) if m else "-"

c_rows = ""
for name, orig_rel, sim_dir in pairs:
    # Original values
    if name.startswith("gt_"):
        orig_jp = PDI_BASE / orig_rel
    else:
        orig_jp = SIM_BASE / orig_rel
    orig = json.loads(orig_jp.read_text()) if orig_jp.exists() else {}
    wmr_orig = fv(orig.get("wmreward_jepa"))
    proxy_orig = fv(orig.get("vjepa_proxy"))

    # Shuffled WMR
    shuf_jp = Path(f"/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/shuffle_test/{name}_shuffled.json")
    shuf = json.loads(shuf_jp.read_text()) if shuf_jp.exists() else {}
    wmr_shuf = fv(shuf.get("wmreward_jepa"))
    pdi_shuf = read_shuf_pdi(name)

    # Delta
    dw = float(wmr_shuf) - float(wmr_orig) if wmr_shuf != "-" and wmr_orig != "-" else None
    dw_s = f"{dw:+.3f}" if dw is not None else "-"

    c_rows += f"<tr><td>{name}</td><td class='n'>{pdi_shuf}</td><td class='n'>{wmr_orig}</td><td class='n'>{proxy_orig}</td><td class='n'>{wmr_shuf}</td><td class='n'>{dw_s}</td></tr>"

c_thead = "<tr><th>video</th><th>pdi shuf↓</th><th>wmr orig↑</th><th>proxy orig↑</th><th>wmr shuf↑</th><th>Δ wmr</th></tr>"

# ── Representative cases ──
rep_cases = [
    ("GT ball orig", "WMR=0.446", "pdi_videos/GT/Longitudinal_Convergence/ball.mp4", "green"),
    ("GT ball SHUFFLED", "WMR=0.439 PDI=0.102", "shuffle_videos/gt_ball_shuffled.mp4", "red"),
    ("Sim baseline orig", "PDI=0.022 WMR=0.457", "sim_videos/ball_block/e07_mu05_m1.mp4", "green"),
    ("Sim baseline SHUFFLED", "WMR=0.441 PDI=0.011", "shuffle_videos/sim_e07_mu05_m1_shuffled.mp4", "red"),
]
rep_cards = ""
for label, sub, vid, hl in rep_cases:
    rep_cards += f'<div class="card {hl}"><video src="{vid}" controls muted preload="metadata"></video><div class="info"><div class="label">{label}</div><div class="sub">{sub}</div></div></div>'

html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PhysV Benchmark — PDl vs WMReward JEPA</title>
<style>
:root{{--bg:#1a1815;--panel:#252320;--line:#3d3830;--text:#e8e4dd;--muted:#9d968a;--accent:#e08840;--red:#e05550;--green:#6db87d;--blue:#6ba4d1}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--text);font-family:system-ui,sans-serif;background:var(--bg)}}
.page{{max-width:1400px;margin:0 auto;padding:24px}}h1{{margin:0 0 4px;font-size:24px}}h2{{font-size:16px;margin:28px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--line)}}
.sub{{color:var(--muted);margin:0 0 6px;font-size:13px}}
.finding{{padding:12px 16px;background:rgba(224,136,64,0.1);border:1px solid var(--accent);border-radius:8px;margin:12px 0 20px;font-size:13px;line-height:1.6}}
.finding strong{{color:var(--accent)}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border-radius:8px;overflow:hidden;margin-bottom:20px}}
th,td{{padding:7px 10px;border-bottom:1px solid var(--line);font-size:12px;white-space:nowrap}}
th{{background:rgba(255,255,255,0.05);font-size:10px;text-transform:uppercase}}
.n{{text-align:right;font-variant-numeric:tabular-nums}}td:first-child{{text-align:left;font-weight:600}}
.grid4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px}}
.card{{border-radius:10px;overflow:hidden;background:#1a1818;border:1px solid var(--line)}}
.card video{{width:100%;display:block;aspect-ratio:16/9;object-fit:cover}}
.card.red{{border-color:var(--red)}}.card.green{{border-color:var(--green)}}
.info{{padding:8px 10px}}.label{{font-size:12px;font-weight:700}}.sub{{font-size:11px;color:var(--muted);margin-top:2px}}
@media(max-width:900px){{.grid4{{grid-template-columns:repeat(2,1fr)}}}}
</style></head><body><div class="page">
<h1>PhysV Benchmark — PDI vs WMReward JEPA</h1>
<p class="sub">PDI↓ (geometric consistency) | WMReward JEPA↑ (predictive similarity) | vjepa_proxy↑ (custom composite)</p>

<div class="finding">
<strong>PDI and JEPA ranks inverted.</strong> GT best PDI but worst JEPA. Shuffling breaks PDI but barely moves JEPA. Both metrics capture complementary signals, neither passes temporal sanity check.
</div>

<h2>Group A: PDI-Bench Generated Videos (4 methods × 15)</h2>
<table>{a_thead}{a_tbody}</table>

<h2>Group B1: Simulation — Ball-Block Physics (8 scenarios)</h2>
<table>{b1_thead}{b1_tbody}</table>

<h2>Group B2: Simulation — JEPA Sensitivity (20 scenarios)</h2>
<table>{b2_thead}{b2_tbody}</table>

<h2>Group C: Frame Shuffle Sanity Check (20 videos)</h2>
<div class="finding"><strong>PDI fails:</strong> shuffled PDIs are absurdly low (GT from 0.38→0.01). <strong>WMR barely moves:</strong> avg |Δ|=0.013, total temporal chaos within noise floor.</div>
<table>{c_thead}{c_rows}</table>

<h2>Representative Cases</h2>
<div class="grid4">{rep_cards}</div>
</div></body></html>"""
(DATA / "index.html").write_text(html)

# Symlinks
for name, target in [
    ("sim_videos", Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos")),
    ("pdi_videos", Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/output")),
    ("shuffle_videos", Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp/videos/shuffle_test")),
]:
    link = DATA / name
    if not link.exists(): link.symlink_to(target)

subprocess.run([sys.executable, "-m", "http.server", "18707", "--directory", str(DATA)])
