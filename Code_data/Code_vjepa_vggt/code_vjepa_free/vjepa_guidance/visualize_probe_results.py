#!/usr/bin/env python3
"""
visualize_probe_results.py

Build a local web visualization for V-JEPA guidance energy persistence experiment results.

Serves an interactive HTML dashboard showing:
- All 3 phases' energy curves (baseline + guided conditions)
- Delta curves (guided - baseline) with y=0 reference
- Persistence scores ranked table
- Best configuration summary

Usage:
  python visualize_probe_results.py \
    --results-dir /data/gaoya/agent-data/outputs/probe_sweep \
    --port 8765
"""
import argparse
import json
import logging
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
import urllib.parse

log = logging.getLogger(__name__)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>V-JEPA Guidance Energy Persistence</title>
<script src="https://cdn.plot.ly/plotly-2.20.0.min.js"></script>
<style>
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    margin: 0;
    padding: 20px;
    background: #f5f5f5;
}}
.container {{
    max-width: 1400px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}}
h1 {{
    color: #1a1a1a;
    margin-bottom: 10px;
}}
.subtitle {{
    color: #666;
    margin-bottom: 30px;
    font-size: 14px;
}}
.phase-section {{
    margin-bottom: 50px;
}}
.phase-title {{
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #2c3e50;
    border-bottom: 2px solid #3498db;
    padding-bottom: 8px;
}}
.best-card {{
    background: #e8f5e9;
    border-left: 4px solid #4caf50;
    padding: 15px;
    margin-bottom: 20px;
    border-radius: 4px;
}}
.best-card h3 {{
    margin: 0 0 8px 0;
    color: #2e7d32;
    font-size: 16px;
}}
.best-card .metric {{
    font-family: monospace;
    font-size: 13px;
    color: #555;
    line-height: 1.6;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
    font-size: 13px;
}}
th {{
    background: #f8f9fa;
    padding: 10px;
    text-align: left;
    border-bottom: 2px solid #dee2e6;
    font-weight: 600;
}}
td {{
    padding: 8px 10px;
    border-bottom: 1px solid #dee2e6;
}}
tr:hover {{
    background: #f8f9fa;
}}
.plot-container {{
    margin: 20px 0;
}}
.legend {{
    font-size: 12px;
    color: #666;
    margin-top: 10px;
    padding: 10px;
    background: #f8f9fa;
    border-radius: 4px;
}}
</style>
</head>
<body>
<div class="container">
<h1>V-JEPA Guidance Energy Persistence Experiment</h1>
<div class="subtitle">Sample: {sample_name}</div>

{phase1_html}

{phase2_html}

{phase3_html}

<div class="legend">
<strong>Metrics:</strong><br>
<strong>persistence_score</strong>: Fraction of post-guidance probe steps where delta &lt; 0 (higher = more durable signal)<br>
<strong>mean_delta</strong>: Average energy difference (guided - baseline) across post-guidance steps (negative = guided is lower)<br>
<strong>Success target</strong>: persistence_score ≥ 0.6 AND mean_delta &lt; -0.002
</div>

</div>
</body>
</html>
"""

PHASE_TEMPLATE = """
<div class="phase-section">
<div class="phase-title">Phase {num}: {title}</div>

<div class="best-card">
<h3>Best Configuration</h3>
<div class="metric">
Label: <strong>{best_label}</strong><br>
Persistence Score: <strong>{best_score:.4f}</strong><br>
Mean Delta: <strong>{best_delta:.6f}</strong><br>
Guidance Steps: {best_percents}<br>
Latent Step Size: {best_ss}<br>
Inner K: {best_k}
</div>
</div>

<table>
<thead>
<tr>
<th>Rank</th>
<th>Label</th>
<th>Persistence</th>
<th>Mean Delta</th>
<th>Step %</th>
<th>Step Size</th>
<th>Inner K</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<div class="plot-container" id="phase{num}_energy"></div>
<div class="plot-container" id="phase{num}_delta"></div>

<script>
{plot_script}
</script>

</div>
"""


def _load_phase_data(phase_dir: Path):
    """Load all records and summary for a phase."""
    summary_path = phase_dir / f"{phase_dir.name}_summary.json"
    if not summary_path.exists():
        return None

    summary = json.loads(summary_path.read_text())

    records = {}
    for f in phase_dir.glob("*_records.json"):
        label = f.stem.replace("_records", "")
        records[label] = json.loads(f.read_text())

    return summary, records


def _build_plotly_traces(records_dict: dict, baseline_key: str = "baseline"):
    """Build Plotly traces for energy curves and delta curves."""
    baseline = records_dict.get(baseline_key, [])
    baseline_map = {r["step"]: r["energy"] for r in baseline if r.get("energy") is not None}

    # Energy traces
    energy_traces = []
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#95a5a6"]

    # Baseline first
    steps_b = [r["step"] for r in baseline if r.get("energy") is not None]
    energies_b = [r["energy"] for r in baseline if r.get("energy") is not None]
    energy_traces.append({
        "x": steps_b,
        "y": energies_b,
        "mode": "lines+markers",
        "name": "baseline",
        "line": {"color": "black", "width": 2, "dash": "dash"},
        "marker": {"size": 4},
    })

    # Guided conditions
    delta_traces = []
    for i, (label, recs) in enumerate(records_dict.items()):
        if label == baseline_key:
            continue

        steps = [r["step"] for r in recs if r.get("energy") is not None]
        energies = [r["energy"] for r in recs if r.get("energy") is not None]

        color = colors[i % len(colors)]
        energy_traces.append({
            "x": steps,
            "y": energies,
            "mode": "lines+markers",
            "name": label,
            "line": {"color": color, "width": 1.5},
            "marker": {"size": 3},
        })

        # Delta trace
        deltas = []
        steps_d = []
        for r in recs:
            if r.get("energy") is not None and r["step"] in baseline_map:
                steps_d.append(r["step"])
                deltas.append(r["energy"] - baseline_map[r["step"]])

        delta_traces.append({
            "x": steps_d,
            "y": deltas,
            "mode": "lines+markers",
            "name": label,
            "line": {"color": color, "width": 1.5},
            "marker": {"size": 3},
        })

    return energy_traces, delta_traces


def _build_phase_html(phase_num: int, phase_dir: Path, title: str) -> str:
    """Build HTML section for one phase."""
    data = _load_phase_data(phase_dir)
    if data is None:
        return f"<div class='phase-section'><div class='phase-title'>Phase {phase_num}: {title}</div><p>No data available.</p></div>"

    summary, records_dict = data
    best = summary.get("best", {})
    ranked = summary.get("ranked", [])

    # Best card
    best_label = best.get("label", "N/A")
    best_score = best.get("persistence_score", 0.0)
    best_delta = best.get("mean_delta_post", 0.0)
    best_percents = best.get("guidance_step_percents", [])
    best_ss = best.get("latent_step_size", "N/A")
    best_k = best.get("inner_k", 1)

    # Table rows
    rows = []
    for rank, item in enumerate(ranked, start=1):
        rows.append(f"""
<tr>
<td>{rank}</td>
<td><strong>{item['label']}</strong></td>
<td>{item['persistence_score']:.4f}</td>
<td>{item['mean_delta_post']:.6f}</td>
<td>{item.get('guidance_step_percents', [])}</td>
<td>{item.get('latent_step_size', 'N/A')}</td>
<td>{item.get('inner_k', 1)}</td>
</tr>
""")

    # Plotly traces
    energy_traces, delta_traces = _build_plotly_traces(records_dict)

    plot_script = f"""
var energy_data = {json.dumps(energy_traces)};
var delta_data = {json.dumps(delta_traces)};

var energy_layout = {{
    title: 'Energy Curves (Raw)',
    xaxis: {{title: 'Denoising step index'}},
    yaxis: {{title: 'V-JEPA Energy'}},
    hovermode: 'x unified',
    legend: {{orientation: 'v', x: 1.02, y: 1}},
    margin: {{l: 60, r: 120, t: 50, b: 50}},
}};

var delta_layout = {{
    title: 'Delta Curves (Guided - Baseline)',
    xaxis: {{title: 'Denoising step index'}},
    yaxis: {{title: 'Delta energy'}},
    hovermode: 'x unified',
    legend: {{orientation: 'v', x: 1.02, y: 1}},
    margin: {{l: 60, r: 120, t: 50, b: 50}},
    shapes: [{{
        type: 'line',
        x0: 0, x1: 1, xref: 'paper',
        y0: 0, y1: 0, yref: 'y',
        line: {{color: 'black', width: 1, dash: 'dot'}}
    }}]
}};

Plotly.newPlot('phase{phase_num}_energy', energy_data, energy_layout, {{responsive: true}});
Plotly.newPlot('phase{phase_num}_delta', delta_data, delta_layout, {{responsive: true}});
"""

    return PHASE_TEMPLATE.format(
        num=phase_num,
        title=title,
        best_label=best_label,
        best_score=best_score,
        best_delta=best_delta,
        best_percents=best_percents,
        best_ss=best_ss,
        best_k=best_k,
        rows="".join(rows),
        plot_script=plot_script,
    )


def build_html(results_dir: Path, sample_name: str) -> str:
    """Build complete HTML page."""
    phase1_html = _build_phase_html(1, results_dir / "phase1", "Timing Sweep")
    phase2_html = _build_phase_html(2, results_dir / "phase2", "Step Size Sweep")
    phase3_html = _build_phase_html(3, results_dir / "phase3", "Step Count + Inner K Sweep")

    return HTML_TEMPLATE.format(
        sample_name=sample_name,
        phase1_html=phase1_html,
        phase2_html=phase2_html,
        phase3_html=phase3_html,
    )


class CustomHandler(SimpleHTTPRequestHandler):
    html_content = None

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(self.html_content.encode("utf-8"))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        log.info("%s - - [%s] %s" % (self.address_string(), self.log_date_time_string(), format % args))


def main():
    parser = argparse.ArgumentParser(description="Visualize V-JEPA probe results")
    parser.add_argument("--results-dir", type=Path, default=Path("/data/gaoya/agent-data/outputs/probe_sweep"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--sample-name", type=str, default="physicIQ_025_Solid_Mechanics_0002")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.results_dir.exists():
        log.error("Results directory not found: %s", args.results_dir)
        return

    log.info("Building HTML from %s ...", args.results_dir)
    html = build_html(args.results_dir, args.sample_name)

    CustomHandler.html_content = html

    server = HTTPServer(("0.0.0.0", args.port), CustomHandler)
    log.info("Serving visualization at http://localhost:%d", args.port)
    log.info("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down server")
        server.shutdown()


if __name__ == "__main__":
    main()
