from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a local portal for V2V VBench/VBench2 batch results.")
    parser.add_argument("--summary-json", type=Path, required=True, help="Summary JSON produced by batch_v2v_vbench_portal.py")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory where index.html is written.")
    return parser.parse_args()


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>V2V VBench Portal</title>
  <style>
    :root {
      --bg: #efe7d7;
      --panel: rgba(255, 252, 245, 0.92);
      --panel-strong: #fffdf7;
      --ink: #231b13;
      --muted: #6f655a;
      --line: #d7cbb8;
      --accent: #a14c24;
      --accent-soft: rgba(161, 76, 36, 0.12);
      --ok: #2f7d58;
      --warn: #a06a10;
      --bad: #9a2f2f;
      --shadow: 0 16px 32px rgba(35, 27, 19, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(161, 76, 36, 0.18), transparent 24%),
        radial-gradient(circle at left 20%, rgba(83, 120, 91, 0.12), transparent 18%),
        linear-gradient(180deg, #f6f1e6 0%, var(--bg) 100%);
    }
    main {
      width: min(100vw - 32px, 1880px);
      margin: 18px auto 32px;
      display: grid;
      gap: 18px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      box-shadow: var(--shadow);
    }
    .hero {
      padding: 24px;
      display: grid;
      gap: 14px;
    }
    .hero h1 {
      margin: 0;
      font-size: clamp(28px, 4vw, 54px);
      line-height: 0.92;
      letter-spacing: -0.03em;
    }
    .hero-copy {
      color: var(--muted);
      line-height: 1.55;
      max-width: 1050px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .stat {
      background: var(--panel-strong);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
    }
    .stat .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat .value {
      margin-top: 6px;
      font-size: 26px;
      font-weight: 700;
      line-height: 1;
    }
    .chips, .dim-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .chip, .dim-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
      font-size: 13px;
    }
    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: var(--muted);
      flex: 0 0 auto;
    }
    .ok .dot { background: var(--ok); }
    .error .dot { background: var(--bad); }
    .pending .dot { background: var(--warn); }
    .layout {
      display: grid;
      grid-template-columns: minmax(360px, 420px) minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .panel {
      padding: 16px;
    }
    .panel h2 {
      margin: 0 0 12px;
      font-size: 18px;
    }
    .preview video {
      width: 100%;
      border-radius: 16px;
      background: #000;
      display: block;
    }
    .preview-meta {
      margin-top: 12px;
      display: grid;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
    }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    .links a {
      color: var(--accent);
      text-decoration: none;
      font-weight: 600;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(240px, 1fr) 220px 220px;
      gap: 10px;
      margin-bottom: 12px;
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel-strong);
      color: var(--ink);
      padding: 10px 12px;
      font-size: 14px;
    }
    .table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: var(--panel-strong);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1700px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 13px;
    }
    th {
      position: sticky;
      top: 0;
      background: #f8f3ea;
      z-index: 1;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
    }
    tr[data-active="true"] {
      background: rgba(161, 76, 36, 0.08);
    }
    tr:hover {
      background: rgba(161, 76, 36, 0.05);
      cursor: pointer;
    }
    td.num {
      font-variant-numeric: tabular-nums;
      text-align: right;
      white-space: nowrap;
    }
    .status-note {
      font-size: 13px;
      color: var(--muted);
      line-height: 1.5;
    }
    .dataset-box {
      display: grid;
      gap: 10px;
      padding: 12px;
      border-radius: 16px;
      background: var(--panel-strong);
      border: 1px solid var(--line);
    }
    .dataset-score {
      font-size: 28px;
      font-weight: 700;
    }
    .empty {
      color: var(--muted);
      padding: 20px;
    }
    code {
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
      font-size: 12px;
      background: rgba(35, 27, 19, 0.05);
      border-radius: 6px;
      padding: 2px 5px;
    }
    @media (max-width: 1250px) {
      .layout { grid-template-columns: 1fr; }
      .controls { grid-template-columns: 1fr; }
      table { min-width: 1200px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div>
        <h1>V2V VBench Portal</h1>
        <div class="hero-copy" id="hero-copy">Loading summary...</div>
      </div>
      <div class="stats" id="stats"></div>
      <div>
        <h2>Dimension Status</h2>
        <div class="dim-grid" id="dim-grid"></div>
      </div>
    </section>

    <section class="layout">
      <aside class="panel preview">
        <h2>Preview</h2>
        <video id="preview-video" controls playsinline preload="metadata"></video>
        <div class="preview-meta" id="preview-meta"></div>
        <div class="links" id="preview-links"></div>
      </aside>

      <section class="panel">
        <h2>Records</h2>
        <div class="controls">
          <input id="search" type="search" placeholder="Search method, group, or path">
          <select id="group-filter"></select>
          <select id="status-filter">
            <option value="all">All rows</option>
            <option value="error">Rows with errors</option>
            <option value="complete">Rows with all evaluated fields</option>
          </select>
        </div>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Group</th>
                <th>Method</th>
                <th>Case</th>
                <th>WM Sim</th>
                <th>WM Surprise</th>
                <th>PhysicsIQ Ctx</th>
                <th>PhysicsIQ NoCtx</th>
                <th>PMF Ctx</th>
                <th>PMF NoCtx</th>
                <th>VideoPhy2</th>
                <th>Cosmos R1</th>
                <th>VBench Subject</th>
                <th>VBench Bg</th>
                <th>VBench Smooth</th>
                <th>VBench Dynamic</th>
                <th>VBench Aesthetic</th>
                <th>VBench Imaging</th>
                <th>VBench2 Anatomy</th>
                <th>VBench2 Identity</th>
                <th>VBench2 Clothes</th>
                <th>VBench2 MultiView</th>
                <th>Errors</th>
              </tr>
            </thead>
            <tbody id="records-body"></tbody>
          </table>
        </div>
      </section>
    </section>

    <section class="panel">
      <h2>Dataset-Level Metrics</h2>
      <div id="dataset-metrics"></div>
    </section>
  </main>

  <script>
    const SUMMARY_PATH = "summary.json";

    function fmt(value, digits = 4) {
      if (value === null || value === undefined || Number.isNaN(value)) return "–";
      if (typeof value !== "number") return String(value);
      return value.toFixed(digits);
    }

    function statusClass(status) {
      if (status === "ok") return "ok";
      if (status === "error") return "error";
      return "pending";
    }

    function scoreOrDash(obj) {
      if (!obj || typeof obj !== "object") return "–";
      if (typeof obj.score === "number") return fmt(obj.score);
      return "–";
    }

    function recordHasErrors(record) {
      return Array.isArray(record.errors) && record.errors.length > 0;
    }

    function recordComplete(record) {
      const vb = record.vbench || {};
      const vb2 = record.vbench2 || {};
      const required = [
        "subject_consistency",
        "background_consistency",
        "motion_smoothness",
        "dynamic_degree",
        "aesthetic_quality",
        "imaging_quality",
      ];
      const required2 = [
        "Human_Anatomy",
        "Human_Identity",
        "Human_Clothes",
        "Multi-View_Consistency",
      ];
      return required.every((dim) => vb[dim] && vb[dim].status === "ok")
        && required2.every((dim) => vb2[dim] && vb2[dim].status === "ok");
    }

    function buildCell(value, cls = "") {
      return `<td class="${cls}">${value}</td>`;
    }

    async function loadSummary() {
      const response = await fetch(SUMMARY_PATH, { cache: "no-store" });
      if (!response.ok) throw new Error(`Failed to load ${SUMMARY_PATH}: ${response.status}`);
      return response.json();
    }

    function populateHeader(summary) {
      document.getElementById("hero-copy").textContent =
        `Root: ${summary.root} | Records: ${summary.records.length} | Last update: ${summary.updated_at || "pending"} | ` +
        `VBench2 Diversity is rendered as a dataset-level score because the official custom-input implementation evaluates prompt groups rather than per-video rows.`;

      const stats = [
        ["Records", summary.records.length],
        ["Groups", new Set(summary.records.map((r) => r.group)).size],
        ["Methods", new Set(summary.records.map((r) => r.method)).size],
        ["Row Errors", summary.records.filter(recordHasErrors).length],
      ];
      const statHtml = stats.map(([label, value]) => `
        <div class="stat">
          <div class="label">${label}</div>
          <div class="value">${value}</div>
        </div>
      `).join("");
      document.getElementById("stats").innerHTML = statHtml;

      const dims = [];
      for (const [dim, item] of Object.entries(summary.dimension_runs.vbench || {})) {
        dims.push({ label: `VBench ${dim}`, status: item.status, score: item.score });
      }
      for (const [dim, item] of Object.entries(summary.dimension_runs.vbench2 || {})) {
        dims.push({ label: `VBench2 ${dim}`, status: item.status, score: item.score });
      }
      const dimHtml = dims.map((item) => `
        <div class="dim-chip ${statusClass(item.status)}">
          <span class="dot"></span>
          <span>${item.label}</span>
          <span>${item.status === "ok" && typeof item.score === "number" ? fmt(item.score) : item.status}</span>
        </div>
      `).join("");
      document.getElementById("dim-grid").innerHTML = dimHtml;
    }

    function populateDatasetMetrics(summary) {
      const box = document.getElementById("dataset-metrics");
      const diversity = summary.dataset_metrics?.vbench2_diversity;
      if (!diversity) {
        box.innerHTML = `<div class="empty">No dataset-level metric available yet.</div>`;
        return;
      }
      box.innerHTML = `
        <div class="dataset-box ${statusClass(diversity.status)}">
          <div><strong>VBench2 Diversity</strong></div>
          <div class="dataset-score">${typeof diversity.score === "number" ? fmt(diversity.score) : "–"}</div>
          <div class="status-note">${diversity.note || ""}</div>
          <div class="status-note">Status: <code>${diversity.status}</code></div>
          ${diversity.result_json ? `<div class="status-note">Result JSON: <code>${diversity.result_json}</code></div>` : ""}
          ${diversity.error ? `<div class="status-note">Error: <code>${diversity.error}</code></div>` : ""}
        </div>
      `;
    }

    function setPreview(record, rowElement) {
      document.getElementById("preview-video").src = record.video_rel;
      const meta = [
        `<strong>${record.case_rel}</strong>`,
        `group=${record.group}`,
        `method=${record.method}`,
        `caption=${record.caption || "–"}`,
      ];
      if (recordHasErrors(record)) {
        meta.push(`errors=${record.errors.join(" | ")}`);
      }
      document.getElementById("preview-meta").innerHTML = meta.map((item) => `<div>${item}</div>`).join("");
      document.getElementById("preview-links").innerHTML = `
        <a href="${record.video_rel}" target="_blank" rel="noopener">Open video</a>
        <a href="${record.json_rel}" target="_blank" rel="noopener">Open source JSON</a>
      `;
      document.querySelectorAll("tbody tr").forEach((row) => row.dataset.active = "false");
      if (rowElement) rowElement.dataset.active = "true";
    }

    function renderRows(summary) {
      const body = document.getElementById("records-body");
      const search = document.getElementById("search").value.trim().toLowerCase();
      const groupFilter = document.getElementById("group-filter").value;
      const statusFilter = document.getElementById("status-filter").value;

      const rows = summary.records.filter((record) => {
        if (groupFilter !== "all" && record.group !== groupFilter) return false;
        if (statusFilter === "error" && !recordHasErrors(record)) return false;
        if (statusFilter === "complete" && !recordComplete(record)) return false;
        if (!search) return true;
        const haystack = [record.group, record.method, record.case_rel, record.caption || ""].join(" ").toLowerCase();
        return haystack.includes(search);
      });

      if (!rows.length) {
        body.innerHTML = `<tr><td class="empty" colspan="22">No rows matched the current filter.</td></tr>`;
        return;
      }

      body.innerHTML = "";
      rows.forEach((record, index) => {
        const tr = document.createElement("tr");
        tr.innerHTML = [
          buildCell(record.group),
          buildCell(record.method),
          buildCell(record.case_rel),
          buildCell(fmt(record.existing_metrics.wmreward_similarity), "num"),
          buildCell(fmt(record.existing_metrics.wmreward_surprise), "num"),
          buildCell(fmt(record.existing_metrics.physics_iq_with_context, 2), "num"),
          buildCell(fmt(record.existing_metrics.physics_iq_without_context, 2), "num"),
          buildCell(fmt(record.existing_metrics.pmf_with_context, 4), "num"),
          buildCell(fmt(record.existing_metrics.pmf_without_context, 4), "num"),
          buildCell(fmt(record.existing_metrics.videophy2, 2), "num"),
          buildCell(fmt(record.existing_metrics.cosmos_reason1, 2), "num"),
          buildCell(scoreOrDash(record.vbench.subject_consistency), "num"),
          buildCell(scoreOrDash(record.vbench.background_consistency), "num"),
          buildCell(scoreOrDash(record.vbench.motion_smoothness), "num"),
          buildCell(scoreOrDash(record.vbench.dynamic_degree), "num"),
          buildCell(scoreOrDash(record.vbench.aesthetic_quality), "num"),
          buildCell(scoreOrDash(record.vbench.imaging_quality), "num"),
          buildCell(scoreOrDash(record.vbench2.Human_Anatomy), "num"),
          buildCell(scoreOrDash(record.vbench2.Human_Identity), "num"),
          buildCell(scoreOrDash(record.vbench2.Human_Clothes), "num"),
          buildCell(scoreOrDash(record.vbench2["Multi-View_Consistency"]), "num"),
          buildCell(record.errors.length ? record.errors.join(" | ") : "–"),
        ].join("");
        tr.addEventListener("click", () => setPreview(record, tr));
        body.appendChild(tr);
        if (index === 0) setPreview(record, tr);
      });
    }

    function populateGroupFilter(summary) {
      const select = document.getElementById("group-filter");
      const groups = ["all", ...Array.from(new Set(summary.records.map((record) => record.group))).sort()];
      select.innerHTML = groups.map((group) => {
        const label = group === "all" ? "All groups" : group;
        return `<option value="${group}">${label}</option>`;
      }).join("");
    }

    async function main() {
      try {
        const summary = await loadSummary();
        populateHeader(summary);
        populateDatasetMetrics(summary);
        populateGroupFilter(summary);
        const rerender = () => renderRows(summary);
        document.getElementById("search").addEventListener("input", rerender);
        document.getElementById("group-filter").addEventListener("change", rerender);
        document.getElementById("status-filter").addEventListener("change", rerender);
        renderRows(summary);
      } catch (error) {
        document.getElementById("hero-copy").textContent = String(error);
      }
    }

    main();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target_summary = args.out_dir / "summary.json"
    if args.summary_json.resolve() != target_summary.resolve():
        target_summary.write_text(args.summary_json.read_text(encoding="utf-8"), encoding="utf-8")
    (args.out_dir / "index.html").write_text(build_html(), encoding="utf-8")


if __name__ == "__main__":
    main()
