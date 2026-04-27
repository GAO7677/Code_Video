#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchlib.manifest import BenchSample, load_manifest
from benchlib.staging import safe_stem


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>PhysV Benchmark Dashboard</title>
  <style>
    :root {
      --bg: #f4f1ea;
      --panel: #fffdf9;
      --line: #d8d0c2;
      --text: #1f1d1a;
      --muted: #6e665b;
      --accent: #0b6e4f;
      --accent-2: #b46a28;
      --shadow: rgba(31, 29, 26, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(180, 106, 40, 0.10), transparent 30%),
        radial-gradient(circle at top right, rgba(11, 110, 79, 0.10), transparent 26%),
        linear-gradient(180deg, #f8f5ee 0%, var(--bg) 100%);
    }
    a { color: var(--accent); }
    .page {
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.3fr 0.7fr;
      gap: 18px;
      align-items: start;
      margin-bottom: 18px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 12px 30px var(--shadow);
    }
    .hero-main {
      padding: 22px 24px;
    }
    .hero-main h1 {
      margin: 0 0 8px 0;
      font-size: 34px;
      line-height: 1.05;
      letter-spacing: -0.02em;
    }
    .hero-main p {
      margin: 0;
      color: var(--muted);
      font-size: 16px;
      line-height: 1.5;
    }
    .hero-side {
      padding: 18px 20px;
    }
    .download-links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 14px;
    }
    .download-links a {
      display: inline-block;
      text-decoration: none;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #faf7f0;
    }
    .summary-wrap {
      margin-bottom: 18px;
      overflow: hidden;
    }
    .summary-wrap h2,
    .controls h2,
    .results h2,
    .detail h2 {
      margin: 0 0 12px 0;
      font-size: 18px;
      letter-spacing: 0.02em;
      text-transform: uppercase;
      color: var(--muted);
    }
    .summary-wrap .inner,
    .controls .inner,
    .results .inner,
    .detail .inner {
      padding: 18px 20px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      font-size: 12px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
      background: #fbf8f2;
      position: sticky;
      top: 0;
      z-index: 1;
    }
    .controls {
      margin-bottom: 18px;
    }
    .control-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }
    .button-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }
    button {
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #faf7f0;
      color: var(--text);
      font: inherit;
      cursor: pointer;
    }
    button:hover {
      background: #f5efe4;
    }
    button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
    }
    label {
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }
    input, select {
      width: 100%;
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: #fff;
      color: var(--text);
      font: inherit;
    }
    .results-detail {
      display: grid;
      grid-template-columns: 1.1fr 0.9fr;
      gap: 18px;
      align-items: start;
    }
    .table-wrap {
      max-height: 74vh;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }
    tbody tr {
      cursor: pointer;
      transition: background 0.15s ease;
    }
    tbody tr:hover {
      background: rgba(11, 110, 79, 0.05);
    }
    tbody tr.active {
      background: rgba(180, 106, 40, 0.10);
    }
    .detail-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .media-card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 10px;
      background: #fff;
    }
    .media-card h3,
    .meta h3 {
      margin: 0 0 8px 0;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }
    video, img {
      width: 100%;
      border-radius: 10px;
      background: #111;
    }
    .meta {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
      padding: 14px;
    }
    .meta-grid {
      display: grid;
      grid-template-columns: 140px 1fr;
      gap: 8px 12px;
      font-size: 14px;
    }
    .meta-key {
      color: var(--muted);
    }
    .score-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .score-box {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: #faf7f0;
    }
    .score-box h4 {
      margin: 0 0 8px 0;
      font-size: 13px;
      color: var(--accent-2);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .score-list {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 6px 10px;
      font-size: 14px;
    }
    .muted { color: var(--muted); }
    .pill {
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: rgba(11, 110, 79, 0.10);
      color: var(--accent);
      font-size: 12px;
      margin-right: 6px;
    }
    .compare-tools {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }
    .compare-tools label {
      flex-direction: row;
      align-items: center;
      gap: 8px;
      color: var(--text);
    }
    .compare-tools input[type="checkbox"] {
      width: auto;
      margin: 0;
    }
    @media (max-width: 1100px) {
      .hero,
      .results-detail,
      .detail-grid,
      .control-grid {
        grid-template-columns: 1fr;
      }
      .table-wrap {
        max-height: 50vh;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="card hero-main">
        <h1>PhysV Benchmark Dashboard</h1>
        <p>
          This page summarizes the benchmark runs built from the first 8 frames as context.
          Click any sample row to inspect the original full video, the future clip used for benchmarking,
          the last context frame, and the merged short / I2V scores.
        </p>
        <div class="download-links">
          <a href="data/summary_scores.csv">Download summary_scores.csv</a>
          <a href="data/sample_scores.csv">Download sample_scores.csv</a>
          <a href="data/summary_scores.json">Open summary_scores.json</a>
          <a href="data/sample_records.json">Open sample_records.json</a>
        </div>
      </div>
      <div class="card hero-side">
        <div><span class="pill" id="hero-datasets">0 datasets</span><span class="pill" id="hero-samples">0 samples</span></div>
        <p class="muted" style="margin-top: 12px;">
          Served as a static page. The video URLs resolve directly under the local data root, so you can inspect
          both source videos and benchmark-prepared future clips without extra copying.
        </p>
      </div>
    </section>

    <section class="card summary-wrap">
      <div class="inner">
        <h2>Aggregate Scores</h2>
        <div class="table-wrap">
          <table id="summary-table"></table>
        </div>
      </div>
    </section>

    <section class="card controls">
      <div class="inner">
        <h2>Filters</h2>
        <div class="control-grid">
          <label>
            Dataset
            <select id="dataset-filter"></select>
          </label>
          <label>
            Search
            <input id="search-filter" type="text" placeholder="sample_id / prompt">
          </label>
          <label>
            Sort By
            <select id="sort-key"></select>
          </label>
          <label>
            Sort Order
            <select id="sort-order">
              <option value="desc">High to low</option>
              <option value="asc">Low to high</option>
            </select>
          </label>
        </div>
        <div class="button-row">
          <button id="export-filtered" class="primary" type="button">Export Current Filtered CSV</button>
          <button id="clear-filters" type="button">Clear Filters</button>
        </div>
      </div>
    </section>

    <section class="results-detail">
      <section class="card results">
        <div class="inner">
          <h2>Samples</h2>
          <p class="muted" id="result-count"></p>
          <div class="table-wrap">
            <table id="sample-table"></table>
          </div>
        </div>
      </section>

      <section class="card detail">
        <div class="inner">
          <h2>Sample Detail</h2>
          <div id="detail-panel" class="muted">Select one sample row to inspect media and scores.</div>
        </div>
      </section>
    </section>
  </div>

  <script>
    const SUMMARY_URL = "data/summary_scores.json";
    const SAMPLE_URL = "data/sample_records.json";

    const SUMMARY_COLUMNS = [
      "dataset", "num_samples",
      "short_subject_consistency", "short_background_consistency", "short_motion_smoothness",
      "short_temporal_flickering", "short_dynamic_degree", "short_aesthetic_quality",
      "i2v_i2v_subject", "i2v_subject_consistency", "i2v_background_consistency",
      "i2v_motion_smoothness", "i2v_dynamic_degree", "i2v_aesthetic_quality"
    ];

    const SAMPLE_COLUMNS = [
      "dataset", "sample_id", "short_subject_consistency", "i2v_i2v_subject",
      "short_background_consistency", "short_dynamic_degree", "short_aesthetic_quality", "prompt"
    ];

    const SORT_KEYS = [
      "sample_id",
      "short_subject_consistency",
      "i2v_i2v_subject",
      "short_background_consistency",
      "short_motion_smoothness",
      "short_temporal_flickering",
      "short_dynamic_degree",
      "short_aesthetic_quality"
    ];

    let summaryRows = [];
    let sampleRows = [];
    let filteredRows = [];
    let selectedSampleId = null;
    let syncLock = false;

    function fmt(value) {
      if (value === null || value === undefined || value === "") return "";
      if (typeof value === "number") return Number(value.toFixed(4)).toString();
      return String(value);
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }

    function renderTable(tableEl, columns, rows) {
      const header = "<thead><tr>" + columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("") + "</tr></thead>";
      const body = "<tbody>" + rows.map((row) => {
        const attrs = row.sample_id ? ` data-sample-id="${escapeHtml(row.sample_id)}"` : "";
        const cls = row.sample_id && row.sample_id === selectedSampleId ? " class=\\"active\\"" : "";
        const cells = columns.map((c) => `<td>${escapeHtml(fmt(row[c]))}</td>`).join("");
        return `<tr${attrs}${cls}>${cells}</tr>`;
      }).join("") + "</tbody>";
      tableEl.innerHTML = header + body;
    }

    function renderSummary() {
      renderTable(document.getElementById("summary-table"), SUMMARY_COLUMNS, summaryRows);
      document.getElementById("hero-datasets").textContent = `${summaryRows.length} datasets`;
      document.getElementById("hero-samples").textContent = `${sampleRows.length} samples`;
    }

    function populateControls() {
      const datasetFilter = document.getElementById("dataset-filter");
      const datasets = ["all", ...new Set(sampleRows.map((r) => r.dataset))];
      datasetFilter.innerHTML = datasets.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");

      const sortKey = document.getElementById("sort-key");
      sortKey.innerHTML = SORT_KEYS.map((k) => `<option value="${escapeHtml(k)}">${escapeHtml(k)}</option>`).join("");
      sortKey.value = "short_subject_consistency";
    }

    function applyFilters() {
      const dataset = document.getElementById("dataset-filter").value;
      const search = document.getElementById("search-filter").value.trim().toLowerCase();
      const sortKey = document.getElementById("sort-key").value;
      const sortOrder = document.getElementById("sort-order").value;

      filteredRows = sampleRows.filter((row) => {
        if (dataset !== "all" && row.dataset !== dataset) return false;
        if (!search) return true;
        return (row.sample_id + " " + row.prompt).toLowerCase().includes(search);
      });

      filteredRows.sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (typeof av === "number" && typeof bv === "number") {
          return sortOrder === "asc" ? av - bv : bv - av;
        }
        const ac = String(av || "");
        const bc = String(bv || "");
        return sortOrder === "asc" ? ac.localeCompare(bc) : bc.localeCompare(ac);
      });

      document.getElementById("result-count").textContent = `${filteredRows.length} samples shown`;
      renderTable(document.getElementById("sample-table"), SAMPLE_COLUMNS, filteredRows);
      bindRowClicks();

      if (!filteredRows.find((row) => row.sample_id === selectedSampleId)) {
        selectedSampleId = filteredRows.length ? filteredRows[0].sample_id : null;
      }
      renderDetail();
    }

    function scorePairs(row, prefix) {
      return Object.entries(row)
        .filter(([key, value]) => key.startsWith(prefix) && typeof value === "number")
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([key, value]) => [key.slice(prefix.length), value]);
    }

    function csvEscape(value) {
      const text = value === null || value === undefined ? "" : String(value);
      if (text.includes(",") || text.includes('"') || text.includes("\\n")) {
        return `"${text.replaceAll('"', '""')}"`;
      }
      return text;
    }

    function downloadFilteredCsv() {
      if (!filteredRows.length) return;
      const columns = Array.from(filteredRows.reduce((acc, row) => {
        Object.keys(row).forEach((key) => acc.add(key));
        return acc;
      }, new Set()));
      const lines = [
        columns.map(csvEscape).join(","),
        ...filteredRows.map((row) => columns.map((key) => csvEscape(row[key])).join(",")),
      ];
      const blob = new Blob([lines.join("\\n")], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "filtered_sample_scores.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    function syncVideos(originalVideo, futureVideo, offsetSeconds, driver) {
      if (syncLock) return;
      syncLock = true;
      try {
        if (driver === originalVideo) {
          futureVideo.currentTime = Math.max(0, originalVideo.currentTime - offsetSeconds);
        } else {
          originalVideo.currentTime = futureVideo.currentTime + offsetSeconds;
        }
      } finally {
        syncLock = false;
      }
    }

    function wireComparePlayers(panel) {
      const originalVideo = panel.querySelector("#orig-video");
      const futureVideo = panel.querySelector("#future-video");
      const syncToggle = panel.querySelector("#sync-toggle");
      const playBoth = panel.querySelector("#play-both");
      const pauseBoth = panel.querySelector("#pause-both");
      const resetBoth = panel.querySelector("#reset-both");
      const offsetInfo = panel.querySelector("#offset-info");
      let offsetSeconds = 0;

        function recomputeOffset() {
          if (originalVideo.duration && futureVideo.duration) {
            offsetSeconds = Math.max(0, originalVideo.duration - futureVideo.duration);
            offsetInfo.textContent = `auto offset: ${Number(offsetSeconds.toFixed(4)).toString()} s`;
          }
        }

      originalVideo.addEventListener("loadedmetadata", recomputeOffset);
      futureVideo.addEventListener("loadedmetadata", recomputeOffset);

      originalVideo.addEventListener("play", () => {
        if (!syncToggle.checked) return;
        futureVideo.play().catch(() => {});
      });
      futureVideo.addEventListener("play", () => {
        if (!syncToggle.checked) return;
        originalVideo.play().catch(() => {});
      });
      originalVideo.addEventListener("pause", () => {
        if (!syncToggle.checked || syncLock) return;
        futureVideo.pause();
      });
      futureVideo.addEventListener("pause", () => {
        if (!syncToggle.checked || syncLock) return;
        originalVideo.pause();
      });

      ["seeked", "timeupdate"].forEach((eventName) => {
        originalVideo.addEventListener(eventName, () => {
          if (!syncToggle.checked) return;
          syncVideos(originalVideo, futureVideo, offsetSeconds, originalVideo);
        });
        futureVideo.addEventListener(eventName, () => {
          if (!syncToggle.checked) return;
          syncVideos(originalVideo, futureVideo, offsetSeconds, futureVideo);
        });
      });

      playBoth.addEventListener("click", () => {
        if (syncToggle.checked) {
          futureVideo.currentTime = Math.max(0, originalVideo.currentTime - offsetSeconds);
        }
        Promise.allSettled([originalVideo.play(), futureVideo.play()]);
      });
      pauseBoth.addEventListener("click", () => {
        originalVideo.pause();
        futureVideo.pause();
      });
      resetBoth.addEventListener("click", () => {
        originalVideo.pause();
        futureVideo.pause();
        originalVideo.currentTime = 0;
        futureVideo.currentTime = 0;
      });
    }

    function renderDetail() {
      const panel = document.getElementById("detail-panel");
      if (!selectedSampleId) {
        panel.innerHTML = '<div class="muted">No sample available for the current filter.</div>';
        return;
      }

      const row = sampleRows.find((item) => item.sample_id === selectedSampleId);
      if (!row) {
        panel.innerHTML = '<div class="muted">Selected sample was not found.</div>';
        return;
      }

      const shortScores = scorePairs(row, "short_");
      const i2vScores = scorePairs(row, "i2v_");

      panel.innerHTML = `
        <div class="compare-tools">
          <button id="play-both" class="primary" type="button">Play Both</button>
          <button id="pause-both" type="button">Pause Both</button>
          <button id="reset-both" type="button">Reset</button>
          <label><input id="sync-toggle" type="checkbox" checked> Sync future clip to original timeline</label>
          <span id="offset-info" class="pill">auto offset: waiting metadata</span>
        </div>
        <div class="detail-grid">
          <div class="media-card">
            <h3>Original Full Video</h3>
            <video id="orig-video" controls preload="metadata" src="${escapeHtml(row.original_video_url || "")}"></video>
          </div>
          <div class="media-card">
            <h3>Future Clip Used For Benchmark</h3>
            <video id="future-video" controls preload="metadata" src="${escapeHtml(row.future_video_url || "")}"></video>
          </div>
        </div>
        <div class="detail-grid" style="margin-top: 12px;">
          <div class="media-card">
            <h3>Last Context Frame</h3>
            <img alt="context frame" src="${escapeHtml(row.context_image_url || "")}">
          </div>
          <div class="meta">
            <h3>Sample Info</h3>
            <div class="meta-grid">
              <div class="meta-key">Dataset</div><div>${escapeHtml(row.dataset)}</div>
              <div class="meta-key">Sample ID</div><div>${escapeHtml(row.sample_id)}</div>
              <div class="meta-key">Prompt</div><div>${escapeHtml(row.prompt)}</div>
              <div class="meta-key">Original Video</div><div><a href="${escapeHtml(row.original_video_url || "")}" target="_blank">${escapeHtml(row.original_video_url || "")}</a></div>
              <div class="meta-key">Future Video</div><div><a href="${escapeHtml(row.future_video_url || "")}" target="_blank">${escapeHtml(row.future_video_url || "")}</a></div>
              <div class="meta-key">Context Image</div><div><a href="${escapeHtml(row.context_image_url || "")}" target="_blank">${escapeHtml(row.context_image_url || "")}</a></div>
            </div>
          </div>
        </div>
        <div class="score-grid">
          <div class="score-box">
            <h4>Short Scores</h4>
            <div class="score-list">
              ${shortScores.map(([key, value]) => `<div>${escapeHtml(key)}</div><div>${escapeHtml(fmt(value))}</div>`).join("")}
            </div>
          </div>
          <div class="score-box">
            <h4>I2V Scores</h4>
            <div class="score-list">
              ${i2vScores.map(([key, value]) => `<div>${escapeHtml(key)}</div><div>${escapeHtml(fmt(value))}</div>`).join("")}
            </div>
          </div>
        </div>
      `;

      wireComparePlayers(panel);

      const rows = document.querySelectorAll("#sample-table tbody tr");
      rows.forEach((tr) => {
        tr.classList.toggle("active", tr.dataset.sampleId === selectedSampleId);
      });
    }

    function bindRowClicks() {
      const rows = document.querySelectorAll("#sample-table tbody tr");
      rows.forEach((tr) => {
        tr.onclick = () => {
          selectedSampleId = tr.dataset.sampleId;
          renderDetail();
        };
      });
    }

    async function loadData() {
      const [summaryResp, sampleResp] = await Promise.all([
        fetch(SUMMARY_URL),
        fetch(SAMPLE_URL),
      ]);
      summaryRows = await summaryResp.json();
      sampleRows = await sampleResp.json();
      renderSummary();
      populateControls();
      applyFilters();
    }

    document.getElementById("dataset-filter")?.addEventListener("change", applyFilters);
    document.getElementById("search-filter")?.addEventListener("input", applyFilters);
    document.getElementById("sort-key")?.addEventListener("change", applyFilters);
    document.getElementById("sort-order")?.addEventListener("change", applyFilters);
    document.getElementById("export-filtered")?.addEventListener("click", downloadFilteredCsv);
    document.getElementById("clear-filters")?.addEventListener("click", () => {
      document.getElementById("dataset-filter").value = "all";
      document.getElementById("search-filter").value = "";
      document.getElementById("sort-key").value = "short_subject_consistency";
      document.getElementById("sort-order").value = "desc";
      applyFilters();
    });

    loadData().catch((err) => {
      document.getElementById("detail-panel").textContent = `Failed to load dashboard data: ${err}`;
    });
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CSVs and a static dashboard for PhysV benchmark results.")
    parser.add_argument("--serve-root", required=True, help="HTTP server root containing both datasets and benchmark outputs.")
    parser.add_argument("--prepared-root", required=True, help="Prepared benchmark root with manifest.jsonl files.")
    parser.add_argument("--output-root", required=True, help="Benchmark output root containing short_ctx8/i2v_ctx8 results.")
    parser.add_argument("--dashboard-dir", required=True, help="Destination directory for dashboard assets.")
    parser.add_argument(
        "--original-root",
        action="append",
        default=[],
        help="Dataset mapping in NAME=PATH form. Repeat for each original dataset root.",
    )
    return parser.parse_args()


def parse_original_roots(items: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --original-root entry: {item}")
        name, raw_path = item.split("=", 1)
        mapping[name] = Path(raw_path).expanduser().resolve()
    return mapping


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(round_payload(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def round_float(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def round_payload(payload: Any, digits: int = 4) -> Any:
    if isinstance(payload, float):
        return round_float(payload, digits)
    if isinstance(payload, list):
        return [round_payload(item, digits) for item in payload]
    if isinstance(payload, dict):
        return {key: round_payload(value, digits) for key, value in payload.items()}
    return payload


def relative_url(path: str | Path | None, serve_root: Path) -> str:
    if not path:
        return ""
    resolved = Path(path).expanduser().resolve()
    return "/" + resolved.relative_to(serve_root).as_posix()


def last_context_image(sample: BenchSample) -> Path | None:
    if sample.context_frame_paths:
        paths = sorted(Path(p).expanduser().resolve() for p in sample.context_frame_paths)
        return paths[-1] if paths else None
    if sample.context_frames_dir:
        ctx_dir = Path(sample.context_frames_dir).expanduser().resolve()
        files = sorted(p for p in ctx_dir.iterdir() if p.is_file())
        return files[-1] if files else None
    return None


def original_sample_index(root: Path) -> dict[str, dict[str, str]]:
    index: dict[str, dict[str, str]] = {}
    for caption_path in sorted(root.rglob("caption_simple.txt")):
        sample_dir = caption_path.parent
        video_path = sample_dir / "videos" / "rgb.mp4"
        if not video_path.exists():
            continue
        rel = sample_dir.relative_to(root)
        sample_id = "__".join(rel.parts)
        index[sample_id] = {
            "sample_dir": str(sample_dir.resolve()),
            "video_path": str(video_path.resolve()),
            "prompt": caption_path.read_text(encoding="utf-8").strip(),
        }
    return index


def detail_lookup(eval_data: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    per_dimension: dict[str, dict[str, dict[str, Any]]] = {}
    for dimension, payload in eval_data.items():
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
            continue
        dim_map: dict[str, dict[str, Any]] = {}
        for item in payload[1]:
            video_path = item.get("video_path")
            if not video_path:
                continue
            dim_map[Path(video_path).name] = item
        per_dimension[dimension] = dim_map
    return per_dimension


def collect_score(detail_map: dict[str, dict[str, dict[str, Any]]], dimension: str, staged_video_name: str) -> float | None:
    item = detail_map.get(dimension, {}).get(staged_video_name)
    if not item:
        return None
    value = item.get("video_results")
    return round_float(value) if isinstance(value, (float, int)) else None


def load_dataset_records(
    dataset_name: str,
    manifest_path: Path,
    short_eval_path: Path,
    i2v_eval_path: Path,
    original_index_map: dict[str, dict[str, str]],
    serve_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    samples = load_manifest(str(manifest_path))
    short_eval = load_json(short_eval_path)
    i2v_eval = load_json(i2v_eval_path)
    short_lookup = detail_lookup(short_eval)
    i2v_lookup = detail_lookup(i2v_eval)

    summary: dict[str, Any] = {"dataset": dataset_name, "num_samples": len(samples)}
    for dimension, payload in short_eval.items():
        summary[f"short_{dimension}"] = round_float(payload[0])
    for dimension, payload in i2v_eval.items():
        summary[f"i2v_{dimension}"] = round_float(payload[0])

    rows: list[dict[str, Any]] = []
    for sample in samples:
        suffix = Path(sample.video_path).suffix.lower()
        staged_video_name = f"{safe_stem(sample)}{suffix}"
        original_meta = original_index_map.get(sample.sample_id, {})
        context_image = last_context_image(sample)

        row: dict[str, Any] = {
            "dataset": dataset_name,
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "original_video_path": original_meta.get("video_path", ""),
            "future_video_path": str(Path(sample.video_path).expanduser().resolve()),
            "context_image_path": str(context_image) if context_image else "",
            "original_video_url": relative_url(original_meta.get("video_path"), serve_root),
            "future_video_url": relative_url(sample.video_path, serve_root),
            "context_image_url": relative_url(context_image, serve_root) if context_image else "",
        }

        for dimension in short_eval:
            row[f"short_{dimension}"] = collect_score(short_lookup, dimension, staged_video_name)
        for dimension in i2v_eval:
            row[f"i2v_{dimension}"] = collect_score(i2v_lookup, dimension, staged_video_name)

        rows.append(row)

    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {path}")

    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(round_payload(rows))


def main() -> None:
    args = parse_args()
    serve_root = Path(args.serve_root).expanduser().resolve()
    prepared_root = Path(args.prepared_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    dashboard_dir = Path(args.dashboard_dir).expanduser().resolve()
    original_roots = parse_original_roots(args.original_root)

    original_indices = {name: original_sample_index(path) for name, path in original_roots.items()}

    summaries: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    for manifest_path in sorted(prepared_root.glob("*/manifest.jsonl")):
        dataset_name = manifest_path.parent.name
        if dataset_name not in original_indices:
            raise KeyError(f"No --original-root mapping provided for dataset {dataset_name}")

        short_eval_path = output_root / dataset_name / "short_ctx8" / "short_ctx8_eval_results.json"
        i2v_eval_path = output_root / dataset_name / "i2v_ctx8" / "i2v_ctx8_eval_results.json"
        if not short_eval_path.exists():
            raise FileNotFoundError(f"Missing short eval file: {short_eval_path}")
        if not i2v_eval_path.exists():
            raise FileNotFoundError(f"Missing i2v eval file: {i2v_eval_path}")

        summary, rows = load_dataset_records(
            dataset_name=dataset_name,
            manifest_path=manifest_path,
            short_eval_path=short_eval_path,
            i2v_eval_path=i2v_eval_path,
            original_index_map=original_indices[dataset_name],
            serve_root=serve_root,
        )
        summaries.append(summary)
        sample_rows.extend(rows)

    data_dir = dashboard_dir / "data"
    write_csv(data_dir / "summary_scores.csv", summaries)
    write_csv(data_dir / "sample_scores.csv", sample_rows)
    write_json(data_dir / "summary_scores.json", summaries)
    write_json(data_dir / "sample_records.json", sample_rows)
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    (dashboard_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")

    payload = {
        "dashboard_dir": str(dashboard_dir),
        "summary_csv": str(data_dir / "summary_scores.csv"),
        "sample_csv": str(data_dir / "sample_scores.csv"),
        "num_datasets": len(summaries),
        "num_samples": len(sample_rows),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
