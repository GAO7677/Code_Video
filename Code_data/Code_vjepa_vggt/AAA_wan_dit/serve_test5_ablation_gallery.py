#!/usr/bin/env python3
"""Serve a live, case-grouped gallery for the five-case ablation pipeline."""

from __future__ import annotations

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from build_v2v_wan_case_gallery import Method, build_manifest


def config_name(mode: str, block: str) -> str:
    return "baseline" if mode == "baseline" else f"{mode}_block{int(block):02d}"


def load_methods(root: Path, run_root: Path) -> list[Method]:
    queue_path = run_root / "generation" / "queue.tsv"
    methods: list[Method] = []
    for line in queue_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        task_id, model, mode, block = line.split("\t")
        name = config_name(mode, block)
        validation = run_root / "generation" / "validations" / f"{task_id}.json"
        if validation.is_file():
            payload = json.loads(validation.read_text(encoding="utf-8"))
            result_dir = Path(payload["result_root"]).expanduser().resolve()
        elif model == "wan_lora":
            result_dir = root / "wan_lora" / name
        elif model == "xssc":
            result_dir = root / "xssc" / name / "results"
        else:
            result_dir = (
                root
                / "PhyRVG"
                / name
                / "input_first5_unique"
                / "physRVG_steps40_512x896_08_49f"
            )
        normalized_model = "physrvg" if model == "physrvg" else model
        methods.append(
            Method(
                method_id=f"{normalized_model}/{name}",
                model=normalized_model,
                mode=mode,
                block_id=None if block == "none" else int(block),
                result_dir=result_dir,
            )
        )
    if len(methods) != 63 or len({method.method_id for method in methods}) != 63:
        raise RuntimeError(f"expected 63 unique methods, found {len(methods)}")
    return sorted(methods, key=lambda method: method.sort_key)


def live_index_html() -> str:
    return r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Five-Case DiT Ablation Videos</title>
  <style>
    :root {
      --bg: #f2f4f3;
      --surface: #fff;
      --text: #17211c;
      --muted: #65736c;
      --line: #ccd5d0;
      --green: #176b45;
      --blue: #285f99;
      --orange: #a34e1c;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 14px Arial, sans-serif;
      letter-spacing: 0;
    }
    button, select { font: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(380px, 1.4fr) auto;
      gap: 14px;
      align-items: center;
      padding: 12px 18px;
      color: #fff;
      background: #18382a;
      border-bottom: 1px solid #315641;
    }
    h1 { margin: 0; font-size: 19px; }
    #status { margin-top: 3px; color: #bad0c4; font-size: 12px; }
    .case-nav { display: grid; grid-template-columns: 38px 1fr 38px; gap: 7px; }
    button, select {
      min-height: 38px;
      border: 1px solid #557765;
      border-radius: 4px;
    }
    button { color: #fff; background: #24503a; cursor: pointer; padding: 7px 11px; }
    button:hover { background: #2e6248; }
    select { min-width: 0; padding: 7px 9px; background: #fff; color: var(--text); }
    .commands { display: flex; flex-wrap: wrap; gap: 7px; justify-content: flex-end; }
    main { max-width: 2200px; margin: auto; padding: 18px 18px 40px; }
    .case-head { display: grid; grid-template-columns: minmax(0, 1fr) 420px; gap: 16px; }
    .case-copy h2 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
    .prompt { margin: 8px 0 0; color: var(--muted); line-height: 1.5; }
    .source { min-width: 0; }
    .source-label { margin-bottom: 6px; color: var(--green); font-weight: 700; font-size: 12px; }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #090d0b;
    }
    .source video { border: 1px solid #18251f; }
    .models { display: grid; gap: 26px; margin-top: 22px; }
    .model-section {
      min-width: 0;
      padding-top: 14px;
      border-top: 3px solid var(--blue);
    }
    .model-section[data-model="xssc"] { border-top-color: var(--green); }
    .model-section[data-model="physrvg"] { border-top-color: var(--orange); }
    .model-header {
      display: grid;
      grid-template-columns: 170px minmax(220px, 360px);
      gap: 12px;
      align-items: start;
      margin-bottom: 12px;
    }
    .model-header h3 { margin: 3px 0 0; font-size: 18px; }
    .baseline-label { margin-bottom: 5px; color: var(--muted); font-size: 11px; font-weight: 700; }
    .matrix-wrap { overflow-x: auto; padding-bottom: 5px; }
    table {
      width: 100%;
      min-width: 1540px;
      border-collapse: collapse;
      table-layout: fixed;
      background: var(--surface);
    }
    th, td { border: 1px solid var(--line); }
    thead th {
      height: 36px;
      padding: 7px;
      background: #e8eeea;
      font-size: 12px;
      text-align: center;
    }
    thead th:first-child { width: 150px; text-align: left; }
    tbody th {
      width: 150px;
      padding: 9px;
      color: #33463c;
      background: #f5f8f6;
      font-size: 12px;
      text-align: left;
      overflow-wrap: anywhere;
    }
    td { padding: 5px; vertical-align: top; background: #fff; }
    .video-cell-label { padding: 2px 1px 5px; color: var(--muted); font-size: 10px; }
    .pending {
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 16 / 9;
      color: #9a592f;
      background: #fff4eb;
      font-size: 12px;
    }
    .empty { color: #9aa39f; text-align: center; }
    @media (max-width: 900px) {
      .topbar { grid-template-columns: 1fr; }
      .commands { justify-content: flex-start; }
      .case-head { grid-template-columns: 1fr; }
      .model-header { grid-template-columns: 1fr; }
      main { padding: 14px 10px 30px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>DiT Ablation Video Matrix</h1>
      <div id="status">Loading...</div>
    </div>
    <div class="case-nav">
      <button id="previousCase" title="Previous case">&lt;</button>
      <select id="caseSelect" aria-label="Select case"></select>
      <button id="nextCase" title="Next case">&gt;</button>
    </div>
    <div class="commands">
      <button id="refreshData">Refresh</button>
      <button id="playVisible">Play all</button>
      <button id="pauseVisible">Pause all</button>
    </div>
  </header>
  <main>
    <section class="case-head">
      <div class="case-copy">
        <h2 id="caseName"></h2>
        <p class="prompt" id="prompt"></p>
      </div>
      <div class="source" id="source"></div>
    </section>
    <div class="models" id="models"></div>
  </main>
  <script>
    const BLOCKS = [0, 5, 11, 17, 19, 29];
    const MODEL_ORDER = ["wan_lora", "xssc", "physrvg"];
    const MODEL_MODES = {
      wan_lora: ["whole_block", "self_attn_zero"],
      xssc: ["whole_block", "self_attn_zero", "object_cross_attn"],
      physrvg: ["whole_block", "self_attn_zero", "text_cross_attn_zero", "ffn_zero", "lora_off"]
    };
    const MODE_LABELS = {
      whole_block: "Whole block bypass",
      self_attn_zero: "Self-attention output = 0",
      object_cross_attn: "Object cross-attention output = 0",
      text_cross_attn_zero: "Text cross-attention output = 0",
      ffn_zero: "FFN output = 0",
      lora_off: "LoRA disabled"
    };
    const state = {manifest: null, caseIndex: 0};
    const select = document.getElementById("caseSelect");

    function makeVideo(url, preload = "none") {
      const video = document.createElement("video");
      video.controls = true;
      video.preload = preload;
      video.playsInline = true;
      video.src = `../${url}`;
      return video;
    }

    function outputFor(item, method) {
      return method ? item.outputs[method.id] : null;
    }

    function methodFor(model, mode, blockId) {
      return state.manifest.methods.find(method =>
        method.model === model && method.mode === mode && method.block_id === blockId
      );
    }

    function appendVideoOrPending(parent, output) {
      if (output && output.video) parent.appendChild(makeVideo(output.video));
      else {
        const pending = document.createElement("div");
        pending.className = "pending";
        pending.textContent = "Pending";
        parent.appendChild(pending);
      }
    }

    function renderModel(item, model) {
      const section = document.createElement("section");
      section.className = "model-section";
      section.dataset.model = model;
      const modelMethods = state.manifest.methods.filter(method => method.model === model);
      const header = document.createElement("div");
      header.className = "model-header";
      const title = document.createElement("h3");
      title.textContent = modelMethods[0].model_label;
      header.appendChild(title);
      const baseline = document.createElement("div");
      const baselineLabel = document.createElement("div");
      baselineLabel.className = "baseline-label";
      baselineLabel.textContent = "Baseline";
      baseline.appendChild(baselineLabel);
      appendVideoOrPending(baseline, outputFor(item, methodFor(model, "baseline", null)));
      header.appendChild(baseline);
      section.appendChild(header);

      const wrap = document.createElement("div");
      wrap.className = "matrix-wrap";
      const table = document.createElement("table");
      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      const modeHead = document.createElement("th");
      modeHead.textContent = "Ablation";
      headRow.appendChild(modeHead);
      BLOCKS.forEach(block => {
        const th = document.createElement("th");
        th.textContent = `Block ${String(block).padStart(2, "0")}`;
        headRow.appendChild(th);
      });
      thead.appendChild(headRow);
      table.appendChild(thead);

      const tbody = document.createElement("tbody");
      MODEL_MODES[model].forEach(mode => {
        const row = document.createElement("tr");
        const label = document.createElement("th");
        label.textContent = MODE_LABELS[mode];
        row.appendChild(label);
        BLOCKS.forEach(block => {
          const cell = document.createElement("td");
          const cellLabel = document.createElement("div");
          cellLabel.className = "video-cell-label";
          cellLabel.textContent = `${MODE_LABELS[mode]} / Block ${String(block).padStart(2, "0")}`;
          cell.appendChild(cellLabel);
          appendVideoOrPending(cell, outputFor(item, methodFor(model, mode, block)));
          row.appendChild(cell);
        });
        tbody.appendChild(row);
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
      section.appendChild(wrap);
      return section;
    }

    function renderSelector() {
      select.replaceChildren();
      state.manifest.cases.forEach((item, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = `${index + 1}. ${item.name}`;
        select.appendChild(option);
      });
      select.value = String(state.caseIndex);
    }

    function renderCase() {
      const item = state.manifest.cases[state.caseIndex];
      select.value = String(state.caseIndex);
      document.getElementById("caseName").textContent = item.name;
      document.getElementById("prompt").textContent = item.prompt || "";
      const source = document.getElementById("source");
      source.replaceChildren();
      const label = document.createElement("div");
      label.className = "source-label";
      label.textContent = "Ground truth / Source";
      source.appendChild(label);
      if (item.gt_video) source.appendChild(makeVideo(item.gt_video, "metadata"));
      else {
        const missing = document.createElement("div");
        missing.className = "pending";
        missing.textContent = "Source unavailable";
        source.appendChild(missing);
      }
      const models = document.getElementById("models");
      models.replaceChildren();
      MODEL_ORDER.forEach(model => models.appendChild(renderModel(item, model)));
      history.replaceState(null, "", `#case=${state.caseIndex + 1}`);
    }

    function moveCase(delta) {
      state.caseIndex = (state.caseIndex + delta + state.manifest.cases.length)
        % state.manifest.cases.length;
      renderCase();
    }

    async function refreshManifest(initial = false) {
      const response = await fetch(`./manifest.json?t=${Date.now()}`, {cache: "no-store"});
      state.manifest = await response.json();
      if (initial) {
        const match = location.hash.match(/case=(\d+)/);
        if (match) state.caseIndex = Math.max(
          0, Math.min(state.manifest.num_cases - 1, Number(match[1]) - 1)
        );
      }
      document.getElementById("status").textContent =
        `${state.manifest.num_cases} cases | ${state.manifest.num_methods} methods | ` +
        `${state.manifest.num_available_videos}/${state.manifest.num_cases * state.manifest.num_methods} videos ready`;
      renderSelector();
      renderCase();
    }

    select.addEventListener("change", () => {
      state.caseIndex = Number(select.value);
      renderCase();
    });
    document.getElementById("previousCase").addEventListener("click", () => moveCase(-1));
    document.getElementById("nextCase").addEventListener("click", () => moveCase(1));
    document.getElementById("refreshData").addEventListener("click", () => refreshManifest());
    document.getElementById("playVisible").addEventListener("click", () => {
      document.querySelectorAll("video").forEach(video => video.play().catch(() => {}));
    });
    document.getElementById("pauseVisible").addEventListener("click", () => {
      document.querySelectorAll("video").forEach(video => video.pause());
    });
    document.addEventListener("keydown", event => {
      if (event.target.matches("select")) return;
      if (event.key === "ArrowLeft") moveCase(-1);
      if (event.key === "ArrowRight") moveCase(1);
    });
    setInterval(() => {
      const playing = Array.from(document.querySelectorAll("video"))
        .some(video => !video.paused && !video.ended);
      if (!playing) refreshManifest().catch(() => {});
    }, 30000);
    refreshManifest(true);
  </script>
</body>
</html>
"""


class GalleryHandler(SimpleHTTPRequestHandler):
    server_version = "AblationGallery/1.0"

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def do_GET(self) -> None:
        if urlparse(self.path).path == "/_gallery/manifest.json":
            try:
                manifest = self.server.build_live_manifest()  # type: ignore[attr-defined]
                body = (
                    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
                ).encode("utf-8")
            except Exception as exc:
                self.send_error(500, explain=str(exc))
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/data/gaoya/agent-data/outputs/wan_dit_ablation/test5_first5"
        ),
    )
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8913)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    run_root = (
        args.run_root.expanduser().resolve()
        if args.run_root is not None
        else root / "_pipeline"
    )
    output_dir = root / "_gallery"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(live_index_html(), encoding="utf-8")

    def build_live_manifest() -> dict[str, object]:
        methods = load_methods(root, run_root)
        manifest = build_manifest(root, output_dir, methods)
        manifest["title"] = "Five-Case Wan DiT Ablation Comparison"
        manifest["num_available_videos"] = sum(
            1
            for case in manifest["cases"]  # type: ignore[union-attr]
            for output in case["outputs"].values()  # type: ignore[index,union-attr]
            if output["video"] is not None
        )
        return manifest

    first_manifest = build_live_manifest()
    print(
        json.dumps(
            {
                "url": f"http://localhost:{args.port}/_gallery/",
                "root": str(root),
                "cases": first_manifest["num_cases"],
                "methods": first_manifest["num_methods"],
                "available_videos": first_manifest["num_available_videos"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    handler = partial(GalleryHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.build_live_manifest = build_live_manifest  # type: ignore[attr-defined]
    server.serve_forever()


if __name__ == "__main__":
    main()
