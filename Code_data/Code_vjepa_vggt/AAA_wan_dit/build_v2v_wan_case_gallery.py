#!/usr/bin/env python3
"""Build a case-grouped comparison portal for Wan DiT ablations."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


METHOD_PATTERN = re.compile(
    r"^(whole_block|self_attn_zero|object_cross_attn)_block(\d{2})$"
)
MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
}
MODE_LABELS = {
    "baseline": "Baseline",
    "whole_block": "Whole block bypass",
    "self_attn_zero": "Self-attention output = 0",
    "object_cross_attn": "Object cross-attention output = 0",
}


@dataclass(frozen=True)
class Method:
    method_id: str
    model: str
    mode: str
    block_id: int | None
    result_dir: Path

    @property
    def label(self) -> str:
        if self.block_id is None:
            return f"{MODEL_LABELS[self.model]} · {MODE_LABELS[self.mode]}"
        return (
            f"{MODEL_LABELS[self.model]} · Block {self.block_id:02d} · "
            f"{MODE_LABELS[self.mode]}"
        )

    @property
    def group(self) -> str:
        if self.block_id is None:
            return "Baseline"
        return f"Block {self.block_id:02d}"

    @property
    def sort_key(self) -> tuple[int, int, int]:
        block_order = -1 if self.block_id is None else self.block_id
        model_order = 0 if self.model == "wan_lora" else 1
        mode_order = {
            "baseline": 0,
            "whole_block": 1,
            "self_attn_zero": 2,
            "object_cross_attn": 3,
        }[self.mode]
        return block_order, model_order, mode_order


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("/data/gaoya/AAA_test_video/0623/test/v2v_wan"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def discover_methods(root: Path) -> list[Method]:
    methods: list[Method] = []
    for model in ("wan_lora", "xssc"):
        model_root = root / model
        for config_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
            if config_dir.name.startswith("_"):
                continue
            if config_dir.name == "baseline":
                mode = "baseline"
                block_id = None
            else:
                match = METHOD_PATTERN.match(config_dir.name)
                if match is None:
                    continue
                mode = match.group(1)
                block_id = int(match.group(2))
            result_dir = config_dir if model == "wan_lora" else config_dir / "results"
            if not result_dir.is_dir():
                continue
            methods.append(
                Method(
                    method_id=f"{model}/{config_dir.name}",
                    model=model,
                    mode=mode,
                    block_id=block_id,
                    result_dir=result_dir,
                )
            )
    return sorted(methods, key=lambda method: method.sort_key)


def load_case_reference(
    *,
    root: Path,
    output_dir: Path,
    case_name: str,
    methods: list[Method],
) -> dict[str, str | None]:
    prompt: str | None = None
    source_video: Path | None = None
    source_json: Path | None = None
    for method in methods:
        result_json = method.result_dir / f"{case_name}.json"
        if not result_json.is_file():
            continue
        try:
            payload = json.loads(result_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        value = payload.get("input_caption")
        if prompt is None and isinstance(value, str) and value.strip():
            prompt = value.strip()
        value = payload.get("source_video")
        if source_video is None and isinstance(value, str):
            candidate = Path(value).expanduser().resolve()
            if candidate.is_file():
                source_video = candidate
        value = payload.get("input_json")
        if source_json is None and isinstance(value, str):
            candidate = Path(value).expanduser().resolve()
            if candidate.is_file():
                source_json = candidate
                try:
                    source_payload = json.loads(candidate.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    source_payload = {}
                source_prompt = source_payload.get("input_caption")
                if prompt is None and isinstance(source_prompt, str) and source_prompt.strip():
                    prompt = source_prompt.strip()
                source_value = source_payload.get("source_video")
                if source_video is None and isinstance(source_value, str):
                    source_candidate = Path(source_value).expanduser().resolve()
                    if source_candidate.is_file():
                        source_video = source_candidate
        if prompt is not None and source_video is not None and source_json is not None:
            break

    gt_url: str | None = None
    if source_video is not None:
        gt_dir = output_dir / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        gt_link = gt_dir / f"{case_name}{source_video.suffix.lower()}"
        if gt_link.is_symlink() and gt_link.resolve() != source_video:
            gt_link.unlink()
        if not gt_link.exists():
            gt_link.symlink_to(source_video)
        gt_url = gt_link.absolute().relative_to(root.absolute()).as_posix()

    return {
        "prompt": prompt,
        "gt_video": gt_url,
        "source_json": str(source_json) if source_json is not None else None,
    }


def build_manifest(
    root: Path,
    output_dir: Path,
    methods: list[Method],
) -> dict[str, object]:
    case_names = sorted(
        {
            video_path.stem
            for method in methods
            for video_path in method.result_dir.glob("*.mp4")
        }
    )
    cases: list[dict[str, object]] = []
    for case_name in case_names:
        reference = load_case_reference(
            root=root,
            output_dir=output_dir,
            case_name=case_name,
            methods=methods,
        )
        outputs: dict[str, dict[str, str | None]] = {}
        input_image: str | None = None
        for method in methods:
            video_path = method.result_dir / f"{case_name}.mp4"
            json_path = method.result_dir / f"{case_name}.json"
            image_path = method.result_dir / f"{case_name}_input_ctx08.jpg"
            if input_image is None and image_path.is_file():
                input_image = relative(image_path, root)
            outputs[method.method_id] = {
                "video": relative(video_path, root) if video_path.is_file() else None,
                "json": relative(json_path, root) if json_path.is_file() else None,
            }
        cases.append(
            {
                "name": case_name,
                "input_image": input_image,
                **reference,
                "outputs": outputs,
            }
        )

    return {
        "title": "Wan DiT Block Ablation Comparison",
        "root": str(root),
        "num_cases": len(cases),
        "num_methods": len(methods),
        "methods": [
            {
                "id": method.method_id,
                "label": method.label,
                "model": method.model,
                "model_label": MODEL_LABELS[method.model],
                "mode": method.mode,
                "mode_label": MODE_LABELS[method.mode],
                "block_id": method.block_id,
                "group": method.group,
            }
            for method in methods
        ],
        "cases": cases,
    }


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wan DiT Block Ablation Comparison</title>
  <style>
    :root {
      --bg: #f4f6f5;
      --surface: #ffffff;
      --surface-alt: #eef3f0;
      --text: #17211c;
      --muted: #607069;
      --line: #ced8d2;
      --green: #146c43;
      --blue: #245f9e;
      --orange: #a64b18;
      --shadow: 0 8px 24px rgba(24, 48, 36, .08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font-family: Inter, "Noto Sans SC", Arial, sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 20;
      display: grid;
      grid-template-columns: minmax(280px, 1fr) minmax(360px, 2fr) auto;
      gap: 16px;
      align-items: center;
      padding: 14px 20px;
      color: #fff;
      background: #18382a;
      border-bottom: 1px solid #295440;
    }
    .title { min-width: 0; }
    .title h1 { margin: 0; font-size: 20px; line-height: 1.25; }
    .title p { margin: 3px 0 0; color: #bcd0c5; font-size: 12px; }
    .case-controls {
      display: grid;
      grid-template-columns: 40px minmax(220px, 1fr) 40px;
      gap: 8px;
    }
    .icon-button, .command {
      min-height: 38px;
      border: 1px solid #4f7462;
      color: #fff;
      background: #214a36;
      cursor: pointer;
    }
    .icon-button { width: 40px; font-size: 22px; }
    .icon-button:hover, .command:hover { background: #2b5d45; }
    select, input {
      width: 100%;
      min-width: 0;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fff;
      color: var(--text);
      padding: 8px 10px;
    }
    .toolbar {
      display: flex;
      gap: 8px;
      justify-content: flex-end;
    }
    .command { border-radius: 4px; padding: 8px 12px; }
    .page { max-width: 1900px; margin: 0 auto; padding: 18px 20px 40px; }
    .case-summary {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 250px;
      gap: 16px;
      align-items: start;
      padding-bottom: 16px;
      border-bottom: 1px solid var(--line);
    }
    .case-summary h2 {
      margin: 0;
      font-size: 19px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .case-meta { margin-top: 8px; color: var(--muted); font-size: 13px; }
    .search-wrap input { border-color: #9eb1a7; }
    .reference {
      display: grid;
      grid-template-columns: minmax(320px, 520px) minmax(0, 1fr);
      gap: 14px;
      margin-top: 16px;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      box-shadow: var(--shadow);
    }
    .reference-media { min-width: 0; }
    .reference-label {
      margin-bottom: 8px;
      color: var(--green);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .reference video {
      border-radius: 4px;
      border: 1px solid #18241e;
    }
    .prompt-panel {
      min-width: 0;
      padding: 4px 6px;
    }
    .prompt-panel h3 { margin: 0 0 10px; font-size: 14px; }
    .prompt-panel p {
      margin: 0;
      color: #34463d;
      font-size: 15px;
      line-height: 1.65;
      overflow-wrap: anywhere;
    }
    .groups { display: grid; gap: 22px; margin-top: 18px; }
    .group { min-width: 0; }
    .group-head {
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 10px;
    }
    .group-head h3 { margin: 0; font-size: 16px; }
    .count {
      color: var(--muted);
      background: var(--surface-alt);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 11px;
    }
    .method-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(285px, 1fr));
      gap: 12px;
    }
    .method {
      min-width: 0;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }
    .method[data-model="wan_lora"] { border-top: 3px solid var(--blue); }
    .method[data-model="xssc"] { border-top: 3px solid var(--green); }
    .method-head {
      min-height: 66px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
    }
    .method-name { margin: 0; font-size: 14px; line-height: 1.35; overflow-wrap: anywhere; }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 7px; }
    .tag { color: var(--muted); font-size: 11px; }
    .tag.model { color: var(--blue); font-weight: 700; }
    .method[data-model="xssc"] .tag.model { color: var(--green); }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #0b0f0d;
    }
    .method-footer {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: center;
      min-height: 38px;
      padding: 7px 10px;
      color: var(--muted);
      font-size: 11px;
    }
    .method-footer a { color: var(--blue); font-weight: 600; text-decoration: none; }
    .missing {
      display: grid;
      place-items: center;
      width: 100%;
      aspect-ratio: 16 / 9;
      color: var(--orange);
      background: #fff5ed;
      font-size: 13px;
    }
    @media (max-width: 900px) {
      .topbar { grid-template-columns: 1fr; }
      .toolbar { justify-content: flex-start; }
      .case-summary { grid-template-columns: 1fr; }
      .reference { grid-template-columns: 1fr; }
      .method-grid { grid-template-columns: 1fr; }
      .page { padding: 14px 12px 28px; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="title">
      <h1>Wan DiT Block Ablation</h1>
      <p id="datasetSummary">Loading comparison manifest...</p>
    </div>
    <div class="case-controls">
      <button class="icon-button" id="previousCase" title="Previous case" aria-label="Previous case">‹</button>
      <select id="caseSelect" aria-label="Select case"></select>
      <button class="icon-button" id="nextCase" title="Next case" aria-label="Next case">›</button>
    </div>
    <div class="toolbar">
      <button class="command" id="playVisible">Play all</button>
      <button class="command" id="pauseVisible">Pause all</button>
    </div>
  </header>
  <main class="page">
    <section class="case-summary">
      <div>
        <h2 id="caseName"></h2>
        <div class="case-meta" id="caseMeta"></div>
      </div>
      <div class="search-wrap">
        <input id="caseSearch" type="search" placeholder="Search cases" aria-label="Search cases">
      </div>
    </section>
    <section class="reference" id="reference"></section>
    <div class="groups" id="groups"></div>
  </main>
  <script>
    const state = { manifest: null, caseIndex: 0, filteredIndices: [] };
    const select = document.getElementById("caseSelect");
    const groupsRoot = document.getElementById("groups");
    const search = document.getElementById("caseSearch");

    function addText(parent, tag, className, value) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      node.textContent = value;
      parent.appendChild(node);
      return node;
    }

    function activeCaseIndex() {
      return state.filteredIndices[state.caseIndex] ?? 0;
    }

    function renderSelector() {
      select.replaceChildren();
      state.filteredIndices.forEach((manifestIndex, filteredIndex) => {
        const option = document.createElement("option");
        option.value = String(filteredIndex);
        option.textContent = `${manifestIndex + 1}. ${state.manifest.cases[manifestIndex].name}`;
        select.appendChild(option);
      });
      state.caseIndex = Math.min(state.caseIndex, Math.max(0, state.filteredIndices.length - 1));
      select.value = String(state.caseIndex);
    }

    function renderCase() {
      if (!state.filteredIndices.length) {
        document.getElementById("caseName").textContent = "No matching cases";
        document.getElementById("caseMeta").textContent = "";
        document.getElementById("reference").replaceChildren();
        groupsRoot.replaceChildren();
        return;
      }
      const manifestIndex = activeCaseIndex();
      const item = state.manifest.cases[manifestIndex];
      document.getElementById("caseName").textContent = item.name;
      document.getElementById("caseMeta").textContent =
        `Case ${manifestIndex + 1} / ${state.manifest.num_cases} · ${state.manifest.num_methods} methods`;
      select.value = String(state.caseIndex);
      history.replaceState(null, "", `#case=${manifestIndex + 1}`);
      groupsRoot.replaceChildren();
      const reference = document.getElementById("reference");
      reference.replaceChildren();
      const referenceMedia = document.createElement("div");
      referenceMedia.className = "reference-media";
      addText(referenceMedia, "div", "reference-label", "Ground truth / Source video");
      if (item.gt_video) {
        const gtVideo = document.createElement("video");
        gtVideo.controls = true;
        gtVideo.preload = "metadata";
        gtVideo.playsInline = true;
        gtVideo.src = `../${item.gt_video}`;
        referenceMedia.appendChild(gtVideo);
      } else {
        addText(referenceMedia, "div", "missing", "Missing GT video");
      }
      reference.appendChild(referenceMedia);
      const promptPanel = document.createElement("div");
      promptPanel.className = "prompt-panel";
      addText(promptPanel, "h3", "", "Prompt");
      addText(promptPanel, "p", "", item.prompt || "Prompt unavailable");
      reference.appendChild(promptPanel);

      const grouped = new Map();
      state.manifest.methods.forEach(method => {
        if (!grouped.has(method.group)) grouped.set(method.group, []);
        grouped.get(method.group).push(method);
      });

      grouped.forEach((methods, groupName) => {
        const section = document.createElement("section");
        section.className = "group";
        const heading = document.createElement("div");
        heading.className = "group-head";
        addText(heading, "h3", "", groupName);
        addText(heading, "span", "count", `${methods.length} methods`);
        section.appendChild(heading);
        const grid = document.createElement("div");
        grid.className = "method-grid";

        methods.forEach(method => {
          const output = item.outputs[method.id];
          const card = document.createElement("article");
          card.className = "method";
          card.dataset.model = method.model;
          const head = document.createElement("div");
          head.className = "method-head";
          addText(head, "h4", "method-name", method.label);
          const tags = document.createElement("div");
          tags.className = "tags";
          addText(tags, "span", "tag model", method.model_label);
          addText(tags, "span", "tag", method.mode_label);
          head.appendChild(tags);
          card.appendChild(head);

          if (output && output.video) {
            const video = document.createElement("video");
            video.controls = true;
            video.preload = "none";
            video.playsInline = true;
            video.src = `../${output.video}`;
            card.appendChild(video);
          } else {
            addText(card, "div", "missing", "Missing video");
          }

          const footer = document.createElement("div");
          footer.className = "method-footer";
          addText(footer, "span", "", method.id);
          if (output && output.json) {
            const link = document.createElement("a");
            link.href = `../${output.json}`;
            link.target = "_blank";
            link.rel = "noreferrer";
            link.textContent = "JSON";
            footer.appendChild(link);
          }
          card.appendChild(footer);
          grid.appendChild(card);
        });
        section.appendChild(grid);
        groupsRoot.appendChild(section);
      });
    }

    function moveCase(delta) {
      if (!state.filteredIndices.length) return;
      state.caseIndex = (state.caseIndex + delta + state.filteredIndices.length) % state.filteredIndices.length;
      renderCase();
    }

    function applySearch() {
      const query = search.value.trim().toLowerCase();
      state.filteredIndices = state.manifest.cases
        .map((item, index) => item.name.toLowerCase().includes(query) ? index : -1)
        .filter(index => index >= 0);
      state.caseIndex = 0;
      renderSelector();
      renderCase();
    }

    select.addEventListener("change", () => {
      state.caseIndex = Number(select.value);
      renderCase();
    });
    search.addEventListener("input", applySearch);
    document.getElementById("previousCase").addEventListener("click", () => moveCase(-1));
    document.getElementById("nextCase").addEventListener("click", () => moveCase(1));
    document.getElementById("playVisible").addEventListener("click", () => {
      document.querySelectorAll("video").forEach(video => video.play().catch(() => {}));
    });
    document.getElementById("pauseVisible").addEventListener("click", () => {
      document.querySelectorAll("video").forEach(video => video.pause());
    });
    document.addEventListener("keydown", event => {
      if (event.target.matches("input, select")) return;
      if (event.key === "ArrowLeft") moveCase(-1);
      if (event.key === "ArrowRight") moveCase(1);
    });

    fetch("./manifest.json")
      .then(response => response.json())
      .then(manifest => {
        state.manifest = manifest;
        state.filteredIndices = manifest.cases.map((_, index) => index);
        const hashMatch = location.hash.match(/case=(\\d+)/);
        if (hashMatch) state.caseIndex = Math.max(0, Math.min(manifest.num_cases - 1, Number(hashMatch[1]) - 1));
        document.getElementById("datasetSummary").textContent =
          `${manifest.num_cases} cases · ${manifest.num_methods} methods · grouped by shared input`;
        renderSelector();
        renderCase();
      });
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "_gallery"
    )
    methods = discover_methods(root)
    manifest = build_manifest(root, output_dir, methods)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "index.html").write_text(INDEX_HTML, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "num_cases": manifest["num_cases"],
                "num_methods": manifest["num_methods"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
