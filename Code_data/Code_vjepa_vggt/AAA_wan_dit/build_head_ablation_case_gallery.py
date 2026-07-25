#!/usr/bin/env python3
"""Build one comparison page per case for Block 17 head ablations."""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


DATA_ROOT = Path("/data/gaoya")
DEFAULT_INPUT_LIST = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_head_ablation/"
    "test5_first5/_run/input_first5_unique.txt"
)
DEFAULT_BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_block17_self_attention/"
    "test5_first5/generated"
)
DEFAULT_ALL_HEAD_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_ablation/test5_first5"
)
DEFAULT_HEAD_ROOT = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_head_ablation/test5_first5"
)
DEFAULT_OUTPUT_DIR = DEFAULT_HEAD_ROOT / "_gallery"

MODEL_LABELS = {
    "wan_lora": "Wan+LoRA",
    "xssc": "Wan+xSSC",
    "physrvg": "PhysRVG",
}
MODEL_ORDER = ("wan_lora", "xssc", "physrvg")
HEADS = (1, 2, 3, 18)


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    detail: str


VARIANTS = (
    Variant("baseline", "Baseline", "No output ablation"),
    Variant(
        "all_heads",
        "All heads = 0",
        "Block 17 self-attention output is zero",
    ),
    *(
        Variant(
            f"head{head:02d}",
            f"Head {head:02d} = 0",
            "Selected head is zero before output projection",
        )
        for head in HEADS
    ),
)


@dataclass(frozen=True)
class Case:
    name: str
    source_json: Path
    prompt: str
    source_video: Path
    context_video: Path
    page_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--all-head-root", type=Path, default=DEFAULT_ALL_HEAD_ROOT)
    parser.add_argument("--head-root", type=Path, default=DEFAULT_HEAD_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def require_path(payload: dict[str, object], key: str, source: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{source}: missing string field {key!r}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def url_for(path: Path) -> str:
    relative = path.resolve().relative_to(DATA_ROOT.resolve()).as_posix()
    return "/" + quote(relative, safe="/")


def safe_page_name(case_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", case_name).strip("._")
    if not normalized:
        raise ValueError(f"cannot make a page name from {case_name!r}")
    return f"{normalized}.html"


def load_cases(input_list: Path) -> list[Case]:
    seen: set[Path] = set()
    cases: list[Case] = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        source_json = Path(line.strip()).expanduser().resolve()
        if source_json in seen:
            continue
        seen.add(source_json)
        payload = load_json(source_json)
        prompt = payload.get("input_caption")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"{source_json}: missing input_caption")
        case_name = source_json.stem
        cases.append(
            Case(
                name=case_name,
                source_json=source_json,
                prompt=prompt,
                source_video=require_path(payload, "source_video", source_json),
                context_video=require_path(payload, "input_video", source_json),
                page_name=safe_page_name(case_name),
            )
        )
    if len(cases) != 5:
        raise ValueError(f"expected five unique cases, found {len(cases)}")
    return cases


def result_paths(
    case: Case,
    baseline_root: Path,
    all_head_root: Path,
    head_root: Path,
) -> dict[str, dict[str, Path]]:
    name = f"{case.name}.mp4"
    paths: dict[str, dict[str, Path]] = {
        "wan_lora": {
            "baseline": baseline_root / "wan_lora" / name,
            "all_heads": all_head_root
            / "wan_lora"
            / "self_attn_zero_block17"
            / name,
        },
        "xssc": {
            "baseline": baseline_root / "xssc" / "results" / name,
            "all_heads": all_head_root
            / "xssc"
            / "self_attn_zero_block17"
            / "results"
            / name,
        },
        "physrvg": {
            "baseline": baseline_root
            / "physrvg"
            / "input_first5_unique"
            / "physRVG_steps40_512x896_08_49f"
            / name,
            "all_heads": all_head_root
            / "PhyRVG"
            / "self_attn_zero_block17"
            / "input_first5_unique"
            / "physRVG_steps40_512x896_08_49f"
            / name,
        },
    }
    for head in HEADS:
        tag = f"self_attn_head_zero_block17_head{head:02d}"
        paths["wan_lora"][f"head{head:02d}"] = (
            head_root / "wan_lora" / tag / name
        )
        paths["xssc"][f"head{head:02d}"] = (
            head_root / "xssc" / tag / "results" / name
        )
        paths["physrvg"][f"head{head:02d}"] = (
            head_root
            / "PhyRVG"
            / tag
            / "input_first5_unique"
            / "physRVG_steps40_512x896_08_49f"
            / name
        )
    for model_paths in paths.values():
        for path in model_paths.values():
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(path)
    return paths


def page_css() -> str:
    return """
    :root {
      --bg: #eef1ef;
      --surface: #ffffff;
      --text: #18201c;
      --muted: #637069;
      --line: #cbd3ce;
      --nav: #1f3028;
      --wan: #285f99;
      --xssc: #24734f;
      --phys: #a65325;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--text);
      background: var(--bg);
      font: 14px Arial, sans-serif;
      letter-spacing: 0;
    }
    a { color: inherit; }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 5;
      display: grid;
      grid-template-columns: minmax(240px, 1fr) auto auto;
      gap: 12px;
      align-items: center;
      padding: 11px 18px;
      color: #fff;
      background: var(--nav);
      border-bottom: 1px solid #3b5146;
    }
    .topbar h1 { margin: 0; font-size: 18px; overflow-wrap: anywhere; }
    .case-nav, .commands { display: flex; gap: 7px; align-items: center; }
    .topbar a, button {
      min-height: 36px;
      padding: 8px 11px;
      border: 1px solid #587164;
      border-radius: 4px;
      color: #fff;
      background: #2b493a;
      text-decoration: none;
      cursor: pointer;
    }
    .topbar a:hover, button:hover { background: #365b48; }
    main { max-width: 2400px; margin: auto; padding: 18px 18px 40px; }
    .case-meta {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) minmax(360px, 640px);
      gap: 18px;
      align-items: start;
    }
    .case-meta h2 { margin: 0; font-size: 20px; overflow-wrap: anywhere; }
    .prompt { margin: 8px 0 0; color: var(--muted); line-height: 1.55; }
    .source-json { margin-top: 7px; color: #7b8580; font-size: 11px; overflow-wrap: anywhere; }
    .references {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }
    .reference-label, .video-label {
      margin-bottom: 5px;
      font-size: 12px;
      font-weight: 700;
    }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: contain;
      background: #080b09;
    }
    .model-section {
      margin-top: 24px;
      padding-top: 12px;
      border-top: 3px solid var(--wan);
    }
    .model-section.xssc { border-top-color: var(--xssc); }
    .model-section.physrvg { border-top-color: var(--phys); }
    .model-section h3 { margin: 0 0 10px; font-size: 18px; }
    .video-grid {
      display: grid;
      grid-template-columns: repeat(6, minmax(250px, 1fr));
      gap: 8px;
      overflow-x: auto;
      padding-bottom: 5px;
    }
    .video-item {
      min-width: 250px;
      padding: 7px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
    }
    .video-detail {
      min-height: 28px;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 11px;
      line-height: 1.3;
    }
    .legend {
      margin-top: 20px;
      padding: 12px 14px;
      color: var(--muted);
      background: #f8faf9;
      border-left: 3px solid #7c8982;
      line-height: 1.5;
    }
    .index-list {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 10px;
      margin-top: 18px;
    }
    .case-link {
      display: block;
      padding: 14px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 6px;
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .case-link:hover { border-color: #658274; }
    @media (max-width: 900px) {
      .topbar { grid-template-columns: 1fr; }
      .case-meta { grid-template-columns: 1fr; }
      .references { grid-template-columns: 1fr; }
      main { padding: 14px 10px 30px; }
    }
    """


def video_html(path: Path, label: str, detail: str) -> str:
    return (
        '<article class="video-item">'
        f'<div class="video-label">{html.escape(label)}</div>'
        f'<div class="video-detail">{html.escape(detail)}</div>'
        f'<video controls preload="metadata" playsinline src="{url_for(path)}"></video>'
        "</article>"
    )


def render_case_page(
    case: Case,
    cases: list[Case],
    index: int,
    outputs: dict[str, dict[str, Path]],
) -> str:
    previous_case = cases[(index - 1) % len(cases)]
    next_case = cases[(index + 1) % len(cases)]
    model_sections: list[str] = []
    for model in MODEL_ORDER:
        videos = "".join(
            video_html(outputs[model][variant.key], variant.label, variant.detail)
            for variant in VARIANTS
        )
        model_sections.append(
            f'<section class="model-section {model}">'
            f"<h3>{MODEL_LABELS[model]}</h3>"
            f'<div class="video-grid">{videos}</div>'
            "</section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(case.name)} - Block 17 Head Ablation</title>
  <style>{page_css()}</style>
</head>
<body>
  <header class="topbar">
    <h1>{html.escape(case.name)}</h1>
    <nav class="case-nav">
      <a href="{previous_case.page_name}" title="Previous case">&lt;</a>
      <a href="index.html">All cases</a>
      <a href="{next_case.page_name}" title="Next case">&gt;</a>
    </nav>
    <div class="commands">
      <button type="button" id="playAll">Play all</button>
      <button type="button" id="pauseAll">Pause all</button>
    </div>
  </header>
  <main>
    <section class="case-meta">
      <div>
        <h2>Case {index + 1} / {len(cases)}</h2>
        <p class="prompt">{html.escape(case.prompt)}</p>
        <div class="source-json">{html.escape(str(case.source_json))}</div>
      </div>
      <div class="references">
        <div>
          <div class="reference-label">GT full video</div>
          <video controls preload="metadata" playsinline src="{url_for(case.source_video)}"></video>
        </div>
        <div>
          <div class="reference-label">Input context (8 frames)</div>
          <video controls preload="metadata" playsinline src="{url_for(case.context_video)}"></video>
        </div>
      </div>
    </section>
    {''.join(model_sections)}
    <div class="legend">
      Baseline videos come from the capture-only attention run, which records
      attention without changing the model output. "All heads = 0" zeros the
      complete Block 17 self-attention output. Per-head variants zero only the
      selected 128-dimensional head before the shared output projection.
    </div>
  </main>
  <script>
    const videos = () => Array.from(document.querySelectorAll("video"));
    document.getElementById("playAll").addEventListener("click", () => {{
      videos().forEach(video => video.play().catch(() => {{}}));
    }});
    document.getElementById("pauseAll").addEventListener("click", () => {{
      videos().forEach(video => video.pause());
    }});
  </script>
</body>
</html>
"""


def render_index(cases: list[Case]) -> str:
    links = "".join(
        f'<a class="case-link" href="{case.page_name}">'
        f"<strong>Case {index + 1}</strong><br>{html.escape(case.name)}"
        "</a>"
        for index, case in enumerate(cases)
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Block 17 Per-Head Ablation Gallery</title>
  <style>{page_css()}</style>
</head>
<body>
  <header class="topbar">
    <h1>Block 17 Per-Head Ablation Gallery</h1>
    <div>5 cases / 3 models / 6 variants</div>
    <div></div>
  </header>
  <main>
    <h2>Case pages</h2>
    <p class="prompt">
      Each page compares GT, context, baseline, all-head self-attention
      ablation, and Head 01/02/03/18 ablations.
    </p>
    <div class="index-list">{links}</div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input_list)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest_cases: list[dict[str, object]] = []
    for index, case in enumerate(cases):
        outputs = result_paths(
            case,
            args.baseline_root,
            args.all_head_root,
            args.head_root,
        )
        page = render_case_page(case, cases, index, outputs)
        (args.output_dir / case.page_name).write_text(page, encoding="utf-8")
        manifest_cases.append(
            {
                "name": case.name,
                "page": case.page_name,
                "source_json": str(case.source_json),
                "prompt": case.prompt,
                "source_video": str(case.source_video),
                "context_video": str(case.context_video),
                "outputs": {
                    model: {key: str(path) for key, path in model_paths.items()}
                    for model, model_paths in outputs.items()
                },
            }
        )

    (args.output_dir / "index.html").write_text(
        render_index(cases), encoding="utf-8"
    )
    manifest = {
        "title": "Block 17 Per-Head Ablation Gallery",
        "num_cases": len(cases),
        "models": list(MODEL_ORDER),
        "variants": [
            {"key": item.key, "label": item.label, "detail": item.detail}
            for item in VARIANTS
        ],
        "expected_result_videos": len(cases) * len(MODEL_ORDER) * len(VARIANTS),
        "cases": manifest_cases,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "cases": len(cases),
                "result_videos": manifest["expected_result_videos"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
