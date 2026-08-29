#!/usr/bin/env python3
"""Build a side-by-side video comparison for original, R002 and R003.

This creates only a small static HTML index; the videos remain in their
original experiment/sample directories and are never copied.
"""
from __future__ import annotations

import html
import json
import os
from pathlib import Path


STRICT_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819_strict")
R002_ROOT = STRICT_ROOT / "refine/R002_all_cases_distinct_texture_20260829"
R003_ROOT = STRICT_ROOT / "refine/R003_natural_common_textures_20260829"
OUTPUT_ROOT = STRICT_ROOT / "refine/comparisons/original_r002_r003_20260829"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative_url(path: Path) -> str:
    return os.path.relpath(path, OUTPUT_ROOT).replace(os.sep, "/")


def video_path(root: Path, case_id: str) -> Path:
    if root == STRICT_ROOT:
        return root / "samples" / case_id / "videos/rgb_cycles.mp4"
    return root / "cases" / case_id / "render/full/rgb_cycles.mp4"


def material_summary(report: dict, dynamic_names: list[str]) -> str:
    assignments = report.get("material_assignments", {})
    values = [f"{name}: {assignments.get(name, '—')}" for name in dynamic_names]
    return ", ".join(values)


def build() -> None:
    manifest_path = R002_ROOT / "case_selection.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    rows = []
    for raw in manifest_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        selection = json.loads(raw)
        case_id = selection["case_id"]
        original_video = video_path(STRICT_ROOT, case_id)
        r002_video = video_path(R002_ROOT, case_id)
        r003_video = video_path(R003_ROOT, case_id)
        if not (original_video.is_file() and r002_video.is_file() and r003_video.is_file()):
            continue
        original_report = load(STRICT_ROOT / "samples" / case_id / "videos/rgb_cycles.json")
        r002_selection = load(R002_ROOT / "cases" / case_id / "selection.json")
        r003_selection = load(R003_ROOT / "cases" / case_id / "selection.json")
        dynamic_names = list(r003_selection["selected"].get("dynamic_actor_names", []))
        rows.append({
            "case_id": case_id,
            "family": r003_selection["family_key"],
            "dynamic_names": dynamic_names,
            "original_material": material_summary(original_report, dynamic_names),
            "r002_material": r002_selection["selected"].get("selected_material", "—"),
            "r003_material": r003_selection["selected"].get("selected_material", "—"),
            "original": relative_url(original_video),
            "r002": relative_url(r002_video),
            "r003": relative_url(r003_video),
        })
    rows.sort(key=lambda row: (row["family"], row["case_id"]))
    if not rows:
        raise RuntimeError("no case has all three original/R002/R003 full videos")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        cards.append(
            "<section class='case-card'>"
            f"<div class='case-heading'><div><span class='eyebrow'>{html.escape(row['family'])}</span>"
            f"<h2>{html.escape(row['case_id'])}</h2><p>dynamic: {html.escape(', '.join(row['dynamic_names']))}</p></div>"
            "<span class='same-source'>same source case / same 90 frames</span></div>"
            "<div class='comparison-grid'>"
            f"<article><div class='method original'><b>最初版本</b><span>strict CYCLES</span></div>"
            f"<p class='material'>{html.escape(row['original_material'])}</p><video controls preload='metadata' src='{html.escape(row['original'])}'></video></article>"
            f"<article><div class='method r002'><b>R002</b><span>official refine / visible texture v3</span></div>"
            f"<p class='material'>{html.escape(row['r002_material'])}</p><video controls preload='metadata' src='{html.escape(row['r002'])}'></video></article>"
            f"<article><div class='method r003'><b>R003</b><span>natural semantic texture / reference</span></div>"
            f"<p class='material'>{html.escape(row['r003_material'])}</p><video controls preload='metadata' src='{html.escape(row['r003'])}'></video></article>"
            "</div></section>"
        )

    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>原始 strict / R002 / R003 对比</title>
<style>
:root{{--ink:#172027;--muted:#65747c;--line:#d8e1e3;--paper:#f5f8f8;--teal:#1c6870;--gold:#a87928;--green:#3c7655}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:1680px;margin:0 auto;padding:34px 28px 64px}} header{{border-bottom:1px solid var(--line);padding-bottom:24px;margin-bottom:28px}}
.eyebrow{{font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}}
h1{{font-size:clamp(27px,4vw,48px);line-height:1.04;margin:10px 0 12px;max-width:860px}} header p{{max-width:920px;color:var(--muted);line-height:1.6;margin:0}}
.legend{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}} .legend span{{padding:8px 12px;border-radius:999px;font-size:13px;font-weight:650}}
.legend .original{{background:#e4edf0;color:#31575f}} .legend .r002{{background:#f5e8cb;color:#805a1b}} .legend .r003{{background:#dceee3;color:#286042}}
.note{{margin:20px 0 30px;padding:15px 18px;background:white;border-left:4px solid var(--teal);box-shadow:0 3px 14px #173a410d;color:#42545c;line-height:1.6}}
.case-card{{background:white;border:1px solid var(--line);margin:0 0 24px;box-shadow:0 4px 18px #173a410b}} .case-heading{{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;padding:17px 19px 13px;border-bottom:1px solid var(--line)}}
.case-heading h2{{font-size:19px;margin:5px 0 3px}} .case-heading p{{color:var(--muted);font-size:13px;margin:0}} .same-source{{font:600 11px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);white-space:nowrap;padding-top:5px}}
.comparison-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1px;background:var(--line)}} article{{background:#fff;padding:14px}} .method{{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:7px}} .method b{{font-size:16px}} .method span{{font:600 10px ui-monospace,SFMono-Regular,Consolas,monospace;text-align:right;color:var(--muted)}}
.method.original b{{color:#31575f}} .method.r002 b{{color:var(--gold)}} .method.r003 b{{color:var(--green)}} .material{{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;color:var(--muted);min-height:30px;margin:0 0 10px;line-height:1.35}} video{{display:block;width:100%;background:#10191c;aspect-ratio:16/9;object-fit:contain}}
footer{{color:var(--muted);font-size:12px;margin-top:30px}} code{{font:inherit;color:var(--ink)}} @media(max-width:900px){{.comparison-grid{{grid-template-columns:1fr}}.case-heading{{display:block}}.same-source{{display:block;margin-top:8px}}.method span{{font-size:9px}}}}
</style></head><body><main>
<header><div class="eyebrow">STRICT CYCLES / VERSION COMPARISON</div>
<h1>最初版本、R002、R003 的同 case 对比</h1>
<p>每一行固定为同一个 source case，三列使用相同的 90 帧、896×512、30 FPS CYCLES 轨迹渲染。三者只比较 RGB 外观，不改变相机、几何物理、轨迹或 GT。</p>
<div class="legend"><span class="original">最初版本 · strict CYCLES</span><span class="r002">R002 · official refine / visible texture v3</span><span class="r003">R003 · natural semantic texture / reference</span></div></header>
<div class="note">当前共有 <b>{len(rows)}</b> 个 case 三个版本均已有完整视频。R002 是当前正式 refine 版本；其中 <code>difficulty_l2_f11_h030_sr048</code> 按指定采用 R003 的篮球纹理，其余 R002 case 保持不变。R002 的老版本仍保留在其目录内；R004 清晰边缘实验已停止，不纳入当前版本对比。</div>
{"".join(cards)}
<footer>生成脚本：<code>scripts/build_refine_version_comparison.py</code> · 页面为手动刷新静态页。</footer>
</main></body></html>"""
    (OUTPUT_ROOT / "index.html").write_text(page, encoding="utf-8")
    (OUTPUT_ROOT / "README.md").write_text(
        f"""# 原始 strict / R002 / R003 对比

当前页面展示 {len(rows)} 个同时具备三种完整 90 帧视频的相同 source case。视频不复制，页面使用原目录相对路径引用。

- 最初版本：strict samples/*/videos/rgb_cycles.mp4
- R002：当前正式 refine 版本；主体为 visible_texture_v3_family_consistent，F11/h030 case 使用指定的 R003 篮球纹理
- R003：自然常见材质版本；参考版本

重新生成：`python3 scripts/build_refine_version_comparison.py`。
""",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_ROOT / "index.html"),
        "shared_complete_cases": len(rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    build()
