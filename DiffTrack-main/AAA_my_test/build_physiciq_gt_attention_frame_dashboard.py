#!/usr/bin/env python3
"""Build a static-frame dashboard aligned to the Wan latent timeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/physiciq_gt_attention_transfer_l23_s39"
)
DEFAULT_GENERATED = Path(
    "/data/gaoya/agent-data/outputs/physiciq_selected_three_model_qk"
)
MODELS = (
    ("gt", "GT source"),
    ("stage1b", "Stage1b step-004000"),
    ("lora", "LoRA step-000500"),
    ("baseline", "Wan2.2 baseline"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--generated-root", type=Path, default=DEFAULT_GENERATED)
    return parser.parse_args()


def track_video(root: Path, model: str, source: str, case: str, region: str) -> Path:
    if model == "gt":
        variant = "gt_whole_stretch" if source == "whole" else "gt_framewise_stretch"
        return root / variant / "cases" / case / "regions" / region / "tracks_qk_L23_S039.mp4"
    return (
        root
        / "comparisons"
        / model
        / "cases"
        / case
        / "regions"
        / region
        / f"tracks_{source}_vs_cotracker.mp4"
    )


def heatmap_montage(
    root: Path, generated_root: Path, model: str, source: str, case: str, region: str
) -> Path:
    if model == "gt":
        variant = "gt_whole_stretch" if source == "whole" else "gt_framewise_stretch"
        return root / variant / "cases" / case / "regions" / region / "heatmap_qk_L23_S039.png"
    if source == "own":
        return (
            generated_root
            / model
            / "cases"
            / case
            / "regions"
            / region
            / "heatmap_qk_L23_S039.png"
        )
    return (
        root
        / "transferred"
        / source
        / model
        / "cases"
        / case
        / "regions"
        / region
        / "heatmap_gt_qk_L23_S039.png"
    )


def export_track_frames(video_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    index = 1
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.imwrite(str(output_dir / f"latent_{index:02d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        index += 1
    capture.release()
    if index != 7:
        raise RuntimeError(f"expected six latent frames in {video_path}, got {index - 1}")


def export_heatmap_frames(montage_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    montage = cv2.imread(str(montage_path), cv2.IMREAD_COLOR)
    if montage is None:
        raise RuntimeError(f"cannot read heatmap montage: {montage_path}")
    height, width = montage.shape[:2]
    if width % 5:
        raise RuntimeError(f"expected five equal heatmap panels in {montage_path}: {width}x{height}")
    panel_width = width // 5
    for offset, latent in enumerate(range(2, 7)):
        panel = montage[:, offset * panel_width : (offset + 1) * panel_width]
        cv2.imwrite(
            str(output_dir / f"latent_{latent:02d}.jpg"),
            panel,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


def collect_cases(generated_root: Path) -> list[dict]:
    cases = []
    for case_dir in sorted((generated_root / "stage1b" / "cases").iterdir()):
        manifest_path = case_dir / "manifest.json"
        if not (case_dir / "complete.json").is_file() or not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases.append(
            {
                "case_key": case_dir.name,
                "label": case_dir.name.removeprefix("case_physiciq_"),
                "regions": [item["region_name"] for item in manifest["query_regions"]],
            }
        )
    return cases


HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GT attention latent frames</title><style>
:root{--paper:#eee9dc;--ink:#17211e;--card:#fffdf8;--line:#b8b09f;--rust:#b64a31;--teal:#176654;--muted:#65706b}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 4% 0,#d7764b38,transparent 36rem),radial-gradient(circle at 96% 4%,#4b9a8038,transparent 36rem),var(--paper);color:var(--ink);font-family:"Trebuchet MS","Noto Sans CJK SC",sans-serif}main{width:min(1900px,calc(100% - 28px));margin:auto;padding:28px 0 60px}h1,h2{font-family:Georgia,"Noto Serif CJK SC",serif;margin:0}h1{font-size:clamp(40px,5vw,76px);line-height:.93;letter-spacing:-.04em}.eyebrow{color:var(--rust);font-size:12px;font-weight:900;letter-spacing:.15em}.lead{max-width:1120px;color:var(--muted);line-height:1.6}.controls{display:grid;grid-template-columns:2fr 1fr 1.2fr 1fr;gap:10px;margin:20px 0}label{font-size:11px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}select{display:block;width:100%;margin-top:5px;padding:10px;border:1px solid var(--ink);background:var(--card);font-weight:800}.board{overflow-x:auto;border:1px solid var(--line);background:#d8d2c4;padding:8px}.timeline{display:grid;gap:7px;min-width:1500px}.head,.model-row{display:grid;grid-template-columns:175px repeat(var(--count),minmax(215px,1fr));gap:7px}.corner,.time-label,.model-label,.tile{background:var(--card);border:1px solid var(--line)}.corner,.time-label{padding:9px;font-weight:900}.time-label{text-align:center}.time-label small{display:block;color:var(--muted);margin-top:3px}.model-label{display:flex;align-items:center;padding:13px;font:700 18px/1.2 Georgia}.tile{margin:0;min-width:0;background:#07100d}.tile img{display:block;width:100%;aspect-ratio:7/4;object-fit:contain;cursor:zoom-in}.tile figcaption{padding:6px 8px;color:#cbd5d0;font-size:10px}.note{margin-top:14px;padding:13px;background:var(--card);border:1px solid var(--line);line-height:1.55}.legend{color:var(--teal);font-weight:900}@media(max-width:760px){.controls{grid-template-columns:1fr}.head,.model-row{grid-template-columns:130px repeat(var(--count),230px)}.timeline{min-width:max-content}}
</style></head><body><main><div class="eyebrow">WAN2.2 · LAYER 23 · STEP 39 · LATENT TIMELINE</div><h1>Attention tracks<br>frame by frame</h1><p class="lead">列严格对应 Wan latent 时间轴，行对应 GT 与三个生成模型。轨迹视图显示 latent 1–6；attention heatmap 从第一个 future latent 开始，显示 latent 2–6。点击任意帧查看原图。</p><section class="controls"><label>Case<select id="case"></select></label><label>Region<select id="region"></select></label><label>Attention source<select id="source"><option value="own">Own Q/K</option><option value="framewise">GT frame-wise VAE</option><option value="whole">GT whole-video VAE</option></select></label><label>View<select id="view"><option value="tracks">Track overlay frames</option><option value="heatmaps">Attention heatmap frames</option></select></label></section><div class="board"><div class="timeline" id="timeline"></div></div><div class="note"><span class="legend">轨迹图：</span>圆点为当前视频 CoTracker，方框为 Q/K argmax；轨迹只连接 latent anchors。GT 行的 Own Q/K 使用 frame-wise 版本。<br><span class="legend">坐标：</span>Stage1b 使用 stretch GT attention，LoRA 与 baseline 使用 cover-crop GT attention。</div></main><script id="payload" type="application/json">__PAYLOAD__</script><script>
const data=JSON.parse(document.getElementById('payload').textContent),caseEl=document.getElementById('case'),regionEl=document.getElementById('region'),sourceEl=document.getElementById('source'),viewEl=document.getElementById('view'),timeline=document.getElementById('timeline');const models=[['gt','GT source'],['stage1b','Stage1b step-004000'],['lora','LoRA step-000500'],['baseline','Wan2.2 baseline']];for(const c of data.cases){const o=document.createElement('option');o.value=c.case_key;o.textContent=c.label;caseEl.append(o)}function current(){return data.cases.find(c=>c.case_key===caseEl.value)}function setRegions(){regionEl.innerHTML=current().regions.map(r=>`<option value="${r}">${r}</option>`).join('')}function effectiveSource(model,source){return model==='gt'&&source==='own'?'framewise':source}function render(){const c=current().case_key,r=regionEl.value,s=sourceEl.value,v=viewEl.value,latents=v==='tracks'?[1,2,3,4,5,6]:[2,3,4,5,6],pixels={1:4,2:8,3:12,4:16,5:20,6:24};timeline.style.setProperty('--count',latents.length);const head=`<div class="head"><div class="corner">Model / latent</div>${latents.map(l=>`<div class="time-label">latent ${l}<small>pixel frame ${pixels[l]}</small></div>`).join('')}</div>`;const rows=models.map(([model,label])=>{const source=effectiveSource(model,s);return `<div class="model-row"><div class="model-label">${label}</div>${latents.map(l=>{const src=`frame_assets/${v}/${source}/${model}/${c}/${r}/latent_${String(l).padStart(2,'0')}.jpg`;return `<figure class="tile"><a href="${src}" target="_blank"><img loading="lazy" src="${src}"></a><figcaption>latent ${l} · pixel ${pixels[l]}</figcaption></figure>`}).join('')}</div>`}).join('');timeline.innerHTML=head+rows}caseEl.addEventListener('change',()=>{setRegions();render()});for(const el of [regionEl,sourceEl,viewEl])el.addEventListener('change',render);setRegions();render();
</script></body></html>'''


def main() -> None:
    args = parse_args()
    root = args.result_root.resolve()
    generated_root = args.generated_root.resolve()
    dashboard = root / "dashboard"
    assets = dashboard / "frame_assets"
    cases = collect_cases(generated_root)
    for case in cases:
        case_key = case["case_key"]
        for region in case["regions"]:
            for source in ("own", "framewise", "whole"):
                for model, _ in MODELS:
                    effective_source = "framewise" if model == "gt" and source == "own" else source
                    export_track_frames(
                        track_video(root, model, effective_source, case_key, region),
                        assets / "tracks" / effective_source / model / case_key / region,
                    )
                    export_heatmap_frames(
                        heatmap_montage(
                            root, generated_root, model, effective_source, case_key, region
                        ),
                        assets / "heatmaps" / effective_source / model / case_key / region,
                    )
    payload = json.dumps({"cases": cases}, ensure_ascii=False).replace("</", "<\\/")
    (dashboard / "index.html").write_text(
        HTML.replace("__PAYLOAD__", payload), encoding="utf-8"
    )
    print(f"Built {dashboard / 'index.html'} with {len(cases)} cases")


if __name__ == "__main__":
    main()
