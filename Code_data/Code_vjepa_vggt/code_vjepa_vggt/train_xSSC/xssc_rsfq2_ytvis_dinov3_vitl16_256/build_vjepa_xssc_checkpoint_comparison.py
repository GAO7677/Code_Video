#!/usr/bin/env python3
"""Build a synchronized static-frame comparison from two xSSC viewer reports."""

from argparse import ArgumentParser
import html
import json
from pathlib import Path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--old-report", type=Path, required=True)
    parser.add_argument("--new-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--old-prefix", default="old")
    parser.add_argument("--new-prefix", default="new")
    parser.add_argument("--old-full-report", type=Path)
    parser.add_argument("--new-full-report", type=Path)
    parser.add_argument("--old-full-prefix", default="old-full")
    parser.add_argument("--new-full-prefix", default="new-full")
    return parser.parse_args()


def mean_metric(cases, source, metric):
    values = [
        case["metrics"][metric]
        for case in cases
        if case["source"] == source and case["metrics"].get(metric) is not None
    ]
    return sum(values) / len(values) if values else None


def prefixed(prefix, value):
    return f"{prefix}/{value}" if value else None


def build_report(
    old,
    new,
    old_prefix,
    new_prefix,
    old_full=None,
    new_full=None,
    old_full_prefix="old-full",
    new_full_prefix="new-full",
):
    old_cases = {case["case_id"]: case for case in old["cases"]}
    new_cases = {case["case_id"]: case for case in new["cases"]}
    old_full_cases = {
        case["source_key"]: case for case in (old_full or {}).get("cases", [])
    }
    new_full_cases = {
        case["source_key"]: case for case in (new_full or {}).get("cases", [])
    }
    if set(old_cases) != set(new_cases):
        raise RuntimeError(
            "case sets differ: "
            f"old-only={sorted(set(old_cases) - set(new_cases))}, "
            f"new-only={sorted(set(new_cases) - set(old_cases))}"
        )
    cases = []
    for old_case in old["cases"]:
        new_case = new_cases[old_case["case_id"]]
        if old_case["frames"] != new_case["frames"]:
            raise RuntimeError(
                f"frame count mismatch for {old_case['case_id']}: "
                f"{old_case['frames']} != {new_case['frames']}"
            )
        case = {
                "case_id": old_case["case_id"],
                "source": old_case["source"],
                "source_key": old_case["source_key"],
                "frames": old_case["frames"],
                "old_shape": old_case["processed_shape"],
                "new_shape": new_case["processed_shape"],
                "old_metrics": old_case["metrics"],
                "new_metrics": new_case["metrics"],
                "old_input": prefixed(
                    old_prefix, old_case["assets"]["original_pattern"]
                ),
                "old_prediction": prefixed(
                    old_prefix, old_case["assets"]["prediction_pattern"]
                ),
                "new_input": prefixed(
                    new_prefix, new_case["assets"]["original_pattern"]
                ),
                "new_prediction": prefixed(
                    new_prefix, new_case["assets"]["prediction_pattern"]
                ),
                "gt": prefixed(new_prefix, new_case["assets"].get("gt_pattern")),
            }
        old_full_case = old_full_cases.get(old_case["source_key"])
        new_full_case = new_full_cases.get(new_case["source_key"])
        if old_full_case is not None:
            case["old_full_prediction"] = prefixed(
                old_full_prefix, old_full_case["assets"]["prediction_pattern"]
            )
            case["old_full_input_policy"] = old_full_case.get("input_policy")
            case["old_full_attention_shapes"] = old_full_case.get("attention_shapes")
            case["old_full_slot_alignment"] = old_full_case.get("slot_alignment")
            case["old_full_similarity_to_t0"] = prefixed(
                old_full_prefix,
                old_full_case["assets"].get("slot_similarity_to_t0"),
            )
            case["old_full_similarity_adjacent"] = prefixed(
                old_full_prefix,
                old_full_case["assets"].get("slot_similarity_adjacent"),
            )
            case["old_full_similarity_data"] = prefixed(
                old_full_prefix,
                old_full_case["assets"].get("slot_similarity_data"),
            )
        if new_full_case is not None:
            case["new_full_prediction"] = prefixed(
                new_full_prefix, new_full_case["assets"]["prediction_pattern"]
            )
            case["new_full_input_policy"] = new_full_case.get("input_policy")
            case["new_full_attention_shapes"] = new_full_case.get("attention_shapes")
            case["new_full_slot_alignment"] = new_full_case.get("slot_alignment")
            case["new_full_similarity_to_t0"] = prefixed(
                new_full_prefix,
                new_full_case["assets"].get("slot_similarity_to_t0"),
            )
            case["new_full_similarity_adjacent"] = prefixed(
                new_full_prefix,
                new_full_case["assets"].get("slot_similarity_adjacent"),
            )
            case["new_full_similarity_data"] = prefixed(
                new_full_prefix,
                new_full_case["assets"].get("slot_similarity_data"),
            )
        cases.append(case)
    aggregates = {}
    for source in ("ytvis_hq_val", "movi_c_val"):
        aggregates[source] = {}
        for metric in ("ari_fg_diagnostic", "mean_best_overlap_diagnostic"):
            aggregates[source][metric] = {
                "old": mean_metric(old["cases"], source, metric),
                "new": mean_metric(new["cases"], source, metric),
            }
    same_model_family = (
        old.get("config") == new.get("config")
        and old.get("condition_mode") == new.get("condition_mode")
    )
    comparison_note = (
        "这是同一模型配置、相同 bbox 条件和相同预处理下的 checkpoint-only "
        "训练进展比较；case 与帧严格配对。"
        if same_model_family
        else (
            "这不是只替换权重的纯消融：两侧模型配置、条件初始化或预处理存在差异。"
            "页面同时展示两侧有效输入，避免掩盖这些变量。"
        )
    )
    if old_full_cases or new_full_cases:
        comparison_note += (
            " 全部 test_5 case 额外展示完整视频单次 forward；"
            "完整输入列不执行跨窗口 Hungarian slot 重编号。"
        )
    return {
        "title": (
            f"V-JEPA xSSC · step-{old['latest_complete_step']} vs "
            f"step-{new['latest_complete_step']}"
        ),
        "old": {
            "step": old["latest_complete_step"],
            "checkpoint": old["checkpoint"],
            "scope": old["checkpoint_scope"],
            "condition": old.get("condition_mode", "unconditioned"),
            "href": f"{old_prefix}/",
        },
        "new": {
            "step": new["latest_complete_step"],
            "checkpoint": new["checkpoint"],
            "scope": new["checkpoint_scope"],
            "condition": new.get("condition_mode", "unknown"),
            "href": f"{new_prefix}/",
        },
        "same_model_family": same_model_family,
        "has_full_forward": bool(old_full_cases or new_full_cases),
        "comparison_note": comparison_note,
        "aggregates": aggregates,
        "cases": cases,
    }


def build_html(report):
    payload = json.dumps(report, separators=(",", ":")).replace("</", "<\\/")
    title = html.escape(report["title"])
    return f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
*{{box-sizing:border-box}}:root{{color-scheme:dark;--paper:#0b1015;--ink:#eef4f8;--muted:#91a0aa;--rule:#33414d;--old:#62a8dc;--new:#f2a65a;--panel:#131b22}}body{{margin:0;background:var(--paper);color:var(--ink);font:14px "IBM Plex Sans","Noto Sans SC",system-ui,sans-serif}}header{{position:sticky;top:0;z-index:5;background:rgba(11,16,21,.97);border-bottom:1px solid var(--rule)}}.bar{{max-width:2400px;margin:auto;padding:10px 16px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}h1{{margin:0 auto 0 0;font:600 19px "IBM Plex Sans Condensed","Arial Narrow",sans-serif;letter-spacing:.02em}}button,select,input{{height:35px;border:1px solid #465661;border-radius:3px;background:#17212a;color:var(--ink);font:inherit}}button{{padding:0 11px;cursor:pointer}}button:focus-visible,select:focus-visible,input:focus-visible{{outline:2px solid var(--new);outline-offset:2px}}select{{padding:0 8px;max-width:min(760px,70vw)}}#slider{{min-width:240px;flex:0 1 400px;accent-color:var(--new)}}main{{max-width:2400px;margin:auto;padding:16px}}.lineage{{display:grid;grid-template-columns:1fr 64px 1fr;border:1px solid var(--rule);background:var(--panel);margin-bottom:14px}}.branch{{padding:14px 16px;min-width:0}}.branch.old{{border-top:3px solid var(--old)}}.branch.new{{border-top:3px solid var(--new)}}.branch b{{display:block;font:600 17px "IBM Plex Sans Condensed","Arial Narrow",sans-serif;margin-bottom:4px}}.branch p{{margin:3px 0;color:var(--muted);line-height:1.45;overflow-wrap:anywhere}}.arrow{{display:grid;place-items:center;color:#71818d;font-size:22px;border-inline:1px solid var(--rule)}}.warning{{border-left:4px solid var(--new);background:#211a13;color:#e3c5a5;padding:10px 13px;margin-bottom:14px;line-height:1.5}}.metrics{{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:8px;margin-bottom:14px}}.metric{{background:var(--panel);border:1px solid var(--rule);padding:10px 12px}}.metric small{{display:block;color:var(--muted);margin-bottom:4px}}.metric span{{font:600 15px ui-monospace,SFMono-Regular,Consolas,monospace}}.metric .a{{color:var(--old)}}.metric .b{{color:var(--new)}}.case-meta{{display:flex;gap:14px;overflow:auto;white-space:nowrap;color:var(--muted);padding:0 2px 12px}}.case-meta strong{{color:var(--ink)}}.compare{{display:grid;grid-template-columns:repeat(var(--panels),minmax(240px,1fr));gap:10px;min-width:calc(var(--panels) * 250px)}}.viewport{{overflow:auto}}figure{{margin:0}}figure img{{display:block;width:100%;height:min(61vh,720px);object-fit:contain;background:#030506;border:1px solid var(--rule)}}figcaption{{padding:7px 3px 0;line-height:1.4;color:var(--muted)}}figcaption b{{display:block;color:var(--ink)}}figure.old img{{border-top:3px solid var(--old)}}figure.new img{{border-top:3px solid var(--new)}}.slot-charts{{display:none;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:24px;padding-top:18px;border-top:1px solid var(--rule)}}.slot-charts h2{{grid-column:1/-1;margin:0;font:600 19px "IBM Plex Sans Condensed","Arial Narrow",sans-serif}}.slot-chart{{min-width:0;border:1px solid var(--rule);background:var(--panel);padding:10px}}.slot-chart.old{{border-top:3px solid var(--old)}}.slot-chart.new{{border-top:3px solid var(--new)}}.slot-chart img{{display:block;width:100%;height:auto;background:#fff}}.slot-chart p{{margin:8px 2px 0;color:var(--muted);line-height:1.5}}.slot-chart b{{color:var(--ink)}}.slot-chart a{{color:#8cc8ed}}.foot{{color:var(--muted);line-height:1.55;margin:15px 0 0}}code{{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;color:#b7d7ec}}@media(max-width:900px){{.lineage{{grid-template-columns:1fr}}.arrow{{display:none}}.metrics{{grid-template-columns:repeat(2,1fr)}}.slot-charts{{grid-template-columns:1fr}}.slot-charts h2{{grid-column:1}}}}@media(max-width:520px){{.metrics{{grid-template-columns:1fr}}}}
</style></head><body><header><div class="bar"><h1>{title}</h1><button onclick="location.href='../'">项目总览</button><button onclick="location.href=D.old.href">旧结果</button><button onclick="location.href=D.new.href">最新结果</button><select id="source"><option value="">全部来源</option><option value="ytvis_hq_val">YTVIS val</option><option value="movi_c_val">MOVi-C val</option><option value="test5">test_5</option></select><button id="prevCase">‹ case</button><select id="case"></select><button id="nextCase">case ›</button><button id="prevFrame">‹ frame</button><input id="slider" type="range" min="0" value="0"><button id="nextFrame">frame ›</button><span id="counter"></span></div></header><main><section class="lineage"><article class="branch old"><b>旧 · step-{report['old']['step']}</b><p>{html.escape(report['old']['scope'])}</p><p>{html.escape(report['old']['condition'])}</p></article><div class="arrow">→</div><article class="branch new"><b>新 · step-{report['new']['step']}</b><p>{html.escape(report['new']['scope'])}</p><p>{html.escape(report['new']['condition'])}</p></article></section><div class="warning">{html.escape(report['comparison_note'])}</div><section id="metrics" class="metrics"></section><div id="meta" class="case-meta"></div><div class="viewport"><section id="grid" class="compare"></section></div><section id="slotCharts" class="slot-charts"></section><p class="foot">所有内容均为静态视频帧图像。左右箭头键切换帧；下拉栏切换 case。slot embedding 曲线使用完整视频单次 forward 的 <code>output.slotz</code>；一个时间步对应两个连续原始帧。ARI-FG 与 mBO 只是在相同 5 个抽样 case 上计算的诊断均值，不代表完整验证集官方指标。</p></main><script>
const D={payload},source=document.getElementById('source'),sel=document.getElementById('case'),slider=document.getElementById('slider'),counter=document.getElementById('counter'),grid=document.getElementById('grid'),meta=document.getElementById('meta'),metrics=document.getElementById('metrics'),slotCharts=document.getElementById('slotCharts');let frame=0,visible=[];const esc=s=>String(s).replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));const fmt=v=>v===null||v===undefined?'—':Number(v).toFixed(4);const pat=(p,i)=>p.replace('{{frame}}',String(i).padStart(4,'0'));const current=()=>D.cases[Number(sel.value)];function fill(){{const prior=current()?.case_id;visible=D.cases.map((c,i)=>[c,i]).filter(([c])=>!source.value||c.source===source.value);sel.innerHTML='';for(const [c,i] of visible){{const o=document.createElement('option');o.value=i;o.textContent=`${{String(i+1).padStart(2,'0')}} | ${{c.source}} | ${{c.case_id}}`;if(c.case_id===prior)o.selected=true;sel.appendChild(o)}}render()}}function update(){{const c=current();frame=Math.max(0,Math.min(frame,c.frames-1));slider.value=frame;counter.textContent=`${{frame+1}} / ${{c.frames}}`;grid.querySelectorAll('img').forEach(img=>img.src=pat(img.dataset.pattern,frame))}}function renderMetrics(c){{const a=c.old_metrics,b=c.new_metrics;metrics.innerHTML=[['case ARI-FG',a.ari_fg_diagnostic,b.ari_fg_diagnostic],['case mBO',a.mean_best_overlap_diagnostic,b.mean_best_overlap_diagnostic],['旧有效 shape',c.old_shape.slice(1,3).join('×'),null],['新有效 shape',null,c.new_shape.slice(1,3).join('×')]].map(([k,x,y])=>`<div class="metric"><small>${{k}}</small><span class="a">${{typeof x==='string'?x:fmt(x)}}</span>${{y!==null?` <span>→</span> <span class="b">${{typeof y==='string'?y:fmt(y)}}</span>`:''}}</div>`).join('')}}function renderSlotCharts(){{const c=current();if(!c)return;const charts=[];if(c.old_full_similarity_to_t0)charts.push(['old',`step-${{D.old.step}} · 与 t0 对比`,c.old_full_similarity_to_t0,c.old_full_similarity_data]);if(c.old_full_similarity_adjacent)charts.push(['old',`step-${{D.old.step}} · 与相邻时间步对比`,c.old_full_similarity_adjacent,c.old_full_similarity_data]);if(c.new_full_similarity_to_t0)charts.push(['new',`step-${{D.new.step}} · 与 t0 对比`,c.new_full_similarity_to_t0,c.new_full_similarity_data]);if(c.new_full_similarity_adjacent)charts.push(['new',`step-${{D.new.step}} · 与相邻时间步对比`,c.new_full_similarity_adjacent,c.new_full_similarity_data]);slotCharts.style.display=charts.length?'grid':'none';slotCharts.innerHTML=charts.length?`<h2>完整视频 slot embedding 时间相似度 · t0 对应原始帧 f0–f1</h2>${{charts.map(([klass,label,img,data])=>`<article class="slot-chart ${{klass}}"><img src="${{img}}" alt="${{esc(label)}}"><p><b>${{esc(label)}}</b> · 颜色与 overlay slot 一致${{data?` · <a href="${{data}}" download>下载 embeddings/相似度 NPZ</a>`:''}}</p></article>`).join('')}}`:''}}function render(){{const c=current();if(!c)return;frame=0;slider.max=c.frames-1;meta.innerHTML=`<strong>${{esc(c.source)}}</strong><span>${{esc(c.case_id)}}</span><span>${{c.frames}} frames</span><span>${{esc(c.source_key)}}</span>`;renderMetrics(c);const panels=[['old','旧有效输入',c.old_input],['old',`旧 step-${{D.old.step}} · 10帧切窗 slots`,c.old_prediction]];if(c.old_full_prediction)panels.push(['old',`旧 step-${{D.old.step}} · 完整${{c.frames}}帧 slots`,c.old_full_prediction]);panels.push(['new','新有效输入',c.new_input],['new',`新 step-${{D.new.step}} · 10帧切窗 slots`,c.new_prediction]);if(c.new_full_prediction)panels.push(['new',`新 step-${{D.new.step}} · 完整${{c.frames}}帧 slots`,c.new_full_prediction]);if(c.gt)panels.push(['gt','GT instances',c.gt]);grid.style.setProperty('--panels',panels.length);grid.innerHTML=panels.map(([klass,label,p])=>`<figure class="${{klass}}"><img data-pattern="${{p}}"><figcaption><b>${{label}}</b>${{label.includes('· 完整')?'单次完整 forward · 无跨窗重编号':klass==='old'?esc(D.old.condition):klass==='new'?esc(D.new.condition):'validation annotation'}}</figcaption></figure>`).join('');renderSlotCharts();update()}}source.onchange=fill;sel.onchange=render;slider.oninput=()=>{{frame=Number(slider.value);update()}};document.getElementById('prevCase').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex-1+sel.options.length)%sel.options.length;render()}};document.getElementById('nextCase').onclick=()=>{{sel.selectedIndex=(sel.selectedIndex+1)%sel.options.length;render()}};document.getElementById('prevFrame').onclick=()=>{{frame--;update()}};document.getElementById('nextFrame').onclick=()=>{{frame++;update()}};document.addEventListener('keydown',e=>{{if(e.key==='ArrowLeft'){{frame--;update()}}if(e.key==='ArrowRight'){{frame++;update()}}}});fill();
</script></body></html>'''


def main():
    args = parse_args()
    old = json.loads(args.old_report.read_text())
    new = json.loads(args.new_report.read_text())
    old_full = json.loads(args.old_full_report.read_text()) if args.old_full_report else None
    new_full = json.loads(args.new_full_report.read_text()) if args.new_full_report else None
    if old_full and old_full.get("latest_complete_step") != old.get("latest_complete_step"):
        raise RuntimeError("old full-forward report checkpoint does not match old report")
    if new_full and new_full.get("latest_complete_step") != new.get("latest_complete_step"):
        raise RuntimeError("new full-forward report checkpoint does not match new report")
    report = build_report(
        old,
        new,
        args.old_prefix,
        args.new_prefix,
        old_full,
        new_full,
        args.old_full_prefix,
        args.new_full_prefix,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for report_path, prefix in (
        (args.old_full_report, args.old_full_prefix),
        (args.new_full_report, args.new_full_prefix),
    ):
        if report_path is None:
            continue
        link = args.output_dir / prefix
        target = report_path.resolve().parent
        if link.is_symlink() and link.resolve() == target:
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"refusing to replace existing full-forward link: {link}")
        link.symlink_to(target, target_is_directory=True)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "index.html").write_text(build_html(report))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "cases": len(report["cases"]),
                "old_step": report["old"]["step"],
                "new_step": report["new"]["step"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
