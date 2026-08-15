#!/usr/bin/env python3
"""Build an auto-refreshing static page for the 30-case six-checkpoint matrix."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

def read(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); args=p.parse_args()
    c=read(args.config,{}) ; root=Path(c["output_root"]); manifest=read(Path(c["cases_manifest"]),{"cases":[]}); video=read(root/"video_status.json",{"state":"queued","entries":{}}); loss=read(root/"loss_status.json",{"state":"queued","entries":{}})
    cases=manifest.get("cases",[]); entries=c.get("entries",[])
    site=root/"hub"; media=site/"media"; media.mkdir(parents=True,exist_ok=True)
    rows=[]; ready_v=ready_l=0
    for case in cases:
        panels=[]
        for e in entries:
            vroot=root/"videos"/e["entry_id"]
            candidates=list(vroot.glob(f"{case['case_id']}*.mp4"))+list(vroot.glob(f"*/{case['case_id']}*.mp4"))
            v=next(iter(candidates),None)
            l=read(root/"losses"/f"{e['entry_id']}.json",{})
            rec=next((x for x in l.get("cases",[]) if x.get("case_id")==case["case_id"]),None)
            if v and v.is_file():
                target=media/f"{case['case_id']}__{e['entry_id']}.mp4"
                if target.is_symlink(): target.unlink()
                elif target.exists(): target.unlink()
                target.symlink_to(v.resolve()); vhtml=f'<video controls preload="none" src="media/{html.escape(target.name)}"></video>'; ready_v+=1
            else: vhtml='<div class="pending">视频待完成</div>'
            if rec:
                m=rec.get("metrics",{}); loss_html=f"main {float(rec.get('loss_main',0)):.6f}"; 
                if 'train/loss_xssc' in m: loss_html+=f" · xSSC {float(m['train/loss_xssc']):.6f}"
                if 'train/loss_vjepa' in m: loss_html+=f" · V-JEPA {float(m['train/loss_vjepa']):.6f}"
                if 'train/loss_total' in m: loss_html+=f" · total {float(m['train/loss_total']):.6f}"
                ready_l+=1
            else: loss_html='loss 待完成'
            panels.append(f'<article class="panel"><h3>{html.escape(e["method_label"])} · {html.escape(e["version"])} · step-{int(e["step"]):04d}</h3>{vhtml}<p>{html.escape(loss_html)}</p></article>')
        rows.append(f'<section class="case"><h2>{html.escape(case["case_id"])}</h2><p>{html.escape(case["prompt"])}</p><div class="grid">{"".join(panels)}</div></section>')
    expected=len(cases)*len(entries)
    refresh='<meta http-equiv="refresh" content="60">' if ready_v<expected or ready_l<expected else ''
    status=f"视频 {ready_v}/{expected} · loss {ready_l}/{expected} · video={html.escape(str(video.get('state')))} · loss={html.escape(str(loss.get('state')))}"
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>30-case validation · Full-SA experiments</title><style>body{{margin:0;background:#10161d;color:#edf4fa;font:14px system-ui,sans-serif}}main{{max-width:1900px;margin:auto;padding:24px}}h1{{font-size:32px}}.summary,.case,.panel{{background:#18232d;border:1px solid #304454;border-radius:12px;padding:14px}}.summary{{position:sticky;top:0;z-index:2}}.case{{margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(6,minmax(200px,1fr));gap:8px}}.panel{{padding:8px}}h3{{font-size:12px;min-height:34px;margin:0 0 6px}}video{{width:100%;aspect-ratio:896/512;background:#000}}p{{color:#a9bac8;font-size:12px}}.pending{{aspect-ratio:896/512;display:grid;place-items:center;background:#263542;color:#8ca0af;font-size:12px}}@media(max-width:1200px){{.grid{{grid-template-columns:repeat(3,minmax(200px,1fr))}}}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style></head><body><main><div class="summary"><h1>30-case train validation · 6 checkpoints</h1><p>{html.escape(c.get('title',''))}</p><p><b>{status}</b> · 固定 seed {manifest.get('seed')} · PyBullet train · 512×896 · 49f · context 8f · 40 steps</p><p><a href="../">返回项目 Hub</a> · <a href="../project-info/">项目说明</a></p></div>{''.join(rows)}</main></body></html>'''
    (site/"index.html").write_text(page,encoding="utf-8")
    print(site)

if __name__=="__main__": main()
