#!/usr/bin/env python3
"""Build an auto-refreshing static page for the 30-case six-checkpoint matrix."""
from __future__ import annotations
import argparse, html, json
from pathlib import Path

def read(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else fallback

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,required=True); args=p.parse_args()
    c=read(args.config,{}) ; root=Path(c["output_root"]); manifest=read(Path(c["cases_manifest"]),{"cases":[]}); video=read(root/"video_status.json",{"state":"queued","entries":{}}); loss=read(root/"loss_status.json",{"state":"queued","entries":{}}); overlay_status=read(root/"trajectory_overlay_status.json",{"state":"queued","entries":{}})
    cases=manifest.get("cases",[]); entries=c.get("entries",[])
    method_columns=[]; method_by_key={}
    for entry in entries:
        method_key=str(entry.get("method_key", entry["entry_id"]))
        column=method_by_key.get(method_key)
        if column is None:
            column={
                "method_key": method_key,
                "method_label": entry["method_label"],
                "color": entry.get("color", "#8ca0af"),
                "entries": [],
            }
            method_by_key[method_key]=column
            method_columns.append(column)
        column["entries"].append(entry)
    weight_steps=sorted({int(entry["step"]) for entry in entries})
    site=root/"hub"; media=site/"media"; media.mkdir(parents=True,exist_ok=True)
    entry_losses={e["entry_id"]:read(root/"losses"/f"{e['entry_id']}.json",{}) for e in entries}
    loss_status_entries=loss.get("entries",{})
    rows=[]; ready_v=ready_l=ready_o=0
    for case in cases:
        cells={}
        for e in entries:
            method_key=str(e.get("method_key", e["entry_id"]))
            is_trajectory=method_key=="cotracker_trajectory"
            vroot=root/"videos"/e["entry_id"]
            candidates=list(vroot.glob(f"{case['case_id']}*.mp4"))+list(vroot.glob(f"*/{case['case_id']}*.mp4"))
            v=next(iter(candidates),None)
            l=entry_losses[e["entry_id"]]
            rec=next((x for x in l.get("cases",[]) if x.get("case_id")==case["case_id"]),None)
            if v and v.is_file():
                target=media/f"{case['case_id']}__{e['entry_id']}.mp4"
                if target.is_symlink(): target.unlink()
                elif target.exists(): target.unlink()
                target.symlink_to(v.resolve()); vhtml=(f'<span class="media-label">生成视频</span>' if is_trajectory else '')+f'<video controls preload="none" data-sync-video src="media/{html.escape(target.name)}"></video>'; ready_v+=1
            else: vhtml=(f'<span class="media-label">生成视频</span>' if is_trajectory else '')+'<div class="pending">视频待完成</div>'
            overlay_metrics=None
            if is_trajectory:
                overlay_dir=root/"trajectory_overlays"/e["entry_id"]/case["case_id"]
                overlay=overlay_dir/"trajectory_overlay.mp4"
                overlay_metrics=read(overlay_dir/"metrics.json",None)
                if overlay.is_file():
                    target=media/f"{case['case_id']}__{e['entry_id']}__trajectory_overlay.mp4"
                    if target.is_symlink(): target.unlink()
                    elif target.exists(): target.unlink()
                    target.symlink_to(overlay.resolve()); vhtml+=f'<span class="media-label">轨迹 Overlay</span><video class="trajectory-overlay" controls preload="none" data-sync-video src="media/{html.escape(target.name)}"></video>'; ready_o+=1
                else: vhtml+='<span class="media-label">轨迹 Overlay</span><div class="pending trajectory-pending">轨迹 Overlay 待完成</div>'
            if rec:
                m=rec.get("metrics",{}); loss_html=f"main {float(rec.get('loss_main',0)):.6f}"; 
                if 'train/loss_xssc' in m: loss_html+=f" · xSSC {float(m['train/loss_xssc']):.6f}"
                if 'train/loss_vjepa' in m: loss_html+=f" · V-JEPA {float(m['train/loss_vjepa']):.6f}"
                if 'train/loss_trajectory' in m: loss_html+=f" · trajectory {float(m['train/loss_trajectory']):.6f}"
                if 'train/loss_total' in m: loss_html+=f" · total {float(m['train/loss_total']):.6f}"
                ready_l+=1
            else:
                entry_status=loss_status_entries.get(e["entry_id"],{})
                state=str(entry_status.get("state","pending"))
                reason=str(entry_status.get("error","")).strip()
                if state in {"blocked","failed"}:
                    detail=f" · {reason[:240]}" if reason else ""
                    loss_html=f"loss {state}{detail}"
                else:
                    loss_html='loss 待完成'
            if overlay_metrics:
                loss_html+=f" · inference trajectory {float(overlay_metrics['trajectory_loss']):.6f} · ADE {float(overlay_metrics['trajectory_normalized_ade']):.5f}"
            cells[(int(e["step"]), method_key)]=(
                vhtml,
                loss_html,
                e["entry_id"],
            )
        method_headers=[]
        for column in method_columns:
            versions=" / ".join(dict.fromkeys(str(e["version"]) for e in column["entries"]))
            method_headers.append(
                f'<th scope="col"><div class="method-heading">'
                f'<span class="method-swatch" style="background:{html.escape(str(column["color"]),quote=True)}"></span>'
                f'<span class="method-heading-name">{html.escape(column["method_label"])}</span>'
                f'<small>{html.escape(versions)}</small></div></th>'
            )
        step_rows=[]
        for step in weight_steps:
            row_cells=[]
            present=0
            for column in method_columns:
                cell=cells.get((step, column["method_key"]))
                if cell is None:
                    row_cells.append('<td class="matrix-empty"><span>—</span><small>该 step 未提供</small></td>')
                    continue
                present+=1
                vhtml, loss_html, entry_id=cell
                row_cells.append(
                    f'<td class="matrix-cell" data-entry-id="{html.escape(entry_id,quote=True)}">'
                    f'<div class="cell-video">{vhtml}</div>'
                    f'<p class="cell-metric">{html.escape(loss_html)}</p></td>'
                )
            step_rows.append(
                f'<tr><th scope="row" class="step-cell"><strong>step-{step:04d}</strong>'
                f'<small>{present}/{len(method_columns)} 方法</small></th>{"".join(row_cells)}</tr>'
            )
        matrix=(
            '<div class="matrix-note"><span>行 = 权重 step</span><span>列 = 方法</span>'
            f'<span>{len(weight_steps)} steps · {len(method_columns)} methods</span></div>'
            '<div class="matrix-wrap"><table class="matrix"><thead><tr>'
            '<th scope="col" class="step-heading">权重 step</th>'
            f'{"".join(method_headers)}</tr></thead><tbody>{"".join(step_rows)}</tbody></table></div>'
        )
        hidden = "" if not rows else " hidden"
        rows.append(f'<section class="case" data-case-id="{html.escape(case["case_id"], quote=True)}"{hidden}><h2>{html.escape(case["case_id"])}</h2><p>{html.escape(case["prompt"])}</p>{matrix}</section>')
    expected=len(cases)*len(entries)
    expected_o=len(cases)*sum(e.get("method_key")=="cotracker_trajectory" for e in entries)
    refresh='<meta http-equiv="refresh" content="60">' if ready_v<expected or ready_l<expected or ready_o<expected_o else ''
    loss_state=str(loss.get("state"))
    if ready_l<expected and loss_state=="complete":
        loss_state="partial"
    status=f"视频 {ready_v}/{expected} · trajectory overlay {ready_o}/{expected_o} · loss {ready_l}/{expected} · video={html.escape(str(video.get('state')))} · overlay={html.escape(str(overlay_status.get('state')))} · loss={html.escape(loss_state)}"
    case_options="".join(
        f'<option value="{html.escape(case["case_id"], quote=True)}">'
        f'{index:02d} · {html.escape(str(case["prompt"]))}</option>'
        for index, case in enumerate(cases, start=1)
    )
    case_ids=[case["case_id"] for case in cases]
    ranking=[]
    for e in entries:
        records={rec.get("case_id"):rec for rec in entry_losses[e["entry_id"]].get("cases",[])}
        values=[]
        for case_id in case_ids:
            value=records.get(case_id,{}).get("loss_main")
            if isinstance(value,(int,float)):
                values.append(float(value))
        mean_loss=sum(values)/len(values) if values else None
        ranking.append({"entry":e,"count":len(values),"mean":mean_loss,"complete":bool(case_ids) and len(values)==len(case_ids)})
    completed=sorted((item for item in ranking if item["complete"]),key=lambda item:(item["mean"],item["entry"]["entry_id"]))
    incomplete=sorted((item for item in ranking if not item["complete"]),key=lambda item:(-item["count"],item["mean"] if item["mean"] is not None else float("inf"),item["entry"]["entry_id"]))
    ordered_ranking=completed+incomplete
    ranks={item["entry"]["entry_id"]:index for index,item in enumerate(completed,start=1)}
    complete_means=[item["mean"] for item in completed]
    best_mean=min(complete_means) if complete_means else None
    worst_mean=max(complete_means) if complete_means else None
    ranking_rows=[]
    for item in ordered_ranking:
        e=item["entry"]; rank=ranks.get(e["entry_id"]); mean_loss=item["mean"]
        rank_html=f'<span class="rank-number rank-{rank if rank and rank<=3 else "other"}">{rank}</span>' if rank else '<span class="rank-pending">—</span>'
        mean_html=f'{mean_loss:.8f}' if mean_loss is not None else '—'
        delta_html=f'+{mean_loss-best_mean:.8f}' if rank and best_mean is not None else '—'
        coverage_class='coverage-complete' if item["complete"] else 'coverage-pending'
        coverage=f'<span class="coverage {coverage_class}">{item["count"]}/{len(case_ids)}</span>'
        if rank and worst_mean is not None and best_mean is not None:
            span=worst_mean-best_mean
            bar_width=18.0 if span==0 else 18.0+82.0*(mean_loss-best_mean)/span
            bar=f'<span class="loss-track" aria-label="mean loss_main {mean_loss:.8f}"><span class="loss-bar" style="width:{bar_width:.2f}%"></span></span>'
        else:
            bar='<span class="loss-track loss-track-pending" aria-hidden="true"></span>'
        method=(f'<span class="method-name"><span class="method-swatch" style="background:{html.escape(str(e.get("color","#8ca0af")),quote=True)}"></span>'
                f'{html.escape(e["method_label"])}</span><span class="method-meta">{html.escape(e["version"])} · step-{int(e["step"]):04d}</span>')
        ranking_rows.append(f'<tr><td>{rank_html}</td><td>{method}</td><td>{coverage}</td><td class="metric">{mean_html}</td><td class="metric delta">{delta_html}</td><td>{bar}</td></tr>')
    ranking_html=(f'<section class="ranking" aria-labelledby="ranking-title"><div class="ranking-heading"><div><h2 id="ranking-title">Val loss 排名</h2>'
                  f'<p>30 个固定 case 的平均 <code>loss_main</code>，越低越好；仅完成 30/30 的 checkpoint 参与正式排名。</p></div>'
                  f'<span class="ranking-count">已排名 {len(completed)}/{len(entries)}</span></div><div class="ranking-table-wrap"><table class="ranking-table"><thead><tr>'
                  f'<th scope="col">排名</th><th scope="col">方法 / checkpoint</th><th scope="col">case</th><th scope="col">mean loss_main</th><th scope="col">相对最佳</th><th scope="col">loss 刻度（短 = 好）</th>'
                  f'</tr></thead><tbody>{"".join(ranking_rows)}</tbody></table></div></section>')
    replay_script='''<script>
(() => {
  const button = document.getElementById('sync-replay');
  const allVideos = [...document.querySelectorAll('[data-sync-video]')];
  allVideos.forEach((video) => {
    video.loop = false;
    video.removeAttribute('loop');
  });

  function currentCase() {
    const cases = [...document.querySelectorAll('.case:not([hidden])')];
    const viewportCenter = window.innerHeight / 2;
    return cases.find((item) => {
      const rect = item.getBoundingClientRect();
      return rect.top <= viewportCenter && rect.bottom >= viewportCenter;
    }) || cases.reduce((nearest, item) => {
      if (!nearest) return item;
      const itemRect = item.getBoundingClientRect();
      const nearestRect = nearest.getBoundingClientRect();
      const itemDistance = Math.abs((itemRect.top + itemRect.bottom) / 2 - viewportCenter);
      const nearestDistance = Math.abs((nearestRect.top + nearestRect.bottom) / 2 - viewportCenter);
      return itemDistance < nearestDistance ? item : nearest;
    }, null);
  }

  function showCase(caseId, updateHash = true) {
    const cases = [...document.querySelectorAll('.case')];
    const target = cases.find((item) => item.dataset.caseId === caseId) || cases[0];
    if (!target) return;
    cases.forEach((item) => {
      const visible = item === target;
      item.hidden = !visible;
      if (!visible) item.querySelectorAll('[data-sync-video]').forEach((video) => video.pause());
    });
    const select = document.getElementById('case-select');
    if (select) select.value = target.dataset.caseId;
    if (updateHash) history.replaceState(null, '', `#${encodeURIComponent(target.dataset.caseId)}`);
    window.scrollTo({top: 0, behavior: 'smooth'});
  }

  function waitUntilReady(video) {
    video.loop = false;
    video.removeAttribute('loop');
    if (video.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA) {
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const finish = () => resolve();
      video.addEventListener('canplay', finish, {once: true});
      video.addEventListener('error', finish, {once: true});
      video.preload = 'auto';
      video.load();
    });
  }

  button.addEventListener('click', async () => {
    if (button.disabled) return;
    const group = currentCase();
    const videos = group ? [...group.querySelectorAll('[data-sync-video]')] : [];
    if (!videos.length) return;

    button.disabled = true;
    button.textContent = '准备中…';
    videos.forEach((video) => video.pause());
    await Promise.all(videos.map(waitUntilReady));
    videos.forEach((video) => {
      video.loop = false;
      video.removeAttribute('loop');
      video.currentTime = 0;
    });
    await new Promise((resolve) => requestAnimationFrame(resolve));
    await Promise.allSettled(videos.map((video) => video.play()));
    button.textContent = '同步重播';
    button.disabled = false;
  });

  const select = document.getElementById('case-select');
  if (select) {
    select.addEventListener('change', () => showCase(select.value));
    window.addEventListener('hashchange', () => {
      const requested = decodeURIComponent(window.location.hash.slice(1));
      showCase(requested, false);
    });
    const requested = decodeURIComponent(window.location.hash.slice(1));
    showCase(requested || select.value, false);
  }
})();
</script>'''
    style = """<style>
body{margin:0;background:#10161d;color:#edf4fa;font:14px system-ui,sans-serif}
main{max-width:1900px;margin:auto;padding:24px}
h1{font-size:32px}
.summary,.case{background:#18232d;border:1px solid #304454;border-radius:12px;padding:14px}
.summary{position:sticky;top:0;z-index:2}
.case{margin:18px 0}
.ranking{margin:18px 0 24px;padding:18px 0;border-top:1px solid #304454;border-bottom:1px solid #304454}
.ranking-heading{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin:0 2px 12px}
.ranking-heading h2{margin:0 0 4px;font-size:20px}
.ranking-heading p{margin:0}
.ranking-count{flex:none;padding:5px 8px;border:1px solid #4e6877;border-radius:4px;color:#c8d8e3;font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace}
.ranking-table-wrap{overflow-x:auto;border:1px solid #304454;border-radius:6px;background:#131d25}
.ranking-table{width:100%;min-width:940px;border-collapse:collapse;font-size:12px}
.ranking-table th,.ranking-table td{padding:9px 10px;border-bottom:1px solid #263846;text-align:left;vertical-align:middle}
.ranking-table th{background:#1d2a34;color:#9fb3c2;font-size:11px;text-transform:uppercase}
.ranking-table tbody tr:last-child td{border-bottom:0}
.ranking-table tbody tr:hover{background:#1a2731}
.rank-number{display:inline-grid;width:25px;height:25px;place-items:center;border:1px solid #496171;border-radius:4px;color:#d9e7ef;font:800 12px ui-monospace,SFMono-Regular,Consolas,monospace}
.rank-1{border-color:#f2c14e;background:#66501b;color:#fff8df}
.rank-2{border-color:#b9c6cf;background:#46535d;color:#f6fbff}
.rank-3{border-color:#c58a5a;background:#5e3f2b;color:#fff3e8}
.rank-pending{display:inline-block;width:25px;text-align:center;color:#637988}
.method-name{display:flex;align-items:center;gap:7px;color:#edf4fa;font-weight:700}
.method-swatch{display:inline-block;width:9px;height:9px;flex:none;border-radius:2px}
.method-meta{display:block;margin:3px 0 0 16px;color:#7f96a5;font:11px ui-monospace,SFMono-Regular,Consolas,monospace}
.coverage{display:inline-block;min-width:44px;padding:3px 5px;border-radius:3px;text-align:center;font:700 11px ui-monospace,SFMono-Regular,Consolas,monospace}
.coverage-complete{background:#173c37;color:#74d4c4}
.coverage-pending{background:#3a3020;color:#e8bd6d}
.metric{color:#e5eef4;font:12px ui-monospace,SFMono-Regular,Consolas,monospace;font-variant-numeric:tabular-nums}
.delta{color:#91a6b5}
.loss-track{display:block;width:100%;min-width:120px;height:8px;overflow:hidden;border-radius:2px;background:#263846}
.loss-bar{display:block;height:100%;background:#58b6a9}
.loss-track-pending{background:repeating-linear-gradient(90deg,#263846 0,#263846 6px,#1a2731 6px,#1a2731 10px)}
code{color:#d4e3ec;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}
.matrix-note{display:flex;align-items:center;gap:18px;margin:18px 0 8px;color:#91a6b5;font:11px ui-monospace,SFMono-Regular,Consolas,monospace;text-transform:uppercase}
.matrix-note span:first-child{color:#f2c14e}
.matrix-wrap{overflow:auto;border:1px solid #304454;border-radius:7px;background:#131d25;isolation:isolate}
.matrix{width:max-content;min-width:100%;border-collapse:separate;border-spacing:0;table-layout:fixed;font-size:12px}
.matrix th,.matrix td{width:224px;min-width:224px;padding:8px;border-right:1px solid #263846;border-bottom:1px solid #263846;text-align:left;vertical-align:top}
.matrix thead th{position:sticky;top:0;z-index:4;background:#1d2a34;color:#edf4fa;text-transform:none}
.matrix thead th:last-child,.matrix tbody td:last-child{border-right:0}
.matrix .step-heading,.matrix .step-cell{width:112px;min-width:112px}
.matrix .step-heading{left:0;z-index:6;color:#f2c14e;font-size:11px;letter-spacing:.05em}
.matrix .step-cell{position:sticky;left:0;z-index:3;background:#17242d;color:#f2c14e}
.matrix .step-cell strong{display:block;font:700 12px ui-monospace,SFMono-Regular,Consolas,monospace}
.matrix .step-cell small{display:block;margin-top:5px;color:#8095a4;font-size:10px;font-weight:400}
.matrix tbody tr:hover td{background:#1a2731}
.matrix tbody tr:hover .step-cell{background:#20313c}
.method-heading{display:grid;grid-template-columns:10px minmax(0,1fr);column-gap:7px;align-items:start;line-height:1.3}
.method-heading .method-swatch{grid-row:1 / span 2;margin-top:4px}
.method-heading-name{font-weight:700;overflow-wrap:anywhere}
.method-heading small{grid-column:2;margin-top:4px;color:#7f96a5;font:10px ui-monospace,SFMono-Regular,Consolas,monospace;text-transform:none}
.matrix-cell{background:#18232d}
.cell-video video,.cell-video .pending{display:block;width:100%;aspect-ratio:896/512;background:#05090d;object-fit:cover}
.cell-video .trajectory-overlay,.cell-video .trajectory-pending{aspect-ratio:2688/512;object-fit:contain}
.media-label{display:block;margin:6px 0 4px;color:#8ea6b5;font:700 10px ui-monospace,SFMono-Regular,Consolas,monospace;text-transform:uppercase}
.cell-metric{min-height:30px;margin:7px 0 0;color:#a9bac8;font:10px/1.45 ui-monospace,SFMono-Regular,Consolas,monospace;overflow-wrap:anywhere}
.matrix-empty{vertical-align:middle!important;background:#151f27;color:#607582;text-align:center!important}
.matrix-empty span{display:block;margin-top:54px;color:#748b9a;font:700 18px ui-monospace,SFMono-Regular,Consolas,monospace}
.matrix-empty small{display:block;margin-top:7px;font-size:10px}
p{color:#a9bac8;font-size:12px}
.pending{display:grid;place-items:center;background:#263542;color:#8ca0af;font-size:12px}
.case-picker{display:flex;align-items:center;gap:10px;margin:16px 0 2px}
.case-picker label{color:#a9bac8;font-size:12px;font-weight:700;white-space:nowrap}
.case-picker select{min-width:min(680px,100%);height:38px;padding:0 34px 0 10px;border:1px solid #4e6877;border-radius:6px;background:#101a22;color:#edf4fa;font:600 13px system-ui,sans-serif}
.case-picker select:focus-visible{outline:3px solid #f2c14e;outline-offset:2px}
.sync-replay{position:fixed;right:18px;bottom:max(18px,env(safe-area-inset-bottom));z-index:10;min-height:42px;padding:0 16px;border:1px solid #58b6a9;border-radius:6px;background:#207b72;color:#f5fffd;font:600 14px system-ui,sans-serif;letter-spacing:0;box-shadow:0 6px 18px rgba(0,0,0,.28);cursor:pointer}
.sync-replay:hover{background:#278b81}
.sync-replay:focus-visible{outline:3px solid #f2c14e;outline-offset:3px}
.sync-replay:disabled{cursor:wait;opacity:.72}
@media(max-width:700px){main{padding:12px}.matrix-note{align-items:flex-start;flex-direction:column;gap:4px}.ranking-heading{align-items:stretch;flex-direction:column}.ranking-count{align-self:flex-start}.case-picker{align-items:stretch;flex-direction:column;gap:6px}.case-picker select{width:100%;min-width:0}.sync-replay{right:12px;bottom:max(12px,env(safe-area-inset-bottom))}}
</style>"""
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>30-case validation · method comparison</title>{style}</head><body><button id="sync-replay" class="sync-replay" type="button" title="同步重播当前 case 的所有对比视频" aria-live="polite">同步重播</button><main><div class="summary"><h1>30-case train validation · {len(entries)} checkpoints</h1><p>{html.escape(c.get('title',''))}</p><p><b>{status}</b> · 固定 seed {manifest.get('seed')} · PyBullet train · 49f · context 8f · 40 steps</p><div class="case-picker"><label for="case-select">选择 case</label><select id="case-select" aria-label="选择要查看的 case">{case_options}</select></div><p><a href="../">返回项目 Hub</a> · <a href="../project-info/">项目说明</a></p></div>{ranking_html}{''.join(rows)}</main>{replay_script}</body></html>'''
    (site/"index.html").write_text(page,encoding="utf-8")
    print(site)

if __name__=="__main__": main()
