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
                target.symlink_to(v.resolve()); vhtml=f'<video controls preload="none" data-sync-video src="media/{html.escape(target.name)}"></video>'; ready_v+=1
            else: vhtml='<div class="pending">视频待完成</div>'
            if rec:
                m=rec.get("metrics",{}); loss_html=f"main {float(rec.get('loss_main',0)):.6f}"; 
                if 'train/loss_xssc' in m: loss_html+=f" · xSSC {float(m['train/loss_xssc']):.6f}"
                if 'train/loss_vjepa' in m: loss_html+=f" · V-JEPA {float(m['train/loss_vjepa']):.6f}"
                if 'train/loss_total' in m: loss_html+=f" · total {float(m['train/loss_total']):.6f}"
                ready_l+=1
            else: loss_html='loss 待完成'
            panels.append(f'<article class="panel"><h3>{html.escape(e["method_label"])} · {html.escape(e["version"])} · step-{int(e["step"]):04d}</h3>{vhtml}<p>{html.escape(loss_html)}</p></article>')
        hidden = "" if not rows else " hidden"
        rows.append(f'<section class="case" data-case-id="{html.escape(case["case_id"], quote=True)}"{hidden}><h2>{html.escape(case["case_id"])}</h2><p>{html.escape(case["prompt"])}</p><div class="grid">{"".join(panels)}</div></section>')
    expected=len(cases)*len(entries)
    refresh='<meta http-equiv="refresh" content="60">' if ready_v<expected or ready_l<expected else ''
    status=f"视频 {ready_v}/{expected} · loss {ready_l}/{expected} · video={html.escape(str(video.get('state')))} · loss={html.escape(str(loss.get('state')))}"
    case_options="".join(
        f'<option value="{html.escape(case["case_id"], quote=True)}">'
        f'{index:02d} · {html.escape(str(case["prompt"]))}</option>'
        for index, case in enumerate(cases, start=1)
    )
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
.summary,.case,.panel{background:#18232d;border:1px solid #304454;border-radius:12px;padding:14px}
.summary{position:sticky;top:0;z-index:2}
.case{margin:18px 0}
.grid{display:grid;grid-template-columns:repeat(6,minmax(200px,1fr));gap:8px}
.panel{padding:8px}
h3{font-size:12px;min-height:34px;margin:0 0 6px}
video{width:100%;aspect-ratio:896/512;background:#000}
p{color:#a9bac8;font-size:12px}
.pending{aspect-ratio:896/512;display:grid;place-items:center;background:#263542;color:#8ca0af;font-size:12px}
.case-picker{display:flex;align-items:center;gap:10px;margin:16px 0 2px}
.case-picker label{color:#a9bac8;font-size:12px;font-weight:700;white-space:nowrap}
.case-picker select{min-width:min(680px,100%);height:38px;padding:0 34px 0 10px;border:1px solid #4e6877;border-radius:6px;background:#101a22;color:#edf4fa;font:600 13px system-ui,sans-serif}
.case-picker select:focus-visible{outline:3px solid #f2c14e;outline-offset:2px}
.sync-replay{position:fixed;right:18px;bottom:max(18px,env(safe-area-inset-bottom));z-index:10;min-height:42px;padding:0 16px;border:1px solid #58b6a9;border-radius:6px;background:#207b72;color:#f5fffd;font:600 14px system-ui,sans-serif;letter-spacing:0;box-shadow:0 6px 18px rgba(0,0,0,.28);cursor:pointer}
.sync-replay:hover{background:#278b81}
.sync-replay:focus-visible{outline:3px solid #f2c14e;outline-offset:3px}
.sync-replay:disabled{cursor:wait;opacity:.72}
@media(max-width:1200px){.grid{grid-template-columns:repeat(3,minmax(200px,1fr))}}
@media(max-width:700px){main{padding:12px}.grid{grid-template-columns:1fr}.case-picker{align-items:stretch;flex-direction:column;gap:6px}.case-picker select{width:100%;min-width:0}.sync-replay{right:12px;bottom:max(12px,env(safe-area-inset-bottom))}}
</style>"""
    page=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">{refresh}<title>30-case validation · method comparison</title>{style}</head><body><button id="sync-replay" class="sync-replay" type="button" title="同步重播当前 case 的所有对比视频" aria-live="polite">同步重播</button><main><div class="summary"><h1>30-case train validation · {len(entries)} checkpoints</h1><p>{html.escape(c.get('title',''))}</p><p><b>{status}</b> · 固定 seed {manifest.get('seed')} · PyBullet train · 49f · context 8f · 40 steps</p><div class="case-picker"><label for="case-select">选择 case</label><select id="case-select" aria-label="选择要查看的 case">{case_options}</select></div><p><a href="../">返回项目 Hub</a> · <a href="../project-info/">项目说明</a></p></div>{''.join(rows)}</main>{replay_script}</body></html>'''
    (site/"index.html").write_text(page,encoding="utf-8")
    print(site)

if __name__=="__main__": main()
