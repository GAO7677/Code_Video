#!/usr/bin/env python3
"""Render fixed PCK@32 Top30 object-query attention on GT training frames."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


HERE = Path(__file__).resolve().parent
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
DIFFTRACK = Path("/home/gaoya/Code_Video/DiffTrack-main")
for path in (HERE, CODE_ROOT, DIFFTRACK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import launch_from_config as launcher
import train_xssc_object_self_attn_lora as train
from AAA_my_test import analyze_wan_gt_toy_worker as gt


DEFAULT_MANIFEST = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/"
    "0717pybullet_5000_vbenchtop5/manifest.json"
)
DEFAULT_SELECTION = HERE / "configs/lora_pck32_top100_heads.json"
DEFAULT_RANKING = Path(
    "/data/gaoya/agent-data/outputs/"
    "attention_lora_neighbor_ranking_seed090094_case001460/seeds/"
    "seed_090094/pck32/all_steps/alpha090/videos/selection.json"
)
STEPS = (0, 9, 19, 29, 39)
HEIGHT = 512
WIDTH = 896
LATENT_FRAMES = 13
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
TILE_W = 160
TILE_H = 92
HEADER_H = 38


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_status(root: Path, state: str, message: str, **extra) -> None:
    atomic_json(root / "status.json", {"state": state, "message": message, **extra})


def write_page(root: Path) -> None:
    page = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Step-500 Top30 Object Query Attention</title><style>
:root{--paper:#eee7d8;--ink:#17251f;--card:#fffdf8;--line:#bdb19c;--green:#176b5c;--rust:#ad452f;--gold:#bc812c}*{box-sizing:border-box}body{margin:0;color:var(--ink);background:radial-gradient(circle at 4% 0,#e99c5550,transparent 34rem),radial-gradient(circle at 98% 2%,#4c947653,transparent 38rem),var(--paper);font-family:"Noto Serif SC","Source Han Serif SC",serif}header{position:sticky;top:0;z-index:10;padding:16px 24px;background:#eee7d8ef;border-bottom:1px solid var(--line);backdrop-filter:blur(11px)}h1{margin:3px 0;font-size:clamp(27px,4vw,48px)}header p{margin:5px 0}.tools{display:flex;gap:9px;align-items:center;flex-wrap:wrap}select,button{padding:8px 11px;border:1px solid var(--line);background:#fff;font-weight:900}.status{font:12px ui-monospace,monospace;color:#58665f}main{width:min(2350px,calc(100% - 18px));margin:auto;padding:18px 0 70px}.summary{padding:12px;border:1px solid var(--line);border-radius:13px;background:var(--card);margin-bottom:13px}.video{max-width:720px}.video video{display:block;width:100%;background:#111}.record{margin:10px 0;padding:10px;border:1px solid var(--line);border-radius:12px;background:var(--card)}.record.aggregate{border-left:7px solid var(--gold)}.record.head{border-left:7px solid var(--green)}.record h2{margin:0 0 7px;font-size:18px}.meta{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px}.pill{padding:4px 7px;border-radius:99px;background:#e9e1d2;font:10px ui-monospace,monospace}.scroll{overflow:auto}.record img{display:block;min-width:2240px;width:2240px;border:1px solid var(--line);background:#111}.pending{padding:60px;border:1px dashed var(--line);border-radius:12px;background:var(--card)}@media(max-width:800px){header{position:static}}
</style></head><body><header><a href="http://localhost:8855/">返回总入口</a><h1>Training Baseline · Fixed Top30 Object Query Attention</h1><p>Wan2.2 + OpenVid LoRA · 不加载Top100训练模块 · GT teacher-forced · Q=F04/latent 1 · S00/S09/S19/S29/S39</p><div class="tools"><label>Sample <select id="case"></select></label><label>Object <select id="region"></select></label><label>Noise <select id="step"></select></label><button id="refresh">手动刷新</button><span id="status" class="status">读取中</span></div></header><main><section id="summary" class="summary"></section><section id="video" class="video"></section><section id="records"></section></main><script>
const e=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));let data=null;const ids=['case','region','step'];function sync(id,values){const el=document.getElementById(id),old=el.value;el.innerHTML=values.map(x=>`<option value="${e(x.value)}">${e(x.label)}</option>`).join('');if(values.some(x=>x.value===old))el.value=old}function render(){if(!data)return;sync('case',data.cases.map(x=>({value:x.case_key,label:`${x.family} · ${x.case_key}`})));const key=document.getElementById('case').value||data.cases[0]?.case_key,c=data.cases.find(x=>x.case_key===key);if(!c)return;sync('region',c.regions.map(x=>({value:x.name,label:x.phrase})));sync('step',data.steps.map(x=>({value:String(x.step),label:`S${String(x.step).padStart(2,'0')} · σ=${Number(x.sigma).toFixed(4)}`})));const region=document.getElementById('region').value||c.regions[0]?.name,step=Number(document.getElementById('step').value||data.steps[0].step),rows=data.records.filter(x=>x.case_key===key&&x.region===region&&x.step===step).sort((a,b)=>a.rank-b.rank);document.getElementById('summary').innerHTML=`<b>${e(c.caption)}</b><div class="meta"><span class="pill">${e(c.family)}</span><span class="pill">Wan2.2 + OpenVid LoRA baseline</span><span class="pill">无Top100训练模块</span><span class="pill">${rows.length-1}/30 heads</span><span class="pill">GT latent加噪后单次前向</span></div>`;document.getElementById('video').innerHTML=`<h2>GT 49 frames</h2><video controls preload="metadata" src="${e(c.gt_video)}"></video>`;document.getElementById('records').innerHTML=rows.length?rows.map(r=>`<article class="record ${r.aggregate?'aggregate':'head'}"><h2>${r.aggregate?'Top30 Head Mean':`#${r.rank} · B${String(r.block).padStart(2,'0')} / H${String(r.head).padStart(2,'0')}`}</h2><div class="meta"><span class="pill">S${String(r.step).padStart(2,'0')}</span><span class="pill">t=${Number(r.timestep).toFixed(3)}</span><span class="pill">σ=${Number(r.sigma).toFixed(5)}</span>${r.pck32==null?'':`<span class="pill">LoRA PCK@32 ${Number(r.pck32).toFixed(3)}</span>`}<span class="pill">Q tokens ${r.query_count}</span></div><div class="scroll"><img loading="lazy" src="${e(r.image)}"></div></article>`).join(''):'<div class="pending">该组合尚未生成</div>'}
async function load(){try{const s=await fetch('status.json?'+Date.now()).then(r=>r.json());document.getElementById('status').textContent=`${s.state} · ${s.message}`;const r=await fetch('catalog.json?'+Date.now());if(r.ok){data=await r.json();render()}else document.getElementById('records').innerHTML='<div class="pending">等待 step-500 与捕获结果</div>'}catch(err){document.getElementById('status').textContent=err}}ids.forEach(id=>document.getElementById(id).addEventListener('change',render));document.getElementById('refresh').addEventListener('click',load);load();
</script></body></html>'''
    (root / "index.html").write_text(page, encoding="utf-8")


def prepare_dataset(args) -> None:
    rows = json.loads(args.training_manifest.read_text(encoding="utf-8"))
    selected = []
    for family in ("F1", "F2", "F3", "F4", "F5"):
        row = next(
            (
                item for item in rows
                if item.get("family_key") == family
                and Path(item["video"]).is_file()
                and item.get("caption")
                and item.get("object_phrases")
            ),
            None,
        )
        if row is not None:
            selected.append(row)
    if len(selected) != 5:
        raise RuntimeError(f"Expected one valid sample from F1-F5, found {len(selected)}")
    dataset_root = args.output_root / "dataset"
    for index, row in enumerate(selected):
        case_key = f"train_top30_{index:02d}_{row['case_id']}"
        case_dir = dataset_root / "cases" / f"case_{case_key}"
        case_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_key": case_key,
            "object_count": len(row["object_phrases"]),
            "family": row["family_key"],
            "base": {
                "video": row["video"],
                "source_video": row["video"],
                "caption": row["caption"],
                "object_phrases": row["object_phrases"],
            },
        }
        atomic_json(case_dir / "case_manifest.json", payload)
    atomic_json(
        dataset_root / "manifest.json",
        {"source": str(args.training_manifest), "samples": selected},
    )
    write_page(args.output_root)
    write_status(args.output_root, "prepared", "5 training samples selected; preparing object caches")


def top30_rows(selection_path: Path, ranking_path: Path) -> list[dict]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    targets = selection["targets"][:30]
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))["top100_step_00"]
    scores = {(int(row["block"]), int(row["head"])): float(row["ranking_score"]) for row in ranking}
    return [
        {
            "rank": index + 1,
            "block": int(row["block"]),
            "head": int(row["head"]),
            "pck32": scores[(int(row["block"]), int(row["head"]))],
        }
        for index, row in enumerate(targets)
    ]


def build_sparse_model(checkpoint: Path, device: str):
    manifest_path = next(
        (parent / "resolved_experiment_config.json" for parent in checkpoint.parents if (parent / "resolved_experiment_config.json").is_file()),
        None,
    )
    if manifest_path is None:
        raise FileNotFoundError("resolved_experiment_config.json not found above checkpoint")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = manifest["resolved_config"]
    command = launcher.build_command(config, Path("/tmp/top30_object_query_unused"))
    train_index = command.index(str(launcher.TRAIN_SCRIPT))
    model_args = train.build_parser().parse_args(command[train_index + 1 :])
    model = train.build_model(model_args, SimpleNamespace(device=torch.device(device)))
    checkpoint_file = train.tvn._resolve_checkpoint_file(checkpoint)
    train.validate_head_selection_resume_checkpoint(model, checkpoint_file)
    load_info = train.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint_file,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(".object_cross_attn.", ".object_gate", ".self_attn."),
    )
    expected = sum(1 for _, parameter in model.named_parameters() if parameter.requires_grad)
    if load_info["loaded_count"] != expected or load_info["skipped_shape_mismatch"]:
        raise RuntimeError(f"Incomplete sparse checkpoint load: {load_info}")
    model.to(torch.device(device))
    model.pipe.to(device=torch.device(device), dtype=model.pipe.torch_dtype)
    model.eval()
    return model, model.pipe, manifest_path


def build_training_baseline(config_path: Path, device: str):
    manifest = json.loads(config_path.read_text(encoding="utf-8"))
    config = manifest["resolved_config"]
    wan_root = Path(config["paths"]["wan_root"])
    openvid_lora = Path(config["paths"]["pretrained_lora_checkpoint"])
    pipe = gt.target.core.build_pipeline(wan_root, device, openvid_lora)
    return None, pipe, config_path


def load_regions(cache_dir: Path) -> tuple[list[dict], np.ndarray, np.ndarray]:
    metadata = json.loads((cache_dir / "regions.json").read_text(encoding="utf-8"))
    with np.load(cache_dir / "regions.npz") as arrays:
        points = arrays["query_points"].astype(np.float32)
        masks = arrays["masks_rhw"].astype(np.uint8)
        context = arrays["context_frame_rgb"].astype(np.uint8)
    regions = []
    for index, row in enumerate(metadata["regions"]):
        if row.get("region_type") != "object":
            continue
        regions.append(
            {
                "name": str(row["region_name"]),
                "phrase": str(row.get("region_phrase") or row["region_name"]),
                "start": int(row["point_start"]),
                "end": int(row["point_end"]),
                "mask": masks[index],
            }
        )
    if not regions:
        raise RuntimeError(f"No object regions in {cache_dir}")
    return regions, points, context


class TopHeadCapture:
    def __init__(self, pipe, targets: list[dict], points: np.ndarray, steps: tuple[int, ...]):
        self.pipe = pipe
        self.targets = targets
        self.by_block: dict[int, list[int]] = {}
        for row in targets:
            self.by_block.setdefault(row["block"], []).append(row["head"])
        self.points = torch.from_numpy(points).float()
        self.steps = set(steps)
        self.records: dict[tuple[int, int, int], np.ndarray] = {}
        self.meta: dict[int, dict] = {}
        self.active = False
        self.current_step = -1
        self.current_grid = None
        self.current_prefix = 0
        self.handles = []
        self.original_model_fn = None

    def install(self):
        self.original_model_fn = self.pipe.model_fn
        original = self.original_model_fn

        def wrapped(*args, **kwargs):
            timestep, latents = kwargs.get("timestep"), kwargs.get("latents")
            if timestep is None or latents is None:
                return original(*args, **kwargs)
            timesteps = self.pipe.scheduler.timesteps.detach().float().cpu()
            value = float(timestep.detach().flatten()[0].float().cpu())
            self.current_step = int(torch.argmin((timesteps - value).abs()).item())
            patch = tuple(int(value) for value in kwargs["dit"].patch_size)
            self.current_grid = (
                int(latents.shape[2] // patch[0]),
                int(latents.shape[3] // patch[1]),
                int(latents.shape[4] // patch[2]),
            )
            clean = kwargs.get("clean_prefix_latents")
            self.current_prefix = int(clean.shape[2]) if clean is not None else int(kwargs.get("num_clean_prefix_latents") or 0)
            sigma = self.pipe.scheduler.sigmas[self.current_step]
            self.meta[self.current_step] = {
                "timestep": value,
                "sigma": float(sigma.detach().float().cpu() if isinstance(sigma, torch.Tensor) else sigma),
            }
            self.active = self.current_step in self.steps
            try:
                return original(*args, **kwargs)
            finally:
                self.active = False

        self.pipe.model_fn = wrapped
        models = [self.pipe.dit]
        if getattr(self.pipe, "dit2", None) is not None and self.pipe.dit2 is not self.pipe.dit:
            models.append(self.pipe.dit2)
        for model in models:
            for block in self.by_block:
                self.handles.append(
                    model.blocks[block].self_attn.attn.register_forward_pre_hook(self._hook(block))
                )

    def _hook(self, block: int):
        def capture(module, inputs):
            if not self.active:
                return
            q, k = inputs[:2]
            if q.shape[0] != 1 or self.current_grid is None:
                raise RuntimeError(f"Unexpected Q/K shape {q.shape}")
            time, height, width = self.current_grid
            sequence, channels = q.shape[1:]
            if sequence != time * height * width:
                raise RuntimeError("Q/K token geometry mismatch")
            heads = int(module.num_heads)
            head_dim = channels // heads
            qh = q[0].reshape(sequence, heads, head_dim).permute(1, 0, 2)
            kh = k[0].reshape(sequence, heads, head_dim).permute(1, 0, 2)
            x = torch.floor(self.points[:, 0] * width / WIDTH).long().clamp(0, width - 1)
            y = torch.floor(self.points[:, 1] * height / HEIGHT).long().clamp(0, height - 1)
            query_time = self.current_prefix - 1
            query_indices = query_time * height * width + y * width + x
            for head in self.by_block[block]:
                key = (self.current_step, block, head)
                if key in self.records:
                    continue
                logits = torch.matmul(qh[head, query_indices].float(), kh[head].float().T) / math.sqrt(head_dim)
                probabilities = logits.softmax(dim=-1).reshape(len(query_indices), time, height, width)
                self.records[key] = probabilities.detach().cpu().numpy().astype(np.float32)
        return capture

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        if self.original_model_fn is not None:
            self.pipe.model_fn = self.original_model_fn


def load_case_manifests(dataset_root: Path) -> list[dict]:
    cases = []
    for path in sorted((dataset_root / "cases").glob("case_*/case_manifest.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["manifest"] = str(path)
        cases.append(row)
    return cases


def overlay(frame: np.ndarray, values: np.ndarray, vmax: float) -> np.ndarray:
    base = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), (TILE_W, TILE_H))
    heat = cv2.resize(values.astype(np.float32), (TILE_W, TILE_H))
    norm = np.clip(heat / max(vmax, 1e-12), 0.0, 1.0)
    color = cv2.applyColorMap(np.uint8(norm * 255), cv2.COLORMAP_TURBO)
    alpha = (0.12 + 0.72 * norm)[..., None]
    return np.uint8(np.clip(base * (1 - alpha) + color * alpha, 0, 255))


def render_strip(frames, values, context, mask, points, title: str, output: Path) -> None:
    canvas = np.full((HEADER_H + TILE_H, (LATENT_FRAMES + 1) * TILE_W, 3), 245, np.uint8)
    cv2.putText(canvas, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .48, (30, 42, 36), 1, cv2.LINE_AA)
    query = cv2.resize(cv2.cvtColor(context, cv2.COLOR_RGB2BGR), (TILE_W, TILE_H))
    resized_mask = cv2.resize(mask, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST) > 0
    query[resized_mask] = (0.7 * query[resized_mask] + 0.3 * np.array([45, 145, 210])).astype(np.uint8)
    for x, y in points:
        cv2.circle(query, (int(x * TILE_W / WIDTH), int(y * TILE_H / HEIGHT)), 3, (20, 20, 235), -1)
    canvas[HEADER_H:, :TILE_W] = query
    positive = values[values > 0]
    vmax = float(np.quantile(positive, .995)) if positive.size else 1.0
    for latent in range(LATENT_FRAMES):
        x0 = (latent + 1) * TILE_W
        canvas[HEADER_H:, x0:x0 + TILE_W] = overlay(frames[latent * 4], values[latent], vmax)
        cv2.putText(canvas, f"K{latent:02d}/F{latent*4:02d}", (x0 + 39, 25), cv2.FONT_HERSHEY_SIMPLEX, .38, (30, 42, 36), 1)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), canvas, [cv2.IMWRITE_JPEG_QUALITY, 91])


def capture(args) -> None:
    targets = top30_rows(args.selection, args.ranking)
    if args.baseline_config is not None:
        model, pipe, experiment_manifest = build_training_baseline(
            args.baseline_config, args.device
        )
        model_label = "Wan2.2 + OpenVid LoRA training baseline; no Top100 modules"
    else:
        model, pipe, experiment_manifest = build_sparse_model(args.checkpoint, args.device)
        model_label = "fixed Top100 step-000500"
    cases = load_case_manifests(args.output_root / "dataset")
    catalog_cases, catalog_records, step_meta = [], [], {}
    for case_index, case in enumerate(cases, start=1):
        case_key, base = case["case_key"], case["base"]
        write_status(args.output_root, "capturing", f"case {case_index}/5: {case_key}")
        cache_dir = args.cache_root / case_key
        regions, points, context_frame = load_regions(cache_dir)
        gt_frames = gt.load_video_prefix(Path(base["source_video"]), 49, HEIGHT, WIDTH, "cache")
        context_frames = gt.load_video_prefix(Path(base["video"]), 8, HEIGHT, WIDTH, "cache")
        if len(gt_frames) != 49 or len(context_frames) != 8:
            raise RuntimeError(f"Invalid frame count for {case_key}")
        gt_latents = gt.encode_gt_video(pipe, gt_frames, "whole_video")
        shared, positive = gt.prepare_conditioning(
            pipe,
            prompt=base["caption"],
            context_video=context_frames,
            height=HEIGHT,
            width=WIDTH,
            num_frames=49,
            sampling_steps=40,
            sigma_shift=5.0,
            cfg_scale=5.0,
            seed=42,
        )
        prefix = gt.validate_geometry(SimpleNamespace(height=HEIGHT, width=WIDTH), gt_latents, shared, 49)
        clean, noise = shared["clean_prefix_latents"], shared["noise"]
        capture_state = TopHeadCapture(pipe, targets, points, STEPS)
        pipe.load_models_to_device(pipe.in_iteration_models)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        capture_state.install()
        try:
            with torch.inference_mode():
                for step in STEPS:
                    timestep = pipe.scheduler.timesteps[step].unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
                    noised = gt_latents.clone()
                    noised[:, :, prefix:] = pipe.scheduler.add_noise(gt_latents[:, :, prefix:], noise[:, :, prefix:], timestep)
                    noised[:, :, :prefix] = clean
                    shared["latents"] = noised
                    pipe.model_fn(**models, **shared, **positive, timestep=timestep)
        finally:
            capture_state.remove()
        expected = len(STEPS) * len(targets)
        if len(capture_state.records) != expected:
            raise RuntimeError(f"Captured {len(capture_state.records)}/{expected} records for {case_key}")
        frame_arrays = [np.asarray(frame.convert("RGB"), dtype=np.uint8) for frame in gt_frames]
        case_root = args.output_root / "cases" / case_key
        case_root.mkdir(parents=True, exist_ok=True)
        gt.save_gt_video(gt_frames, case_root / "gt.mp4", 30, 6)
        raw = {}
        region_catalog = []
        for region in regions:
            region_catalog.append({"name": region["name"], "phrase": region["phrase"]})
            region_points = points[region["start"]:region["end"]]
            for step in STEPS:
                head_maps = []
                for target in targets:
                    values = capture_state.records[(step, target["block"], target["head"])][region["start"]:region["end"]].mean(axis=0)
                    head_maps.append(values)
                    raw[f"s{step:02d}_b{target['block']:02d}_h{target['head']:02d}_{region['name']}"] = values
                    image_path = case_root / f"S{step:02d}" / region["name"] / f"rank{target['rank']:02d}_b{target['block']:02d}_h{target['head']:02d}.jpg"
                    render_strip(frame_arrays, values, context_frame, region["mask"], region_points, f"#{target['rank']:02d} B{target['block']:02d} H{target['head']:02d} | object-query mean | S{step:02d}", image_path)
                    meta = capture_state.meta[step]
                    catalog_records.append({
                        "case_key": case_key, "region": region["name"], "step": step,
                        "rank": target["rank"], "block": target["block"], "head": target["head"],
                        "pck32": target["pck32"], "query_count": len(region_points),
                        "timestep": meta["timestep"], "sigma": meta["sigma"], "aggregate": False,
                        "image": str(image_path.relative_to(args.output_root)),
                    })
                aggregate = np.mean(np.stack(head_maps), axis=0)
                raw[f"s{step:02d}_top30_mean_{region['name']}"] = aggregate
                aggregate_path = case_root / f"S{step:02d}" / region["name"] / "top30_mean.jpg"
                render_strip(frame_arrays, aggregate, context_frame, region["mask"], region_points, f"TOP30 HEAD MEAN | object-query mean | S{step:02d}", aggregate_path)
                meta = capture_state.meta[step]
                step_meta[step] = {"step": step, **meta}
                catalog_records.append({
                    "case_key": case_key, "region": region["name"], "step": step,
                    "rank": 0, "block": None, "head": None, "pck32": None,
                    "query_count": len(region_points), "timestep": meta["timestep"],
                    "sigma": meta["sigma"], "aggregate": True,
                    "image": str(aggregate_path.relative_to(args.output_root)),
                })
        np.savez_compressed(case_root / "attention_probabilities.npz", **raw)
        catalog_cases.append({
            "case_key": case_key, "family": case.get("family", "PyBullet"),
            "caption": base["caption"], "regions": region_catalog,
            "gt_video": str((case_root / "gt.mp4").relative_to(args.output_root)),
        })
        atomic_json(case_root / "complete.json", {"case_key": case_key, "records": len(raw)})
        del capture_state, gt_latents, shared, positive
        torch.cuda.empty_cache()
    atomic_json(args.output_root / "catalog.json", {
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "model": model_label, "experiment_manifest": str(experiment_manifest),
        "protocol": "GT_latent_teacher_forced_fixed_noise_object_query_global_softmax",
        "cases": catalog_cases, "steps": [step_meta[step] for step in STEPS],
        "targets": targets, "records": catalog_records,
    })
    write_status(args.output_root, "complete", "5 cases x 5 noise levels x Top30 heads rendered")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "capture"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--ranking", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--baseline-config", type=Path)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare":
        prepare_dataset(args)
        return
    if args.cache_root is None or (args.checkpoint is None and args.baseline_config is None):
        raise ValueError("capture requires --cache-root and a checkpoint or baseline config")
    capture(args)


if __name__ == "__main__":
    main()
