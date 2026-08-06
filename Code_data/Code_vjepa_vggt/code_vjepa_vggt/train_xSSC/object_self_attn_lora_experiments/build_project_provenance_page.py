#!/usr/bin/env python3
"""Build the compact project, weight-lineage, and head-provenance page."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
from pathlib import Path
import shutil
from typing import Any


OPENVID_LORA = (
    "/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/"
    "openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/"
    "checkpoint.safetensors"
)
XSSC_26K = (
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/"
    "restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-026000.pth"
)
XSSC_50K = XSSC_26K.replace("step-026000.pth", "step-050000.pth")
OFFICIAL_XSSC = "/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth"
HEAD_GALLERY = Path(
    "/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery"
)
EXPERIMENT_ROOT = Path(__file__).resolve().parent
PROJECT_MARKDOWN = EXPERIMENT_ROOT / "PROJECT_INFO_AND_PROVENANCE.md"

CORE_PARAMETER_ROWS = [
    ("Wan DiT", 4_999_787_712, "冻结；OpenVid LoRA先合并到该权重"),
    ("UMT5", 5_680_910_336, "冻结文本编码器"),
    ("Wan VAE", 704_688_668, "冻结视频编码/解码器"),
    ("DINOv3 ViT-L/16", 303_129_600, "Object实验冻结"),
    ("xSSC非Backbone", 81_043_968, "Wan训练时冻结；xSSC训练时可训练"),
    ("SAM2.1 Hiera-L", 224_446_898, "冻结，仅生成AMG pseudo boxes"),
]
WAN_CORE_PARAMS = sum(value for _, value, _ in CORE_PARAMETER_ROWS[:3])
DINO3_OBJECT_FROZEN_PARAMS = sum(value for _, value, _ in CORE_PARAMETER_ROWS[3:])
OFFICIAL_XSSC_PARAMS = 34_048_128
OFFICIAL_OBJECT_FROZEN_PARAMS = 224_446_898 + OFFICIAL_XSSC_PARAMS

METHOD_SPECS = {
    "object_only": {
        "label": "Object-only",
        "trainable": 25_458_688,
        "modules": "projector/time 1.774M + object-attn LoRA 23.593M + gate 0.092M",
        "object": "xSSC-26k",
        "data": "PyBullet/Kubric/OpenVid = 30/30/40",
    },
    "full_sa_resume": {
        "label": "Full-SA + Object",
        "trainable": 49_051_648,
        "modules": "Object-only模块 + 30层Self-Attn Q/K/V/O LoRA 23.593M",
        "object": "xSSC-26k",
        "data": "30/30/40",
    },
    "s_head59_resume": {
        "label": "S-head59 + Object",
        "trainable": 34_682_880,
        "modules": "Object-only模块 + same-frame S59 compact head LoRA 9.224M",
        "object": "xSSC-26k",
        "data": "30/30/40",
    },
    "t_head70": {
        "label": "T-head70 + Object",
        "trainable": 34_863_104,
        "modules": "Object-only模块 + T70 compact head LoRA 9.404M",
        "object": "xSSC-26k",
        "data": "30/30/40",
    },
    "slot_dedup_merge": {
        "label": "Full-SA + Object + Slot-Dedup",
        "trainable": 49_051_648,
        "modules": "与Full-SA + Object相同；Dedup不新增参数",
        "object": "xSSC-26k；merge@cosine 0.94，min_keep=3",
        "data": "30/30/40",
    },
    "slot_dedup_merge_xssc_step050000": {
        "label": "Full-SA + Object + Dedup (xSSC-50k)",
        "trainable": 49_051_648,
        "modules": "与Full-SA + Object相同；Dedup不新增参数",
        "object": "xSSC-50k；merge@cosine 0.94，min_keep=3",
        "data": "30/30/40",
    },
    "t_head70_slot_dedup_merge_xssc_step050000": {
        "label": "T-head70 + Object + Dedup (xSSC-50k)",
        "trainable": 34_863_104,
        "modules": "Object-only模块 + T70 compact head LoRA；Dedup零参数",
        "object": "xSSC-50k",
        "data": "30/30/40",
    },
    "full_sa_no_object": {
        "label": "Full-SA + No-Object",
        "trainable": 23_592_960,
        "modules": "30层Self-Attn Q/K/V/O LoRA",
        "object": "完全跳过SAM2、DINO、xSSC和object cross-attn",
        "data": "30/30/40",
    },
    "full_sa_no_object_vjepa_loss": {
        "label": "Full-SA + No-Object + V-JEPA Loss",
        "trainable": 23_592_960,
        "modules": "30层Self-Attn Q/K/V/O LoRA；训练时增加冻结V-JEPA2.1/Tiny-VAE辅助损失",
        "object": "完全跳过SAM2、DINO、xSSC和object cross-attn",
        "data": "30/30/40",
    },
    "full_sa_no_object_pybullet100": {
        "label": "Full-SA + No-Object (PyBullet 100%)",
        "trainable": 23_592_960,
        "modules": "30层Self-Attn Q/K/V/O LoRA",
        "object": "关闭",
        "data": "100/0/0",
    },
    "full_sa_no_object_kubric100": {
        "label": "Full-SA + No-Object (Kubric 100%)",
        "trainable": 23_592_960,
        "modules": "30层Self-Attn Q/K/V/O LoRA",
        "object": "关闭",
        "data": "0/100/0",
    },
    "t_head70_no_object": {
        "label": "T-head70 + No-Object",
        "trainable": 9_404_416,
        "modules": "仅T70 compact head LoRA",
        "object": "关闭",
        "data": "30/30/40",
    },
    "t_head100_lora_pck32_no_object": {
        "label": "Motion-head100 (PCK32 Top100) + No-Object",
        "trainable": 11_075_584,
        "modules": "仅Motion Top100 compact head LoRA",
        "object": "关闭",
        "data": "30/30/40",
    },
    "official_xssc_object_only": {
        "label": "Object-only + Official DINOv2 xSSC",
        "trainable": 24_671_744,
        "modules": "projector/time 0.987M + object-attn LoRA 23.593M + gate 0.092M",
        "object": "官方rsfq2_r-ytvis/42-0130.pth",
        "data": "30/30/40",
    },
}

EXTRA_RUNS = [
    {
        "key": "object_only",
        "label": "Object-only",
        "watch_roots": [
            "/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/"
            "object_only_gpu1_formal/formal_20260729T184553Z/checkpoints"
        ],
    },
    {
        "key": "official_xssc_object_only",
        "label": "Object-only + Official DINOv2 xSSC",
        "watch_roots": [
            "/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/"
            "official_xssc_object_only_gpu01_formal_1500/formal_20260803T084903Z/checkpoints"
        ],
    },
]


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_params(value: int) -> str:
    return f"{value / 1_000_000:.3f}M"


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def replace_symlink(source: Path, destination: Path) -> None:
    if not source.is_dir():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(source.resolve(), target_is_directory=True)
    if destination.is_symlink() or destination.is_file():
        os.replace(temporary, destination)
    elif destination.exists():
        temporary.unlink(missing_ok=True)
    else:
        os.replace(temporary, destination)


def find_steps(root: Path) -> list[int]:
    steps: list[int] = []
    if not root.is_dir():
        return steps
    for path in root.glob("step-*"):
        try:
            steps.append(int(path.name.removeprefix("step-")))
        except ValueError:
            continue
    return sorted(set(steps))


def read_run(root: Path) -> dict[str, Any]:
    run_root = root.parent
    resolved_path = run_root / "resolved_experiment_config.json"
    resolved: dict[str, Any] = {}
    if resolved_path.is_file():
        resolved = load_json(resolved_path)
    cfg = resolved.get("resolved_config", {})
    paths = cfg.get("paths", {})
    adaptation = cfg.get("adaptation", {})
    checkpointing = cfg.get("checkpointing", {})
    head_snapshot = resolved.get("head_selection_snapshot") or {}
    object_enabled = bool(adaptation.get("enable_object_branch", True))
    return {
        "run_root": str(run_root),
        "resolved_config": str(resolved_path) if resolved_path.is_file() else "",
        "steps": find_steps(root),
        "resume_from": checkpointing.get("resume_from"),
        "xssc_checkpoint": paths.get("xssc_checkpoint") if object_enabled else None,
        "config_sources": resolved.get("config_sources", []),
        "head_snapshot_sha256": head_snapshot.get("sha256"),
    }


def inventory_methods(config: dict[str, Any]) -> list[dict[str, Any]]:
    methods: list[dict[str, Any]] = []
    seen: set[str] = set()
    for method in [*EXTRA_RUNS, *config.get("methods", [])]:
        key = str(method["key"])
        roots = [Path(value).resolve() for value in method.get("watch_roots", [])]
        signature = key + "|" + "|".join(str(root) for root in roots)
        if signature in seen:
            continue
        seen.add(signature)
        methods.append(
            {
                "key": key,
                "label": method.get("label", METHOD_SPECS.get(key, {}).get("label", key)),
                "runs": [read_run(root) for root in roots],
            }
        )
    return methods


def write_inventory_csv(path: Path, methods: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method_key",
                "method_label",
                "trainable_params",
                "run_root",
                "checkpoint_steps",
                "resume_from",
                "xssc_checkpoint",
                "resolved_config",
                "head_snapshot_sha256",
            ],
        )
        writer.writeheader()
        for method in methods:
            spec = METHOD_SPECS.get(method["key"], {})
            for run in method["runs"] or [{}]:
                writer.writerow(
                    {
                        "method_key": method["key"],
                        "method_label": method["label"],
                        "trainable_params": spec.get("trainable", ""),
                        "run_root": run.get("run_root", ""),
                        "checkpoint_steps": ",".join(map(str, run.get("steps", []))),
                        "resume_from": run.get("resume_from") or "",
                        "xssc_checkpoint": run.get("xssc_checkpoint") or "",
                        "resolved_config": run.get("resolved_config", ""),
                        "head_snapshot_sha256": run.get("head_snapshot_sha256") or "",
                    }
                )


def method_rows(methods: list[dict[str, Any]]) -> str:
    order: list[str] = []
    labels: dict[str, str] = {}
    for method in methods:
        if method["key"] not in order:
            order.append(method["key"])
        labels[method["key"]] = str(method["label"])
    rows: list[str] = []
    for key in order:
        spec = METHOD_SPECS.get(key)
        if spec is None:
            continue
        if "no_object" in key:
            frozen = WAN_CORE_PARAMS
        elif key == "official_xssc_object_only":
            frozen = WAN_CORE_PARAMS + OFFICIAL_OBJECT_FROZEN_PARAMS
        else:
            frozen = WAN_CORE_PARAMS + DINO3_OBJECT_FROZEN_PARAMS
        total = frozen + int(spec["trainable"])
        rows.append(
            f"<tr><th>{esc(labels.get(key, spec['label']))}</th>"
            f"<td>{esc(spec['modules'])}</td><td>{fmt_params(spec['trainable'])}</td>"
            f"<td>{fmt_params(frozen)} / {fmt_params(total)}</td>"
            f"<td>{esc(spec['object'])}</td><td>{esc(spec['data'])}</td></tr>"
        )
    return "".join(rows)


def checkpoint_blocks(methods: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for method in methods:
        runs: list[str] = []
        for run in method["runs"]:
            steps = run.get("steps", [])
            step_text = ", ".join(f"{step:,}" for step in steps) or "暂无checkpoint"
            details = [
                f"<div><b>Run</b><code>{esc(run['run_root'])}</code></div>",
                f"<div><b>Steps</b><span>{esc(step_text)}</span></div>",
            ]
            if run.get("xssc_checkpoint"):
                details.append(
                    f"<div><b>xSSC</b><code>{esc(run['xssc_checkpoint'])}</code></div>"
                )
            if run.get("resume_from"):
                details.append(
                    f"<div><b>Resume</b><code>{esc(run['resume_from'])}</code></div>"
                )
            if run.get("resolved_config"):
                details.append(
                    f"<div><b>配置快照</b><code>{esc(run['resolved_config'])}</code></div>"
                )
            if run.get("head_snapshot_sha256"):
                details.append(
                    f"<div><b>Head SHA256</b><code>{esc(run['head_snapshot_sha256'])}</code></div>"
                )
            runs.append("<div class='run'>" + "".join(details) + "</div>")
        blocks.append(
            f"<details><summary>{esc(method['label'])}<span>{len(method['runs'])} run</span>"
            f"</summary>{''.join(runs) or '<p>暂无已登记run。</p>'}</details>"
        )
    return "".join(blocks)


def build_html(methods: list[dict[str, Any]]) -> str:
    core_total = WAN_CORE_PARAMS
    object_frozen = DINO3_OBJECT_FROZEN_PARAMS
    core_rows = "".join(
        f"<tr><th>{esc(name)}</th><td>{fmt_params(value)}</td><td>{esc(status)}</td></tr>"
        for name, value, status in CORE_PARAMETER_ROWS
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC-Wan 项目信息与权重溯源</title>
<style>
:root{{--bg:#f4f6f6;--paper:#fff;--ink:#172126;--muted:#66757b;--line:#d7dfe1;
--teal:#006d77;--red:#a23b32;--green:#28745b;--amber:#95640d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.55 Inter,"Noto Sans SC",Arial,sans-serif}}
header{{background:var(--paper);border-bottom:1px solid var(--line);padding:22px max(20px,calc((100% - 1220px)/2))}}
h1{{font-size:25px;margin:0 0 5px}}h2{{font-size:19px;margin:0 0 12px}}h3{{font-size:15px;margin:0 0 5px}}
p{{margin:5px 0}}a{{color:var(--teal);font-weight:750;text-decoration:none}}header a{{display:inline-block;margin-top:10px}}
main{{max-width:1220px;margin:auto;padding:20px}}section{{padding:19px 0;border-bottom:1px solid var(--line)}}
.lede{{font-size:16px;max-width:1050px}}.flow{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:7px;margin-top:14px}}
.flow div{{background:var(--paper);border:1px solid var(--line);border-top:3px solid var(--teal);padding:11px;border-radius:5px}}
.flow b,.flow span{{display:block}}.flow span{{color:var(--muted);font-size:12px;margin-top:4px}}
.table-wrap{{overflow:auto;border:1px solid var(--line);background:var(--paper);border-radius:5px}}
table{{width:100%;border-collapse:collapse;min-width:760px}}th,td{{padding:9px 10px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}}
thead th{{background:#eaf0f1;font-size:12px}}tbody th{{white-space:nowrap}}code{{display:block;white-space:normal;overflow-wrap:anywhere;color:#3d5057;font:12px/1.45 monospace}}
.note{{padding:10px 12px;border-left:3px solid var(--amber);background:#fff8e8;margin:10px 0}}
.ok{{border-left-color:var(--green);background:#edf7f2}}.warn{{border-left-color:var(--red);background:#fff0ee}}
.links{{display:flex;gap:8px;flex-wrap:wrap}}.links a{{padding:7px 10px;background:#e6f1f1;border-radius:4px}}
details{{background:var(--paper);border:1px solid var(--line);border-radius:5px;margin:7px 0}}summary{{cursor:pointer;padding:10px 12px;font-weight:800}}
summary span{{float:right;color:var(--muted);font-weight:500}}.run{{padding:10px 12px;border-top:1px solid var(--line);display:grid;gap:5px}}
.run b{{display:inline-block;min-width:76px}}.run span{{color:var(--muted)}}.run code{{margin-top:2px}}
.formula{{font-family:monospace;font-size:12px}}.small{{font-size:12px;color:var(--muted)}}
@media(max-width:850px){{.flow{{grid-template-columns:1fr 1fr}}}}@media(max-width:540px){{main{{padding:12px}}.flow{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>xSSC-Wan 项目信息与权重溯源</h1>
<p>方法、训练模块、参数规模、权重来源与 Head 分类证据的统一索引。</p><a href="../">返回8844总入口</a></header>
<main>
<section><h2>项目目标与前向流程</h2>
<p class="lede">项目以 Wan2.2 TI2V-5B 为生成主干，把冻结 xSSC 从8帧上下文提取的对象级 slots 注入 DiT，并比较全 Self-Attention、特定功能 Head、Object分支和 Slot去重对物理合理视频生成的影响。</p>
<div class="flow"><div><b>8帧 Context</b><span>SAM2 AMG生成并过滤pseudo boxes</span></div>
<div><b>冻结 DINOv3+xSSC</b><span>slots [B,8,11,512]</span></div>
<div><b>可选 Slot-Dedup</b><span>跨时间track相似度merge；不新增参数</span></div>
<div><b>Object Tokens</b><span>LN + Linear + time embedding → [B,88,3072]</span></div>
<div><b>Wan DiT</b><span>49帧、512×896；object cross-attn与Self-Attn LoRA</span></div></div></section>

<section><h2>权重主链</h2>
<div class="flow"><div><b>Wan Base</b><span>Wan2.2 TI2V-5B</span></div><div><b>OpenVid LoRA 10k</b><span>先合并进Wan并卸载旧PEFT wrapper</span></div>
<div><b>xSSC 26k / 50k</b><span>DINOv3 MOVi-C迁移训练；Wan阶段冻结</span></div><div><b>实验适配器</b><span>Object / Full-SA / S59 / T70 / Motion100</span></div><div><b>Checkpoint评测</b><span>test_5与PhysicIQ，完整配置见下方</span></div></div>
<p class="small">OpenVid LoRA合并后已成为主干初始化，不作为当前训练的独立可训练LoRA；当前checkpoint只保存新实验增量。</p></section>

<section><h2>上游 OpenVid LoRA · step-010000</h2>
<div class="table-wrap"><table><thead><tr><th>数据</th><th>有效样本</th><th>训练设置</th><th>可训练模块</th></tr></thead><tbody>
<tr><td>OpenVid 65,975 (85.66%)<br>MOVI-D 3,083×2=6,166 (8.01%)<br>Genesis rigid 2,438×2=4,876 (6.33%)</td><td>77,017</td>
<td>10,000 optimizer steps；4卡×batch1×acc4，有效batch16；24帧384×672；AdamW lr=1e-4；约160,000次clip使用。checkpoint时间跨度约64小时，含中断/resume；最后8k→10k纯到步约4小时29分。</td>
<td>rank32：30层Self-Attn Q/K/V/O、文本Cross-Attn Q/K/V/O、FFN 0/2，共300个LoRA模块、80.609M参数。</td></tr></tbody></table></div>
<code>{esc(OPENVID_LORA)}</code></section>

<section><h2>模型参数口径</h2>
<div class="table-wrap"><table><thead><tr><th>组件</th><th>参数量</th><th>状态</th></tr></thead><tbody>{core_rows}</tbody></table></div>
<p class="small">Wan+UMT5+VAE核心合计 {fmt_params(core_total)}。Object辅助组件合计 {fmt_params(object_frozen)}。这是组件参数清单，不代表全部同时常驻GPU。</p>
<div class="note ok">DINOv3+xSSC总计384.174M，其中DINOv3 303.130M冻结；xSSC非Backbone 81.044M在xSSC训练中可训练，但在Wan训练中冻结。</div></section>

<section><h2>xSSC 权重来源</h2>
<div class="table-wrap"><table><thead><tr><th>权重</th><th>训练数据与过程</th><th>训练模块</th><th>Wan中用途</th></tr></thead><tbody>
<tr><th>DINOv3 xSSC-26k</th><td>MOVi-C；从15k迁移续训，6帧随机clip，256×256；训练链最终到50k</td><td>Encoder MLP、initialization、SlotAttention、transition、4层decoder；DINOv3冻结</td><td>多数Object实验</td></tr>
<tr><th>DINOv3 xSSC-50k</th><td>与26k同一run的最终权重；2卡×64×acc3，有效batch384</td><td>同上</td><td>标记为xSSC-50k的Dedup实验</td></tr>
<tr><th>官方DINOv2 xSSC</th><td>官方 rsfq2_r-ytvis 配置和权重；checkpoint共34.048M参数</td><td>在Wan实验中整体冻结</td><td>Official xSSC Object-only对照</td></tr></tbody></table></div>
<code>{esc(XSSC_26K)}</code><code>{esc(XSSC_50K)}</code><code>{esc(OFFICIAL_XSSC)}</code></section>

<section><h2>Wan 实验：训练模块与参数</h2>
<div class="table-wrap"><table><thead><tr><th>方法</th><th>实际可训练模块</th><th>可训练参数</th><th>冻结 / 总涉及参数</th><th>Object条件</th><th>训练数据</th></tr></thead><tbody>{method_rows(methods)}</tbody></table></div>
<p class="small">所有方法都以已合并OpenVid step-010000的Wan为初始化。Object实验中的SAM2、DINOv3、xSSC均冻结；No-Object实验完全跳过这些模块。</p></section>

<section><h2>Head 分类来源</h2>
<p>原始实验覆盖3个模型、20个case、22个三模型共同完成的seed、去噪步5/15/25/35，以及30×24=720个Self-Attention Head。每条记录保留原始特征，不只保存rank与最终score。</p>
<div class="table-wrap"><table><thead><tr><th>类别</th><th>分数</th><th>公共稳定数量</th></tr></thead><tbody>
<tr><th>S</th><td class="formula">0.55 rank(local_enrichment) + 0.45 rank(same_frame_mass)</td><td>159</td></tr>
<tr><th>T</th><td class="formula">0.55 rank(trajectory_selectivity_log2) + 0.25 rank(trajectory_enrichment) + 0.20 rank(mean_time_distance)</td><td>13</td></tr>
<tr><th>P</th><td class="formula">0.75 rank(fixed_position_enrichment) + 0.25 rank(aligned_enrichment)</td><td>82</td></tr>
<tr><th>C</th><td class="formula">0.55 rank(object_context_enrichment) + 0.25 rank(full_context_enrichment) + 0.20 rank(history_bias)</td><td>20</td></tr>
<tr><th>G</th><td class="formula">0.60 rank(full_entropy) + 0.25 rank(full_mean_time_distance) + 0.15 rank(-same_frame_mass)</td><td>75</td></tr>
<tr><th>M</th><td>跨模型不一致、margin&lt;0.08或四步一致率&lt;0.75</td><td>371</td></tr></tbody></table></div>
<p class="small">只有三个模型得到相同且非M的角色，才进入公共稳定S/T/P/C/G。轨迹少于8个有效时刻或有效率低于0.8时禁用T/P判定。</p>
<div class="links"><a href="../head-evidence/fulltoken-head-classification.html">原始分类Pilot</a><a href="../head-evidence/head_roles_50seeds/">多seed稳定性</a><a href="../head-evidence/common-stc-all-heads-qk-seed851/">各类Head Q@K</a><a href="../head-evidence/head-role-depth-distribution/">最终类别深度分布</a></div>
<div class="note">S159进一步互斥划分为local-dominant 100和same-frame-dominant 59；当前S-head训练使用后者59个。</div>
<div class="note warn"><b>口径限制：</b>T-head70是冻结的用户提供训练名单，不等于公共稳定T13，名单身份和哈希可追溯，但上游阈值/完整筛选链不足；Motion-head100来自单case、单seed的Wan+LoRA Object PCK@32 Top100，也不是语义T类。</div>
<code>/data/gaoya/agent-data/outputs/head_classification_csv/common22_public_stable/head_classification_all_720.csv</code>
<code>/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/head_classification</code></section>

<section><h2>Checkpoint 与配置快照</h2><p>以下内容由watcher配置和run目录动态扫描。展开后可核对checkpoint step、resume父权重、xSSC路径、配置快照及Head名单哈希。</p>
{checkpoint_blocks(methods)}
<div class="links"><a href="PROJECT_INFO_AND_PROVENANCE.md">查看Markdown文档</a><a href="checkpoint_inventory.csv">下载checkpoint清单CSV</a><a href="../checkpoint-watch/">打开自动评测</a><a href="../physiciq-average-metrics/">打开平均指标表</a></div></section>

<section><h2>主要代码与配置</h2>
<code>{esc(EXPERIMENT_ROOT / 'train_xssc_object_self_attn_lora.py')}</code>
<code>{esc(EXPERIMENT_ROOT / 'configs/base.json')}</code>
<code>{esc(EXPERIMENT_ROOT / 'build_xssc_lora_checkpoint_dashboard.py')}</code>
<code>/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAtrain.md</code></section>
</main></body></html>"""


def build_project_info_page(config: dict[str, Any], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    methods = inventory_methods(config)
    write_inventory_csv(output_root / "checkpoint_inventory.csv", methods)
    (output_root / "provenance.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "openvid_lora": OPENVID_LORA,
                "xssc_26k": XSSC_26K,
                "xssc_50k": XSSC_50K,
                "official_xssc": OFFICIAL_XSSC,
                "methods": methods,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    path = output_root / "index.html"
    path.write_text(build_html(methods), encoding="utf-8")
    if PROJECT_MARKDOWN.is_file():
        shutil.copy2(PROJECT_MARKDOWN, output_root / PROJECT_MARKDOWN.name)
    replace_symlink(HEAD_GALLERY, output_root.parent / "head-evidence")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    print(build_project_info_page(config, args.output.resolve()))


if __name__ == "__main__":
    main()
