#!/usr/bin/env python3
"""Render a local HTML/SVG architecture view for the oracle-state Wan adapter."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize where the oracle-state adapter is inserted into Wan DiT."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/tmp/oracle_state_adapter_arch"),
    )
    parser.add_argument("--title", type=str, default="Oracle State Adapter in Wan DiT")
    parser.add_argument("--num_blocks", type=int, default=30)
    parser.add_argument("--dit_dim", type=int, default=3072)
    parser.add_argument("--adapter_hidden", type=int, default=1024)
    parser.add_argument("--sample_future_frames", type=int, default=13)
    parser.add_argument("--sample_future_latent_frames", type=int, default=4)
    parser.add_argument("--sample_total_latent_frames", type=int, default=6)
    parser.add_argument("--sample_prefix_latent_frames", type=int, default=2)
    parser.add_argument("--sample_spatial_tokens_per_frame", type=int, default=920)
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rect(x: int, y: int, w: int, h: int, text: str, fill: str, stroke: str = "#233043", text_color: str = "#eaf2ff", radius: int = 18, font_size: int = 16) -> str:
    lines = text.split("\n")
    line_height = font_size + 6
    total_text_height = line_height * len(lines)
    start_y = y + (h - total_text_height) / 2 + font_size - 2
    text_svg = []
    for idx, line in enumerate(lines):
        text_svg.append(
            f'<text x="{x + w/2:.1f}" y="{start_y + idx*line_height:.1f}" text-anchor="middle" '
            f'font-family="Segoe UI, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="{font_size}" fill="{text_color}">{html.escape(line)}</text>'
        )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        + "".join(text_svg)
    )


def arrow(x1: int, y1: int, x2: int, y2: int, color: str = "#8ce99a", width: int = 4, dashed: bool = False) -> str:
    dash = ' stroke-dasharray="10 8"' if dashed else ""
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
        f'stroke-linecap="round" marker-end="url(#arrowhead)"{dash}/>'
    )


def label(x: int, y: int, text: str, color: str = "#9fb2c8", size: int = 15, anchor: str = "start") -> str:
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" '
        f'font-family="Segoe UI, PingFang SC, Noto Sans CJK SC, sans-serif" font-size="{size}" fill="{color}">{html.escape(text)}</text>'
    )


def architecture_svg(args: argparse.Namespace) -> str:
    W, H = 1780, 1120
    main_x = 1020
    main_w = 640
    branch_x = 90
    branch_w = 620
    block_h = 78
    block_gap = 20

    branch_boxes = []
    y = 120
    branch_specs = [
        ("Future oracle state\n[B, K, N, 9]", "#15324a"),
        (f"State MLP + static id embeddings\nhidden={args.adapter_hidden}", "#184e57"),
        (f"Object tokens\n[B, K, N, {args.adapter_hidden}]", "#1f5f5b"),
        (f"Frame attention pool\n[B, K, {args.adapter_hidden}]", "#246d5a"),
        (f"Temporal encoder\n[B, K, {args.adapter_hidden}]", "#2b7a55"),
        (
            f"Time align to future latent axis\n[B, F_fut, {args.adapter_hidden}]\nexample: [B, {args.sample_future_latent_frames}, {args.adapter_hidden}]",
            "#337f4d",
        ),
    ]
    branch_centers = []
    for idx, (text, fill) in enumerate(branch_specs):
        h = 100 if idx in (1, 5) else 84
        branch_boxes.append(rect(branch_x, y, branch_w, h, text, fill=fill))
        branch_centers.append((branch_x + branch_w, y + h // 2))
        if idx < len(branch_specs) - 1:
            branch_boxes.append(arrow(branch_x + branch_w // 2, y + h, branch_x + branch_w // 2, y + h + 34, color="#74c0fc"))
        y += h + 34

    main_boxes = []
    y = 110
    main_boxes.append(rect(main_x, y, main_w, 86, "Video latents after VAE\n[B, 48, F, H, W]", fill="#2b1f57"))
    y += 122
    main_boxes.append(rect(main_x, y, main_w, 86, f"Patchify + flatten\nhidden states [B, F*h*w, {args.dit_dim}]", fill="#3b2a73"))
    y += 138
    stack_top = y
    stack_h = args.num_blocks * block_h + (args.num_blocks - 1) * block_gap
    main_boxes.append(
        f'<rect x="{main_x-20}" y="{stack_top-26}" width="{main_w+40}" height="{stack_h+56}" rx="28" fill="rgba(255,255,255,0.02)" stroke="#44506b" stroke-width="2"/>'
    )
    main_boxes.append(label(main_x - 4, stack_top - 38, f"Wan DiT block stack ({args.num_blocks} blocks)", color="#f8f9fa", size=22))

    shown = [0, 1, 2, args.num_blocks - 3, args.num_blocks - 2, args.num_blocks - 1]
    shown_set = set(shown)
    current_y = stack_top
    block_centers = {}
    for idx in range(args.num_blocks):
        if idx in shown_set:
            text = f"DiT block {idx}\ninput/output: [B, F*h*w, {args.dit_dim}]"
            main_boxes.append(rect(main_x, current_y, main_w, block_h, text, fill="#4c357a"))
            block_centers[idx] = current_y + block_h // 2
        elif idx == 3:
            dots_y = current_y + block_h // 2 + 5
            main_boxes.append(label(main_x + main_w // 2, dots_y, "⋮", color="#d0d8e8", size=38, anchor="middle"))
        current_y += block_h + block_gap

    tail_y = stack_top + stack_h + 36
    main_boxes.append(rect(main_x, tail_y, main_w, 86, "Head + unpatchify\nnoise prediction", fill="#3a2352"))

    arrows = [
        arrow(main_x + main_w // 2, 196, main_x + main_w // 2, 232, color="#c77dff"),
        arrow(main_x + main_w // 2, 318, main_x + main_w // 2, stack_top - 10, color="#c77dff"),
        arrow(main_x + main_w // 2, stack_top + stack_h + 28, main_x + main_w // 2, tail_y - 10, color="#c77dff"),
    ]

    film_box_x = 760
    film_box_w = 210
    mod_y = stack_top + 10
    arrows.append(arrow(branch_centers[-1][0] + 30, branch_centers[-1][1], film_box_x - 20, branch_centers[-1][1], color="#ffd43b"))
    main_boxes.append(rect(film_box_x, branch_centers[-1][1] - 48, film_box_w, 96, f"Per-block modulation head\n[B, F_fut, {args.dit_dim * 2}]\n→ gamma, beta", fill="#7c5c12"))

    for idx in shown:
        center_y = block_centers[idx]
        main_boxes.append(rect(film_box_x, center_y - 38, film_box_w, 76, f"FiLM @ block {idx}\nfuture only", fill="#72510f", font_size=15))
        arrows.append(arrow(film_box_x + film_box_w, center_y, main_x - 12, center_y, color="#ffd43b"))
        main_boxes.append(label(main_x + main_w + 26, center_y + 4, "apply_block_modulation()", color="#ffe066", size=15))

    extra = [
        label(branch_x, 64, "Adapter branch: encodes future state into a latent-time motion plan", color="#eaf2ff", size=22),
        label(main_x, 64, "Main branch: Wan DiT over spatiotemporal latent tokens", color="#eaf2ff", size=22),
        label(branch_x, 830, "Key point: the adapter does not replace Wan tokens.", color="#ffd43b", size=19),
        label(branch_x, 862, "It mainly produces per-block, per-future-frame gamma/beta modulation parameters.", color="#ffd43b", size=19),
        label(branch_x, 894, f"For the example sample: raw future K={args.sample_future_frames} → future latent frames F_fut={args.sample_future_latent_frames}.", color="#ffd43b", size=19),
        label(branch_x, 926, f"Context prefix uses zeros in gamma/beta, so only future latent frames are modulated.", color="#ffd43b", size=19),
        label(main_x, 992, f"Example latent grid: total F={args.sample_total_latent_frames}, prefix={args.sample_prefix_latent_frames}, spatial tokens/frame={args.sample_spatial_tokens_per_frame}.", color="#adb5bd", size=17),
    ]

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="arrowhead" markerWidth="12" markerHeight="10" refX="10" refY="5" orient="auto">
      <polygon points="0 0, 12 5, 0 10" fill="#8ce99a"/>
    </marker>
  </defs>
  <rect width="{W}" height="{H}" fill="#0b0f14"/>
  {''.join(branch_boxes)}
  {''.join(main_boxes)}
  {''.join(arrows)}
  {''.join(extra)}
</svg>"""


def build_html(args: argparse.Namespace) -> str:
    summary = {
        "num_blocks": args.num_blocks,
        "dit_dim": args.dit_dim,
        "adapter_hidden": args.adapter_hidden,
        "sample_future_frames": args.sample_future_frames,
        "sample_future_latent_frames": args.sample_future_latent_frames,
        "sample_total_latent_frames": args.sample_total_latent_frames,
        "sample_prefix_latent_frames": args.sample_prefix_latent_frames,
        "sample_spatial_tokens_per_frame": args.sample_spatial_tokens_per_frame,
    }
    code_snippet = """for block_id, block in enumerate(dit.blocks):
    x = gradient_checkpoint_forward(block, ...)
    if animate_adapter is not None and state_plan_tokens is not None:
        x = animate_adapter.apply_block_modulation(
            block_idx=block_id,
            hidden_states=x,
            future_plan_tokens=state_plan_tokens,
            total_frames=int(f),
            clean_prefix_len=int(clean_prefix_len),
            spatial_tokens_per_frame=spatial_tokens_per_frame,
        )"""
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{html.escape(args.title)}</title>
  <style>
    :root {{
      --bg: #0c1118;
      --card: #111923;
      --card2: #16212d;
      --text: #edf2f7;
      --muted: #99a9bd;
      --line: #273444;
      --accent: #ffd43b;
      --accent2: #74c0fc;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(116,192,252,0.12), transparent 24%),
        radial-gradient(circle at top right, rgba(255,212,59,0.10), transparent 22%),
        var(--bg);
    }}
    .page {{ width: min(1800px, calc(100vw - 24px)); margin: 18px auto 32px; }}
    .hero, .section {{
      background: linear-gradient(180deg, var(--card), var(--card2));
      border: 1px solid var(--line);
      border-radius: 22px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.22);
      padding: 22px 24px;
      margin-bottom: 18px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    p {{ margin: 8px 0; line-height: 1.7; color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-top: 14px;
    }}
    .metric {{
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.03);
    }}
    .metric .k {{ display:block; color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }}
    .metric .v {{ display:block; font-size: 23px; font-weight: 700; word-break: break-word; }}
    .diagram-wrap {{
      overflow-x: auto;
      border-radius: 20px;
      border: 1px solid var(--line);
      background: #091018;
      padding: 12px;
    }}
    .diagram-wrap img {{
      display: block;
      width: min(1780px, 100%);
      height: auto;
      margin: 0 auto;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px 18px;
      background: rgba(255,255,255,0.03);
    }}
    code {{
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 20px;
      color: var(--muted);
      line-height: 1.7;
    }}
    li + li {{ margin-top: 6px; }}
    .hint {{ color: #ffe066; }}
    pre {{
      margin: 0;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: #0b1320;
      overflow-x: auto;
      color: #d9e2ef;
      line-height: 1.55;
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(args.title)}</h1>
      <p>这张图回答两个问题：adapter 插在 Wan DiT 的哪些层，以及它在当前实现里主要做什么。结论先放前面：<span class="hint">adapter 挂在每一个 DiT block 后面，一共 {args.num_blocks} 处；它的主要职责确实就是从 future-state 条件里生成每层、每个 future latent frame 的 <code>gamma/beta</code> 调制参数。</span></p>
      <div class="metrics">
        <div class="metric"><span class="k">DiT blocks</span><span class="v">{args.num_blocks}</span></div>
        <div class="metric"><span class="k">DiT hidden dim</span><span class="v">{args.dit_dim}</span></div>
        <div class="metric"><span class="k">Adapter hidden dim</span><span class="v">{args.adapter_hidden}</span></div>
        <div class="metric"><span class="k">Future raw -> latent</span><span class="v">{args.sample_future_frames} -> {args.sample_future_latent_frames}</span></div>
      </div>
    </section>

    <section class="section">
      <h2>1. 架构图</h2>
      <div class="diagram-wrap">
        <img src="oracle_state_adapter_arch.svg" alt="Oracle state adapter architecture">
      </div>
      <p>图里左边是 adapter 分支，右边是 Wan DiT 主干。你会看到 adapter 不是插进 VAE、文本编码器或者输出头，而是在每个 DiT block 后面调用一次 <code>apply_block_modulation()</code>。</p>
    </section>

    <section class="section">
      <h2>2. adapter 主要是不是就在生成调制参数</h2>
      <div class="grid">
        <div class="card">
          <h3>是的，主职责就是这个</h3>
          <ul>
            <li>先把 future oracle state 编成一串 future-aligned plan tokens：<code>[B, F_fut, 1024]</code>。</li>
            <li>然后每个 block 各自用一个小 head，把这串 token 映射到 <code>[B, F_fut, 2*3072]</code>。</li>
            <li>最后切成 <code>gamma</code> 和 <code>beta</code>，用于 FiLM / bias-style modulation。</li>
          </ul>
        </div>
        <div class="card">
          <h3>但它不只是“线性层吐参数”</h3>
          <ul>
            <li>前面还有一个小状态编码器：状态 MLP、静态 embedding、帧内 pooling、时间编码。</li>
            <li>这一步的作用是先把“未来物体状态轨迹”压成一个 latent-time motion plan。</li>
            <li>真正生成调制参数的是每层自己的 <code>modulation_heads[block_id]</code>。</li>
          </ul>
        </div>
      </div>
    </section>

    <section class="section">
      <h2>3. 插入层的精确定义</h2>
      <p>当前实现不是只插在某几个 block，也不是只插 temporal-only block。因为这份 Wan 训练代码里没有单独暴露“纯 temporal blocks”列表，所以我们是在 block 循环里，对 <code>dit.blocks</code> 的每一个 block 后面都做一次调制；同时通过把 hidden states reshape 成 <code>[B, F, spatial_tokens_per_frame, C]</code>，只对 future 帧位置施加非零调制，context 前缀补零。</p>
      <pre>{html.escape(code_snippet)}</pre>
    </section>

    <section class="section">
      <h2>4. 当前页面配置</h2>
      <pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
      <p>生成这张图的脚本就在：<code>/home/gaoya/Code_Video/Code_data/Code_train/train_0419/state_adapter/visualizations/visualize_oracle_state_adapter_arch.py</code></p>
    </section>
  </div>
</body>
</html>"""


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    svg = architecture_svg(args)
    (args.output_dir / "oracle_state_adapter_arch.svg").write_text(svg, encoding="utf-8")
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "num_blocks": args.num_blocks,
                "dit_dim": args.dit_dim,
                "adapter_hidden": args.adapter_hidden,
                "sample_future_frames": args.sample_future_frames,
                "sample_future_latent_frames": args.sample_future_latent_frames,
                "sample_total_latent_frames": args.sample_total_latent_frames,
                "sample_prefix_latent_frames": args.sample_prefix_latent_frames,
                "sample_spatial_tokens_per_frame": args.sample_spatial_tokens_per_frame,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "index.html").write_text(build_html(args), encoding="utf-8")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
