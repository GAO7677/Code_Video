from __future__ import annotations

import argparse
import http.server
import socketserver
from pathlib import Path

from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_ball_block_ti2v_vjepa_vggt.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="/data/gaoya/AAA_test_video/0529/vjepa_vggt/reports/object_pipeline_report",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    trainer = ContextVideoTrainer(cfg, build_optimizer=False)
    html_path = trainer.write_inspection_report(args.output_dir)
    print(f"inspection report: {html_path}")

    if not args.serve:
        return

    output_dir = Path(args.output_dir)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    with socketserver.TCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{args.port}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
