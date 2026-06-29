from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_FEATURE_ROOT = Path("/data/gaoya/agent-data/outputs/vjepa_wan_precheck")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run V-JEPA extraction over a baseline manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, default=DEFAULT_FEATURE_ROOT)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--target-frames", type=int, default=64)
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Value forwarded to extract_vjepa_features.py so the child process uses the intended GPU mapping.",
    )
    parser.add_argument(
        "--out-layers",
        nargs="+",
        type=int,
        default=[5, 7, 8, 9, 11, 17, 23],
        help="V-JEPA block indices to extract from the local V-JEPA 2.1 Large backbone copy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.expanduser().resolve().read_text())
    cases = manifest["cases"]
    if args.limit is not None:
        cases = cases[: args.limit]

    script_path = Path(__file__).with_name("extract_vjepa_features.py")
    run_root = args.feature_root.expanduser().resolve() / manifest["run_name"]
    run_root.mkdir(parents=True, exist_ok=True)

    for case in cases:
        case_id = case["case_id"]
        case_video = Path(case["video_path"])
        case_out = run_root / case_id
        cmd = [
            args.python_bin,
            str(script_path),
            "--video",
            str(case_video),
            "--output-dir",
            str(case_out),
            "--target-frames",
            str(args.target_frames),
            "--out-layers",
            *[str(x) for x in args.out_layers],
        ]
        if args.cuda_visible_devices is not None:
            cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    print(str(run_root))


if __name__ == "__main__":
    main()
