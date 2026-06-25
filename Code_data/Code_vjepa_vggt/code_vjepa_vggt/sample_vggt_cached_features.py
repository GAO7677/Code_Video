from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from code_vjepa_vggt.utils.vggt_cache import sample_dense_patch_tokens_at_query_points


def _load_query_points(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, torch.Tensor):
            return payload
        if isinstance(payload, dict):
            for key in ("query_points", "points", "xy"):
                if key in payload:
                    value = payload[key]
                    if isinstance(value, torch.Tensor):
                        return value
                    return torch.as_tensor(value)
        raise ValueError(f"unsupported pt payload in {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("query_points", "points", "xy"):
                if key in data:
                    return torch.as_tensor(data[key], dtype=torch.float32)
        raise ValueError(f"unsupported json payload in {path}")
    rows = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        rows.append([float(parts[0]), float(parts[1])])
    if not rows:
        raise ValueError(f"no query points found in {path}")
    return torch.tensor(rows, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample cached VGGT dense features at query points.")
    parser.add_argument("--cache-file", required=True, help="path to a *.vggt.pt cache file")
    parser.add_argument("--query-points", required=True, help="query points file: .pt / .json / .txt")
    parser.add_argument("--output-file", required=True, help="output .pt file")
    parser.add_argument("--frame-ids", default=None, help="optional frame ids, one per query point or per time step")
    args = parser.parse_args()

    cache_path = Path(args.cache_file).expanduser().resolve()
    payload = torch.load(cache_path, map_location="cpu")
    dense_patch_tokens = payload["dense_patch_tokens"]
    if not isinstance(dense_patch_tokens, torch.Tensor):
        dense_patch_tokens = torch.as_tensor(dense_patch_tokens)
    if dense_patch_tokens.ndim == 4:
        dense_patch_tokens = dense_patch_tokens.unsqueeze(0)
    patch_grid_hw = tuple(int(v) for v in payload["patch_grid_hw"])
    input_hw = tuple(int(v) for v in payload["input_hw"])

    query_points = _load_query_points(Path(args.query_points).expanduser().resolve()).float()
    if query_points.ndim == 2:
        query_points = query_points
    elif query_points.ndim == 3:
        query_points = query_points
    else:
        raise ValueError(f"query points must have shape [N,2] or [T,N,2], got {list(query_points.shape)}")

    if args.frame_ids is not None:
        frame_ids = torch.as_tensor([int(v) for v in str(args.frame_ids).split(",") if v.strip()], dtype=torch.long)
        if query_points.ndim == 3 and frame_ids.numel() != query_points.shape[0]:
            raise ValueError(
                f"frame_ids count {int(frame_ids.numel())} does not match query time dimension {int(query_points.shape[0])}"
            )
    sampled = sample_dense_patch_tokens_at_query_points(
        dense_patch_tokens,
        query_points,
        image_hw=input_hw,
    )

    output = {
        "cache_file": str(cache_path),
        "patch_grid_hw": list(patch_grid_hw),
        "input_hw": list(input_hw),
        "query_points": query_points,
        "sampled_features": sampled.squeeze(0).contiguous(),
        "sampled_features_shape": list(sampled.squeeze(0).shape),
    }
    out_path = Path(args.output_file).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, out_path)
    print(json.dumps({k: v if isinstance(v, (str, int, float, list, dict, bool)) else str(type(v)) for k, v in output.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
