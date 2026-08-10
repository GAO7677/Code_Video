#!/usr/bin/env python3
"""Extract the frozen EMA encoder from the full V-JEPA2.1 training checkpoint."""

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    if "ema_encoder" not in payload:
        raise KeyError(f"Checkpoint has no ema_encoder: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_suffix(f"{output.suffix}.tmp")
    torch.save(
        {
            "ema_encoder": payload["ema_encoder"],
            "source_checkpoint": str(source),
            "source_epoch": payload.get("epoch"),
        },
        temporary_output,
    )
    os.replace(temporary_output, output)
    print(
        f"extracted_tensors={len(payload['ema_encoder'])} "
        f"output={output} size_gib={output.stat().st_size / 1024**3:.3f}"
    )


if __name__ == "__main__":
    main()
