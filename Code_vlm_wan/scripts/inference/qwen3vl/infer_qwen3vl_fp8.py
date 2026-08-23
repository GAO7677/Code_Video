#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Official-style transformers demo for Qwen3-VL-32B-Thinking-FP8.

Usage:
    python infer_qwen3vl_fp8.py \
        --ckpt /home/gaoya/data/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8 \
        --image demo_receipt.png \
        --prompt "Read all the text in the image." \
        --max-new-tokens 512
"""
import argparse
import time

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="/home/gaoya/data/ckpt/Qwen-Qwen3-VL-32B-Thinking-FP8")
    parser.add_argument("--image", default="demo_receipt.png")
    parser.add_argument("--prompt", default="Read all the text in the image.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    print(f"[1/4] Loading processor from {args.ckpt}")
    processor = AutoProcessor.from_pretrained(args.ckpt, trust_remote_code=True)

    print("[2/4] Loading model (FP8 fine-grained) ...")
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt,
        dtype="auto",
        attn_implementation="sdpa",
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"      model loaded in {time.time() - t0:.1f}s")
    print(f"      device_map keys: {list(model.hf_device_map.keys()) if hasattr(model, 'hf_device_map') else 'N/A'}")
    model.eval()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    print("\n===== INPUT (messages) =====")
    print(messages)

    print("\n[3/4] Tokenizing / preparing inputs ...")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    print("===== INPUT (tensor shapes) =====")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__} (len={len(v) if hasattr(v, '__len__') else '?'})")

    # show the actual prompt string that was fed to the model
    prompt_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    print("\n===== INPUT (decoded prompt string) =====")
    print(prompt_text)

    print("\n[4/4] Generating ...")
    t1 = time.time()
    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            top_k=20,
        )
    dt = time.time() - t1
    n_new = generated_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"      generated {n_new} tokens in {dt:.1f}s ({n_new / dt:.1f} tok/s)")

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n===== OUTPUT (generated text) =====")
    print(output_text)

    # split thinking vs answer if the model emitted <think>...</think>
    if "<｜end▁of▁thinking｜>" in output_text:
        print("\n===== OUTPUT (final answer only) =====")
        print(output_text.split("<｜end▁of▁thinking｜>", 1)[1].strip())


if __name__ == "__main__":
    main()
