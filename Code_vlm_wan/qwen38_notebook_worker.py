"""Persistent Qwen3.8 worker used from a Jupyter kernel.

The worker loads the processor and model once per kernel, then processes case
videos sequentially.  Keeping the object in a notebook variable avoids a
second model load when multiple cases are submitted from later cells.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any, Iterable

import kernels
import torch
import transformers
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen35


class Qwen38NotebookWorker:
    def __init__(
        self,
        model_path: str | Path = "/data/gaoya/ckpt/Qwen-Qwen3.8-27B-FP8",
        physical_gpu: int = 7,
        max_memory_gib: int = 46,
    ) -> None:
        self.model_path = str(model_path)
        self.physical_gpu = physical_gpu
        started = time.time()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, local_files_only=True, trust_remote_code=True
        )
        self.model, loading_info = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            torch_dtype="auto",
            device_map={"": "cuda:0"},
            low_cpu_mem_usage=True,
            max_memory={0: f"{max_memory_gib}GiB"},
            trust_remote_code=True,
            output_loading_info=True,
        )
        self.model.eval()
        self.loading = {
            key: sorted(value) if isinstance(value, (set, list, tuple)) else value
            for key, value in loading_info.items()
        }
        self.model_loaded_seconds = round(time.time() - started, 3)
        print(
            f"model_loaded_seconds={self.model_loaded_seconds} "
            f"visible_gpu={torch.cuda.get_device_name(0)} physical_gpu={physical_gpu}",
            flush=True,
        )

    @staticmethod
    def _append(path: Path, record: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def caption_case(
        self,
        case_json: str | Path,
        prompt: str,
        *,
        fps: float = 15.0,
        max_frames: int = 64,
        max_pixels: int = 6_500_000,
        max_new_tokens: int = 256,
    ) -> dict[str, Any]:
        case_path = Path(case_json)
        case = json.loads(case_path.read_text(encoding="utf-8"))
        video_path = Path(case["source_video"])
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        started = time.time()
        record: dict[str, Any] = {
            "case_number": None,
            "case_id": case_path.stem,
            "case_json": str(case_path),
            "video_key": "source_video",
            "video": str(video_path),
            "source_video": str(video_path),
            "input_caption": case.get("input_caption"),
            "model": self.model_path,
            "physical_gpu": self.physical_gpu,
            "runtime": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "kernels": kernels.__version__,
                "qwen_fast_path": qwen35.is_fast_path_available,
                "causal_conv_available": qwen35.causal_conv1d_fn is not None,
                "fla_available": qwen35.chunk_gated_delta_rule is not None,
            },
            "prompt": prompt,
            "video_request": {
                "fps": fps,
                "max_frames": max_frames,
                "max_pixels": max_pixels,
            },
            "sampling": {
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_new_tokens": max_new_tokens,
            },
            "loading": self.loading,
        }
        try:
            messages = [{"role": "user", "content": [
                {"type": "video", "video": str(video_path), "fps": fps,
                 "max_frames": max_frames, "max_pixels": max_pixels},
                {"type": "text", "text": prompt},
            ]}]
            chat_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            _, video_inputs, video_kwargs = process_vision_info(
                messages, image_patch_size=16, return_video_kwargs=True,
                return_video_metadata=True,
            )
            raw_video, metadata = video_inputs[0]
            inputs = self.processor(
                text=[chat_text], videos=[raw_video], video_metadata=[metadata],
                padding=True, return_tensors="pt", **video_kwargs,
            )
            inputs = {
                key: value.to("cuda:0") if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
            record["video_metadata"] = str(metadata)
            record["input_shapes"] = {
                key: list(value.shape) for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.0,
                use_cache=True,
            )
            generated_tokens = generated[:, inputs["input_ids"].shape[1]:]
            record["new_token_ids"] = generated_tokens[0].detach().cpu().tolist()
            record["raw_output"] = self.processor.batch_decode(
                generated_tokens, skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            record["caption"] = self.processor.batch_decode(
                generated_tokens, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            record["status"] = "ok"
        except Exception as error:
            record["status"] = "error"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
        finally:
            record["elapsed_seconds"] = round(time.time() - started, 3)
            torch.cuda.empty_cache()
            gc.collect()
        return record

    def run_cases(
        self,
        case_list: str | Path | Iterable[str | Path],
        output: str | Path,
        prompt_file: str | Path,
        *,
        resume: bool = True,
        **generation_kwargs: Any,
    ) -> list[dict[str, Any]]:
        if isinstance(case_list, (str, Path)):
            cases = [Path(line) for line in Path(case_list).read_text().splitlines() if line.strip()]
        else:
            cases = [Path(item) for item in case_list]
        prompt = Path(prompt_file).read_text(encoding="utf-8").strip()
        output_path = Path(output)
        done: dict[str, dict[str, Any]] = {}
        if resume and output_path.is_file():
            for line in output_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    done[row["case_id"]] = row
        results = list(done.values())
        for number, case_path in enumerate(cases, start=1):
            case_id = case_path.stem
            if case_id in done and done[case_id].get("status") == "ok":
                print(f"skip={number}/{len(cases)} case={case_id}", flush=True)
                continue
            row = self.caption_case(case_path, prompt, **generation_kwargs)
            row["case_number"] = number
            self._append(output_path, row)
            done[case_id] = row
            results.append(row)
            print(
                f"case={number}/{len(cases)} case_id={case_id} "
                f"status={row['status']} elapsed_seconds={row['elapsed_seconds']}",
                flush=True,
            )
        return results
