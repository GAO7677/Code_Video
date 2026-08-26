#!/usr/bin/env python3
"""Single entry point for the Physics-IQ-Verified strict P0 protocol.

Model inference stays in a small adapter.  This program owns the parts that
must not drift between models: the official 198-case input set, 72@24 V2V
conditioning, 189@24 raw output, 69-frame prefix removal, 120@24 submission
videos, prompt digest, output naming, validation, and official scoring.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Optional, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_WORKSPACE = Path(
    os.environ.get(
        "PHYSIQ_WORKSPACE",
        "/data/gaoya/AAA_test_video/0623/test/physicsiq/physicsiq_verified",
    )
)
DEFAULT_INPUT_LIST = DEFAULT_WORKSPACE / "inputs/bpp/verified_v2v_bpp_198.txt"
DEFAULT_DESCRIPTIONS = (
    ROOT.parent / "physics-IQ-benchmark-main" / "descriptions" / "best_practice" / "descriptions_base.csv"
)
DEFAULT_PROMPT_CONFIG = ROOT / "common" / "physicsiq_p0_prompt.env"
DEFAULT_FFPROBE = Path(
    os.environ.get("PHYSIQ_FFPROBE", "/home/gaoya/miniconda3/envs/wan-cu128/bin/ffprobe")
)
DEFAULT_OFFICIAL_RUNNER = ROOT / "run_verified_official.sh"
DEFAULT_AGGREGATOR = ROOT / "aggregate_verified_official.sh"

EXPECTED_INPUT_LIST_SHA256 = (
    "f0cbcd79cc7d523fd0c30ef6053373163dbc3667da88baa5d10e205def177956"
)
EXPECTED_DESCRIPTIONS_SHA256 = (
    "20ffd208acc0b0f50d4638d1da69218168e78336e96118244a53d0ae046729c8"
)


@dataclass(frozen=True)
class P0Contract:
    protocol: str = "Physics-IQ-Verified-P0"
    prompt_setting: str = "bpp"
    input_mode: str = "v2v"
    cases: int = 198
    condition_frames: int = 72
    condition_fps: int = 24
    condition_seconds: float = 3.0
    height: int = 512
    width: int = 896
    raw_frames: int = 189
    prefix_frames: int = 69
    submission_frames: int = 120
    fps: int = 24
    steps: int = 40
    guidance: float = 5.0
    seed: int = 42
    context_mask_mode: str = "dynamic_effective"
    do_cfg: bool = False


CONTRACT = P0Contract()


class ProtocolError(RuntimeError):
    """A user-actionable protocol or output validation error."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def resolve_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ProtocolError(f"{label} not found: {path}")
    return path


def resolve_dir(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ProtocolError(f"{label} not found: {path}")
    return path


def parse_prompt_env(path: Path) -> dict[str, str]:
    path = resolve_file(path, "prompt config")
    values: dict[str, str] = {}
    wanted = {
        "PHYSIQ_P0_NEGATIVE_PROMPT",
        "PHYSIQ_P0_NEGATIVE_PROMPT_VERSION",
        "PHYSIQ_P0_NEGATIVE_PROMPT_SHA256",
    }
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if key not in wanted:
            continue
        try:
            parsed = shlex.split(raw_value.strip(), comments=False, posix=True)
        except ValueError as exc:
            raise ProtocolError(f"cannot parse {key} in {path}: {exc}") from exc
        values[key] = parsed[0] if parsed else ""
    missing = sorted(wanted - values.keys())
    if missing:
        raise ProtocolError(f"prompt config is missing: {', '.join(missing)}")
    actual_hash = sha256_bytes(values["PHYSIQ_P0_NEGATIVE_PROMPT"].encode("utf-8"))
    if actual_hash != values["PHYSIQ_P0_NEGATIVE_PROMPT_SHA256"]:
        raise ProtocolError(
            "canonical negative-prompt SHA256 mismatch: "
            f"declared={values['PHYSIQ_P0_NEGATIVE_PROMPT_SHA256']} actual={actual_hash}"
        )
    return values


def official_names(descriptions_file: Path) -> list[str]:
    descriptions_file = resolve_file(descriptions_file, "descriptions CSV")
    with descriptions_file.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if "_take-1_" in row["scenario"]]
    rows.sort(key=lambda row: int(row["scenario"].split("_", 1)[0]))
    names = [row["generated_video_name"] for row in rows]
    if len(names) != CONTRACT.cases or len(set(names)) != CONTRACT.cases:
        raise ProtocolError(
            f"official descriptions must define {CONTRACT.cases} unique take-1 names; got {len(names)}"
        )
    for index, name in enumerate(names, start=1):
        if not name.startswith(f"{index:04d}_"):
            raise ProtocolError(f"non-contiguous official generated name: {name}")
    return names


def parse_rate(value: str) -> float:
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)


def probe_video(ffprobe: Path, path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            subprocess.check_output(
                [
                    str(ffprobe),
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
                stderr=subprocess.STDOUT,
            )
        )
        stream = payload["streams"][0]
        return {
            "name": path.name,
            "frames": int(stream["nb_read_frames"]),
            "fps": parse_rate(stream["avg_frame_rate"]),
            "duration": float(payload["format"]["duration"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
        }
    except (KeyError, IndexError, ValueError, subprocess.SubprocessError) as exc:
        raise ProtocolError(f"ffprobe failed for {path}: {exc}") from exc


def case_path_from_line(input_list: Path, value: str) -> Path:
    declared = Path(value)
    if declared.is_file():
        return declared.resolve()
    candidate = input_list.parent / "jsons" / declared.name
    if candidate.is_file():
        return candidate.resolve()
    raise ProtocolError(f"case JSON not found: {value}")


def source_path_from_case(case_path: Path, payload: dict[str, Any]) -> Path:
    raw = payload.get("source_video") or payload.get("input_video")
    if not isinstance(raw, str):
        raise ProtocolError(f"case has no source_video/input_video: {case_path}")
    declared = Path(raw)
    if declared.is_file():
        return declared.resolve()
    candidate = case_path.parent.parent / "conditioning" / "24FPS" / declared.name
    if candidate.is_file():
        return candidate.resolve()
    raise ProtocolError(f"conditioning video not found for {case_path}: {raw}")


def read_cases(
    input_list: Path,
    names: Sequence[str],
    ffprobe: Path,
    workers: int,
    allow_noncanonical: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_list = resolve_file(input_list, "P0 input list")
    lines = [line.strip() for line in input_list.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != CONTRACT.cases:
        raise ProtocolError(f"P0 input list must contain {CONTRACT.cases} cases; got {len(lines)}")
    input_hash = sha256_file(input_list)
    if not allow_noncanonical and input_hash != EXPECTED_INPUT_LIST_SHA256:
        raise ProtocolError(
            f"input-list SHA256 mismatch: expected {EXPECTED_INPUT_LIST_SHA256}, got {input_hash}"
        )

    cases: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        case_path = case_path_from_line(input_list, line)
        try:
            payload = json.loads(case_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"cannot read case JSON {case_path}: {exc}") from exc
        expected_name = names[index]
        checks = {
            "prompt_setting": CONTRACT.prompt_setting,
            "input_mode": CONTRACT.input_mode,
            "conditioning_frames": CONTRACT.condition_frames,
            "conditioning_fps": CONTRACT.condition_fps,
        }
        for key, expected in checks.items():
            if payload.get(key) != expected:
                raise ProtocolError(
                    f"{case_path}: {key}={payload.get(key)!r}, expected {expected!r}"
                )
        if payload.get("generated_video_name") != expected_name:
            raise ProtocolError(
                f"{case_path}: generated_video_name={payload.get('generated_video_name')!r}, "
                f"expected {expected_name!r}"
            )
        if not isinstance(payload.get("input_caption"), str) or not payload["input_caption"].strip():
            raise ProtocolError(f"{case_path}: input_caption is missing or empty")
        source = source_path_from_case(case_path, payload)
        duration = float(payload.get("conditioning_duration_seconds", CONTRACT.condition_seconds))
        if abs(duration - CONTRACT.condition_seconds) > 1e-6:
            raise ProtocolError(f"{case_path}: condition duration is {duration}, expected 3.0")
        cases.append(
            {
                "index": index + 1,
                "case_json": str(case_path),
                "source_video": str(source),
                "generated_video_name": expected_name,
                "input_caption": str(payload["input_caption"]),
            }
        )

    def check_condition(case: dict[str, Any]) -> dict[str, Any]:
        info = probe_video(ffprobe, Path(case["source_video"]))
        if info["frames"] != CONTRACT.condition_frames:
            raise ProtocolError(
                f"{case['source_video']}: condition has {info['frames']} frames, "
                f"expected {CONTRACT.condition_frames}"
            )
        if abs(float(info["fps"]) - CONTRACT.condition_fps) > 1e-6:
            raise ProtocolError(
                f"{case['source_video']}: condition has {info['fps']} FPS, expected {CONTRACT.condition_fps}"
            )
        if abs(float(info["duration"]) - CONTRACT.condition_seconds) > 0.001:
            raise ProtocolError(
                f"{case['source_video']}: condition duration is {info['duration']}, expected 3.0"
            )
        return info

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        condition_info = list(executor.map(check_condition, cases))
    return cases, {
        "input_list": str(input_list),
        "input_list_sha256": input_hash,
        "case_count": len(cases),
        "condition_probe": {
            "frames": CONTRACT.condition_frames,
            "fps": CONTRACT.condition_fps,
            "seconds": CONTRACT.condition_seconds,
            "resolutions": sorted(
                {f"{item['width']}x{item['height']}" for item in condition_info}
            ),
        },
    }


def validate_video_set(
    folder: Path,
    names: Sequence[str],
    ffprobe: Path,
    workers: int,
    raw: bool = False,
) -> dict[str, Any]:
    folder = resolve_dir(folder, "video folder")
    expected = set(names)
    files = [path for path in folder.iterdir() if path.is_file()]
    mp4_names = {path.name for path in files if path.suffix.lower() == ".mp4"}
    if not raw:
        non_mp4 = sorted(path.name for path in files if path.suffix.lower() != ".mp4")
        if non_mp4:
            raise ProtocolError(f"submission must be MP4-only; unexpected files: {non_mp4[:8]}")
    missing = sorted(expected - mp4_names)
    extra = sorted(mp4_names - expected)
    if missing or extra:
        raise ProtocolError(
            f"{folder}: video set mismatch; missing={missing[:8]}, extra={extra[:8]}"
        )

    paths = [folder / name for name in names]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        metadata = list(executor.map(lambda path: probe_video(ffprobe, path), paths))

    expected_frames = CONTRACT.raw_frames if raw else CONTRACT.submission_frames
    invalid = []
    for item in metadata:
        if item["frames"] != expected_frames:
            invalid.append((item["name"], "frames", item["frames"], expected_frames))
        if abs(float(item["fps"]) - CONTRACT.fps) > 1e-6:
            invalid.append((item["name"], "fps", item["fps"], CONTRACT.fps))
        if item["width"] != CONTRACT.width or item["height"] != CONTRACT.height:
            invalid.append(
                (
                    item["name"],
                    "resolution",
                    f"{item['width']}x{item['height']}",
                    f"{CONTRACT.width}x{CONTRACT.height}",
                )
            )
        if not raw and abs(float(item["duration"]) - 5.0) > 0.001:
            invalid.append((item["name"], "duration", item["duration"], 5.0))
    if invalid:
        raise ProtocolError(f"invalid {'raw' if raw else 'submission'} videos: {invalid[:8]}")
    return {
        "folder": str(folder),
        "video_count": len(metadata),
        "frames": expected_frames,
        "fps": CONTRACT.fps,
        "resolution": f"{CONTRACT.width}x{CONTRACT.height}",
        "duration_seconds": None if raw else 5.0,
    }


def validate_encoding_manifest(path: Path) -> dict[str, Any]:
    path = resolve_file(path, "adapter encoding manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read adapter encoding manifest {path}: {exc}") from exc
    encoding = payload.get("encoding")
    if not isinstance(encoding, dict):
        raise ProtocolError(f"adapter encoding manifest has no encoding object: {path}")
    expected = {
        "mode": "in_memory_slice_then_export_to_video",
        "macro_block_size": 1,
        "intermediate_decode": False,
    }
    for key, value in expected.items():
        if encoding.get(key) != value:
            raise ProtocolError(
                f"adapter encoding mismatch in {path}: {key}={encoding.get(key)!r}, expected {value!r}"
            )
    expected_slice = f"raw[{CONTRACT.prefix_frames}:{CONTRACT.raw_frames}]"
    if encoding.get("slice") not in {expected_slice, "raw[69:]"}:
        raise ProtocolError(
            f"adapter encoding mismatch in {path}: slice={encoding.get('slice')!r}, "
            f"expected {expected_slice!r}"
        )
    return {"manifest": str(path), "encoding": encoding}


def contract_dict(prompt: dict[str, str]) -> dict[str, Any]:
    value = asdict(CONTRACT)
    value["negative_prompt_version"] = prompt["PHYSIQ_P0_NEGATIVE_PROMPT_VERSION"]
    value["negative_prompt_sha256"] = prompt["PHYSIQ_P0_NEGATIVE_PROMPT_SHA256"]
    value["encoding"] = {
        "raw_frames": CONTRACT.raw_frames,
        "prefix_frames_removed": CONTRACT.prefix_frames,
        "submission_frames": CONTRACT.submission_frames,
        "mode": "adapter-defined; must preserve the declared frame sequence",
    }
    return value


def common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-list", type=Path, default=DEFAULT_INPUT_LIST)
    parser.add_argument("--descriptions-file", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument("--prompt-config", type=Path, default=DEFAULT_PROMPT_CONFIG)
    parser.add_argument("--ffprobe", type=Path, default=DEFAULT_FFPROBE)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Allow a non-canonical input list/description/prompt; output is not strict P0.",
    )


def load_protocol_inputs(args: argparse.Namespace) -> tuple[list[str], dict[str, str], list[dict[str, Any]], dict[str, Any]]:
    descriptions = resolve_file(args.descriptions_file, "descriptions CSV")
    descriptions_hash = sha256_file(descriptions)
    if not args.allow_noncanonical and descriptions_hash != EXPECTED_DESCRIPTIONS_SHA256:
        raise ProtocolError(
            f"descriptions SHA256 mismatch: expected {EXPECTED_DESCRIPTIONS_SHA256}, got {descriptions_hash}"
        )
    names = official_names(descriptions)
    prompt = parse_prompt_env(args.prompt_config)
    if not args.allow_noncanonical:
        if prompt["PHYSIQ_P0_NEGATIVE_PROMPT_VERSION"] != "physrvg-72f-adapted-long-v1":
            raise ProtocolError("canonical P0 prompt version mismatch")
    cases, input_summary = read_cases(
        args.input_list,
        names,
        resolve_file(args.ffprobe, "ffprobe"),
        args.workers,
        args.allow_noncanonical,
    )
    input_summary["descriptions_file"] = str(descriptions)
    input_summary["descriptions_sha256"] = descriptions_hash
    input_summary["generated_names_sha256"] = sha256_bytes(("\n".join(names) + "\n").encode())
    input_summary["prompt_config"] = str(resolve_file(args.prompt_config, "prompt config"))
    return names, prompt, cases, input_summary


def print_json(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def preflight(args: argparse.Namespace) -> None:
    names, prompt, _cases, summary = load_protocol_inputs(args)
    result = {
        "status": "PASS",
        "protocol": contract_dict(prompt),
        "inputs": summary,
        "official_names_first": names[:3],
        "official_names_last": names[-3:],
    }
    if args.summary_json:
        output = args.summary_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(result)


def validate_command(args: argparse.Namespace) -> None:
    names, prompt, _cases, summary = load_protocol_inputs(args)
    submission = validate_video_set(
        args.run_folder,
        names,
        resolve_file(args.ffprobe, "ffprobe"),
        args.workers,
        raw=False,
    )
    result = {
        "status": "PASS",
        "protocol": contract_dict(prompt),
        "inputs": summary,
        "submission": submission,
    }
    if args.summary_json:
        output = args.summary_json.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_json(result)


def check_gpu(value: str) -> str:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens or any(not token.isdigit() for token in tokens):
        raise ProtocolError(f"GPU selection must be numeric, got {value!r}")
    if "4" in tokens:
        raise ProtocolError("GPU 4 is prohibited by the workspace rules")
    return ",".join(tokens)


def safe_name(value: str, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ProtocolError(f"{label} must contain only letters, digits, '.', '_' or '-': {value!r}")
    return value


def adapter_environment(
    args: argparse.Namespace,
    prompt: dict[str, str],
    run_name: str,
    raw_root: Path,
    submission_root: Path,
    result_file: Path,
    gpu: str,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "PHYSIQ_P0_PROTOCOL": CONTRACT.protocol,
            "PHYSIQ_WORKSPACE": str(args.workspace.expanduser().resolve()),
            "PHYSIQ_DATASET": str(args.dataset.expanduser().resolve()),
            "PHYSIQ_PROMPT_SETTING": CONTRACT.prompt_setting,
            "PHYSIQ_INPUT_MODE": CONTRACT.input_mode,
            "PHYSIQ_INPUT_LIST": str(args.input_list.expanduser().resolve()),
            "PHYSIQ_P0_INPUT_LIST": str(args.input_list.expanduser().resolve()),
            "PHYSIQ_DESCRIPTIONS_FILE": str(args.descriptions_file.expanduser().resolve()),
            "PHYSIQ_MODEL_NAME": args.model_name,
            "PHYSIQ_RUN_NAME": run_name,
            "PHYSIQ_RUN_INDEX": str(args.run_index),
            "PHYSIQ_RUN_TAG": f"run_{args.run_index:02d}",
            "PHYSIQ_SEED": str(CONTRACT.seed),
            "PHYSIQ_GPU_ID": gpu,
            "PHYSIQ_DEVICE": "cuda:0",
            "PHYSIQ_RAW_ROOT": str(raw_root),
            "PHYSIQ_SUBMISSION_ROOT": str(submission_root),
            "PHYSIQ_RESULT_FILE": str(result_file),
            "PHYSIQ_ENCODING_MANIFEST": str(submission_root.parent / f"{run_name}.manifest.json"),
            "PHYSIQ_P0_CASE_COUNT": str(CONTRACT.cases),
            "PHYSIQ_CONDITION_FRAMES": str(CONTRACT.condition_frames),
            "PHYSIQ_CONDITION_FPS": str(CONTRACT.condition_fps),
            "PHYSIQ_RAW_FRAMES": str(CONTRACT.raw_frames),
            "PHYSIQ_PREFIX_FRAMES": str(CONTRACT.prefix_frames),
            "PHYSIQ_SUBMISSION_FRAMES": str(CONTRACT.submission_frames),
            "PHYSIQ_FPS": str(CONTRACT.fps),
            "PHYSIQ_HEIGHT": str(CONTRACT.height),
            "PHYSIQ_WIDTH": str(CONTRACT.width),
            "PHYSIQ_NUM_INFERENCE_STEPS": str(CONTRACT.steps),
            "PHYSIQ_GUIDANCE_SCALE": str(CONTRACT.guidance),
            "PHYSIQ_CONTEXT_FRAMES": str(CONTRACT.condition_frames),
            "PHYSIQ_CONTEXT_MASK_MODE": CONTRACT.context_mask_mode,
            "PHYSIQ_DO_CFG": "0",
            "PHYSIQ_RESET_GLOBAL_SEED_PER_CASE": "1",
            "PHYSIQ_RNG_MODE": "global_seed_per_case",
            "PHYSIQ_NEGATIVE_PROMPT": prompt["PHYSIQ_P0_NEGATIVE_PROMPT"],
            "PHYSIQ_NEGATIVE_PROMPT_VERSION": prompt["PHYSIQ_P0_NEGATIVE_PROMPT_VERSION"],
            "PHYSIQ_NEGATIVE_PROMPT_SHA256": prompt["PHYSIQ_P0_NEGATIVE_PROMPT_SHA256"],
            "PHYSIQ_FORCE": "1" if args.force else "0",
            "CUDA_VISIBLE_DEVICES": gpu,
        }
    )
    return environment


def read_result_file(path: Path) -> Path:
    if not path.is_file():
        raise ProtocolError(
            f"adapter did not write {path}; it must publish the final folder via PHYSIQ_RESULT_FILE"
        )
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ProtocolError(f"PHYSIQ_RESULT_FILE must contain exactly one folder path: {path}")
    return Path(lines[0]).expanduser().resolve()


def generate_command(args: argparse.Namespace) -> None:
    names, prompt, _cases, input_summary = load_protocol_inputs(args)
    gpu = check_gpu(args.gpu)
    adapter = resolve_file(Path(args.adapter), "adapter")
    model_name = safe_name(args.model_name, "model name")
    run_name = safe_name(args.run_name or f"{model_name}-bpp-run_{args.run_index:02d}", "run name")
    workspace = args.workspace.expanduser().resolve()
    raw_root = workspace / "raw" / run_name
    submission_root = workspace / "generated_videos_5s" / run_name
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else workspace / "manifests" / f"{run_name}.p0_protocol.json"
    )
    result_cache = Path("/data/gaoya/agent-data/cache")
    result_cache.mkdir(parents=True, exist_ok=True)
    result_dir = Path(tempfile.mkdtemp(prefix="physicsiq-p0-result-", dir=result_cache))
    result_file = result_dir / "result.txt"
    environment = adapter_environment(
        args, prompt, run_name, raw_root, submission_root, result_file, gpu
    )
    command = [str(adapter), *args.adapter_arg]
    if adapter.suffix == ".sh":
        command.insert(0, "bash")
    print("P0 protocol: strict")
    print(f"adapter: {adapter}")
    print(f"model: {model_name}")
    print(f"run: {run_name}")
    print(f"gpu: {gpu}")
    print(f"input: {args.input_list}")
    print(f"submission target: {submission_root}")
    print("adapter command:", " ".join(shlex.quote(item) for item in command))
    if args.dry_run:
        shutil.rmtree(result_dir, ignore_errors=True)
        return

    raw_root.parent.mkdir(parents=True, exist_ok=True)
    submission_root.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, env=environment)
        if result_file.is_file():
            result_folder = read_result_file(result_file)
        elif submission_root.is_dir():
            result_folder = submission_root
        else:
            raise ProtocolError(
                "adapter completed without PHYSIQ_RESULT_FILE and without the expected submission folder"
            )
        if result_folder != submission_root and not args.allow_external_output:
            raise ProtocolError(
                f"adapter published {result_folder}, expected {submission_root}; "
                "use the shared output root or explicitly pass --allow-external-output for legacy runs"
            )
        submission = validate_video_set(
            result_folder,
            names,
            resolve_file(args.ffprobe, "ffprobe"),
            args.workers,
            raw=False,
        )
        raw_summary: Optional[dict[str, Any]] = None
        if args.require_raw:
            if not raw_root.is_dir():
                raise ProtocolError(f"strict P0 generation requires raw output folder: {raw_root}")
            raw_summary = validate_video_set(
                raw_root,
                names,
                resolve_file(args.ffprobe, "ffprobe"),
                args.workers,
                raw=True,
            )
        encoding_manifest_path = (
            args.encoding_manifest.expanduser().resolve()
            if args.encoding_manifest
            else submission_root.parent / f"{run_name}.manifest.json"
        )
        encoding_summary: Optional[dict[str, Any]] = None
        if args.require_direct_encoding:
            encoding_summary = validate_encoding_manifest(encoding_manifest_path)
        elif encoding_manifest_path.is_file():
            encoding_summary = validate_encoding_manifest(encoding_manifest_path)
        manifest = {
            "status": "PASS",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol": contract_dict(prompt),
            "model_name": model_name,
            "run_name": run_name,
            "adapter": str(adapter),
            "adapter_args": args.adapter_arg,
            "gpu": gpu,
            "inputs": input_summary,
            "raw": raw_summary,
            "encoding": encoding_summary,
            "submission": submission,
            "submission_folder": str(result_folder),
            "strict_p0": (
                not args.allow_noncanonical
                and not args.allow_external_output
                and args.require_raw
                and args.require_direct_encoding
            ),
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"submission=PASS {result_folder}")
        print(f"manifest={manifest_path}")
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)


def score_command(args: argparse.Namespace) -> None:
    if len(args.run_folders) > 4:
        raise ProtocolError("official aggregation accepts at most four runs")
    names, prompt, _cases, _summary = load_protocol_inputs(args)
    ffprobe = resolve_file(args.ffprobe, "ffprobe")
    folders = [resolve_dir(folder, "run folder") for folder in args.run_folders]
    for folder in folders:
        validate_video_set(folder, names, ffprobe, args.workers, raw=False)
    output = (
        args.output_folder.expanduser().resolve()
        if args.output_folder
        else args.workspace.expanduser().resolve() / "evaluation" / folders[0].name
    )
    output.mkdir(parents=True, exist_ok=True)
    command = [
        "bash",
        str(resolve_file(args.official_runner, "official runner")),
        "--output-folder",
        str(output),
        "--descriptions-file",
        str(resolve_file(args.descriptions_file, "descriptions CSV")),
        "--n-process",
        str(args.n_process),
        *[str(folder) for folder in folders],
    ]
    print("official command:", " ".join(shlex.quote(item) for item in command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    results_dir = output / "physics-IQ-benchmark-verified" / "results"
    csv_paths = []
    for folder in folders:
        csv_path = results_dir / f"{folder.name}.csv"
        if not csv_path.is_file():
            raise ProtocolError(f"official result CSV not found: {csv_path}")
        csv_paths.append(csv_path)
    summary_path = output / f"{folders[0].name.rsplit('-run_', 1)[0]}_verified_summary.csv"
    aggregate_command = [
        "bash",
        str(resolve_file(args.aggregator, "official aggregator")),
        *[str(path) for path in csv_paths],
        "--save-csv",
        str(summary_path),
        "--model-name",
        args.model_name or folders[0].name,
    ]
    print("aggregate command:", " ".join(shlex.quote(item) for item in aggregate_command))
    subprocess.run(aggregate_command, check=True)
    print(f"verified_summary={summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="validate the shared P0 inputs")
    common_options(preflight_parser)
    preflight_parser.add_argument("--summary-json", type=Path)
    preflight_parser.set_defaults(handler=preflight)

    validate_parser = subparsers.add_parser("validate", help="validate a final 198-video submission")
    common_options(validate_parser)
    validate_parser.add_argument("run_folder", type=Path)
    validate_parser.add_argument("--summary-json", type=Path)
    validate_parser.set_defaults(handler=validate_command)

    generate_parser = subparsers.add_parser("generate", help="run a model adapter under the P0 contract")
    common_options(generate_parser)
    generate_parser.add_argument("--adapter", required=True, help="executable model adapter")
    generate_parser.add_argument("--model-name", required=True)
    generate_parser.add_argument("--run-name")
    generate_parser.add_argument("--run-index", type=int, default=1, choices=range(1, 5))
    generate_parser.add_argument("--gpu", default="0", help="physical GPU id(s), excluding GPU 4")
    generate_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    generate_parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified"),
    )
    generate_parser.add_argument("--adapter-arg", action="append", default=[])
    generate_parser.add_argument("--manifest", type=Path)
    generate_parser.add_argument("--encoding-manifest", type=Path)
    generate_parser.add_argument("--force", action="store_true")
    generate_parser.add_argument("--require-raw", action=argparse.BooleanOptionalAction, default=True)
    generate_parser.add_argument(
        "--require-direct-encoding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the adapter manifest to declare in-memory slice + direct export (default: yes).",
    )
    generate_parser.add_argument("--allow-external-output", action="store_true")
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.set_defaults(handler=generate_command)

    score_parser = subparsers.add_parser("score", help="validate and run the official Verified scorer")
    common_options(score_parser)
    score_parser.add_argument("run_folders", nargs="+", type=Path)
    score_parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    score_parser.add_argument("--output-folder", type=Path)
    score_parser.add_argument("--n-process", type=int, default=0)
    score_parser.add_argument("--official-runner", type=Path, default=DEFAULT_OFFICIAL_RUNNER)
    score_parser.add_argument("--aggregator", type=Path, default=DEFAULT_AGGREGATOR)
    score_parser.add_argument("--model-name")
    score_parser.add_argument("--dry-run", action="store_true")
    score_parser.set_defaults(handler=score_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ProtocolError as exc:
        print(f"P0 protocol error: {exc}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}: {exc.cmd}", file=sys.stderr)
        return exc.returncode or 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
