from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_STATE_ADAPTER_KEYS_TI2V = {
    "state_adapter_config",
    "state_adapter",
    "model_state_adapter",
}
REQUIRED_STATE_ADAPTER_KEYS_I2V = {
    "state_adapter_config",
    "state_adapter",
    "low_noise_model_state_adapter",
    "high_noise_model_state_adapter",
}


@dataclass(slots=True)
class StateConditionBundleRecord:
    sample_id: str
    bundle_dir: Path
    episode_path: Path
    image_path: Path
    state_condition_path: Path
    meta_path: Path
    prompt_path: Path
    prompt: str
    meta: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_state_condition_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def load_episode_npz(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def is_ti2v_state_adapter_checkpoint(state_bundle: dict[str, object]) -> bool:
    return REQUIRED_STATE_ADAPTER_KEYS_TI2V.issubset(state_bundle.keys())


def is_i2v_state_adapter_checkpoint(state_bundle: dict[str, object]) -> bool:
    return REQUIRED_STATE_ADAPTER_KEYS_I2V.issubset(state_bundle.keys())


def _discover_bundle_dirs(root: Path) -> list[Path]:
    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_file():
        bundle_dirs: list[Path] = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                state_condition_path = Path(record["state_condition_path"]).resolve()
                bundle_dirs.append(state_condition_path.parent)
        return bundle_dirs
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "state_condition.npz").is_file() and (path / "meta.json").is_file()
    )


def discover_state_condition_bundles(root: str | Path, limit: int = 0) -> list[StateConditionBundleRecord]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"state-condition root does not exist: {root}")
    bundle_dirs = _discover_bundle_dirs(root)
    if limit > 0:
        bundle_dirs = bundle_dirs[:limit]
    records: list[StateConditionBundleRecord] = []
    for bundle_dir in bundle_dirs:
        meta_path = bundle_dir / "meta.json"
        state_condition_path = bundle_dir / "state_condition.npz"
        image_path = bundle_dir / "input_image.png"
        prompt_path = bundle_dir / "prompt.txt"
        meta = _read_json(meta_path)
        episode_path = Path(str(meta["episode_path"])).resolve()
        prompt = prompt_path.read_text(encoding="utf-8").strip() if prompt_path.is_file() else str(meta.get("prompt", ""))
        records.append(
            StateConditionBundleRecord(
                sample_id=str(meta.get("sample_id", bundle_dir.name)),
                bundle_dir=bundle_dir.resolve(),
                episode_path=episode_path,
                image_path=image_path.resolve(),
                state_condition_path=state_condition_path.resolve(),
                meta_path=meta_path.resolve(),
                prompt_path=prompt_path.resolve(),
                prompt=prompt,
                meta=meta,
            )
        )
    if not records:
        raise FileNotFoundError(f"no state-condition bundles found under {root}")
    return records
