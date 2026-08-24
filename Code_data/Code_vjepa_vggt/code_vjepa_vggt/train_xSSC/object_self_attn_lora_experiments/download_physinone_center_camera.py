#!/usr/bin/env python3
"""Selective PhysInOne downloader: one per-case center static camera, RGB only.

PhysInOne publishes one ZIP per case.  This reader uses HTTP Range requests,
so it fetches the ZIP directory, the small trajectory/camera metadata, and the
RGB members of the selected camera without downloading the other streams.
The selected camera is the static camera whose optical axis is closest to
horizontal (front-facing rather than top-down), with camera-cluster distance
as a small center-view tie-breaker.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import traceback
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests


@dataclass(frozen=True)
class Entry:
    name: str
    method: int
    csize: int
    usize: int
    offset: int


class RangeZip:
    def __init__(self, url: str, token: str | None):
        self.base_url = url
        self.url: str | None = None
        self.session = requests.Session()
        self.session.headers["Accept-Encoding"] = "identity"
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"
        self.entries: dict[str, Entry] = {}

    def request(self, headers: dict[str, str]) -> requests.Response:
        response = self.session.get(
            self.url or self.base_url,
            headers=headers,
            allow_redirects=True,
            timeout=(30, 180),
        )
        response.raise_for_status()
        if self.url is None:
            self.url = response.url
        return response

    @staticmethod
    def content_range(response: requests.Response) -> tuple[int, int, int]:
        value = response.headers.get("Content-Range", "")
        try:
            unit, numbers = value.split(" ", 1)
            interval, total = numbers.split("/", 1)
            start, end = interval.split("-", 1)
            if unit != "bytes":
                raise ValueError
            return int(start), int(end), int(total)
        except (ValueError, AttributeError) as exc:
            raise RuntimeError(f"Invalid Content-Range: {value!r}") from exc

    def get_range(self, start: int, end: int) -> bytes:
        response = self.request({"Range": f"bytes={start}-{end}"})
        if response.status_code != 206:
            raise RuntimeError(f"Server ignored Range {start}-{end}")
        actual_start, actual_end, _ = self.content_range(response)
        data = response.content
        if actual_start != start or actual_end != end or len(data) != end - start + 1:
            raise RuntimeError(
                f"Bad range response: requested {start}-{end}, "
                f"returned {actual_start}-{actual_end}, bytes={len(data)}"
            )
        return data

    def load_directory(self) -> None:
        response = self.request({"Range": "bytes=-4194304"})
        if response.status_code != 206:
            raise RuntimeError("Server ignored the ZIP tail Range request")
        tail_start, tail_end, total = self.content_range(response)
        tail = response.content
        if tail_end - tail_start + 1 != len(tail):
            raise RuntimeError("Invalid ZIP tail response")
        eocd_pos = tail.rfind(b"PK\x05\x06")
        if eocd_pos < 0:
            raise RuntimeError("ZIP end-of-central-directory record not found")
        values = struct.unpack_from("<4s4H2LH", tail, eocd_pos)
        _, _, _, _, count, central_size, central_offset, _ = values
        if count == 0xFFFF or central_size == 0xFFFFFFFF or central_offset == 0xFFFFFFFF:
            raise RuntimeError("ZIP64 archive is not supported by the range reader")
        central = self.get_range(central_offset, central_offset + central_size - 1)
        pos = 0
        parsed = 0
        while pos + 46 <= len(central) and central[pos : pos + 4] == b"PK\x01\x02":
            values = struct.unpack_from("<4s6H3L5H2L", central, pos)
            (
                _, _, _, _, method, _, _, _, csize, usize,
                name_size, extra_size, comment_size, _, _, _, offset,
            ) = values
            name_start = pos + 46
            name = central[name_start : name_start + name_size].decode("utf-8")
            extra_start = name_start + name_size
            extra = central[extra_start : extra_start + extra_size]
            csize, usize, offset = self.zip64_values(csize, usize, offset, extra)
            self.entries[name] = Entry(name, method, csize, usize, offset)
            pos += 46 + name_size + extra_size + comment_size
            parsed += 1
        if parsed != count:
            raise RuntimeError(f"Parsed {parsed} entries, expected {count}")

    @staticmethod
    def zip64_values(csize: int, usize: int, offset: int, extra: bytes) -> tuple[int, int, int]:
        if csize != 0xFFFFFFFF and usize != 0xFFFFFFFF and offset != 0xFFFFFFFF:
            return csize, usize, offset
        pos = 0
        while pos + 4 <= len(extra):
            tag, size = struct.unpack_from("<HH", extra, pos)
            body = extra[pos + 4 : pos + 4 + size]
            pos += 4 + size
            if tag != 1:
                continue
            body_pos = 0
            if usize == 0xFFFFFFFF:
                usize = struct.unpack_from("<Q", body, body_pos)[0]
                body_pos += 8
            if csize == 0xFFFFFFFF:
                csize = struct.unpack_from("<Q", body, body_pos)[0]
                body_pos += 8
            if offset == 0xFFFFFFFF:
                offset = struct.unpack_from("<Q", body, body_pos)[0]
            break
        return csize, usize, offset

    @staticmethod
    def data_start(entry: Entry, header: bytes) -> int:
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid local header: {entry.name}")
        values = struct.unpack_from("<4s5H3L2H", header, 0)
        name_size, extra_size = values[9], values[10]
        return entry.offset + 30 + name_size + extra_size

    @staticmethod
    def decompress(entry: Entry, data: bytes) -> bytes:
        if entry.method == 0:
            output = data
        elif entry.method == 8:
            output = zlib.decompress(data, -15)
        else:
            raise RuntimeError(f"Unsupported ZIP method {entry.method}: {entry.name}")
        if len(output) != entry.usize:
            raise RuntimeError(f"Size mismatch for {entry.name}")
        return output

    def read(self, entry: Entry) -> bytes:
        header = self.get_range(entry.offset, entry.offset + 4095)
        start = self.data_start(entry, header)
        return self.decompress(entry, self.get_range(start, start + entry.csize - 1))

    def read_contiguous(self, entries: list[Entry]) -> dict[str, bytes]:
        first = min(entries, key=lambda item: item.offset)
        last = max(entries, key=lambda item: item.offset)
        first_header = self.get_range(first.offset, first.offset + 4095)
        last_header = self.get_range(last.offset, last.offset + 4095)
        span_start = first.offset
        last_start = self.data_start(last, last_header)
        span_end = last_start + last.csize - 1
        payload = self.get_range(span_start, span_end)
        output = {}
        for entry in entries:
            local_header = payload[entry.offset - span_start : entry.offset - span_start + 4096]
            start = self.data_start(entry, local_header) - span_start
            compressed = payload[start : start + entry.csize]
            output[entry.name] = self.decompress(entry, compressed)
        # Keep this read so the first header is validated even if the archive
        # has an unusual local-header layout.
        self.data_start(first, first_header)
        return output


def camera_position(camera: dict) -> tuple[float, float, float] | None:
    frames = camera.get("frames") or []
    matrix = frames[0].get("transform_matrix") if frames else None
    if not matrix:
        return None
    return tuple(float(matrix[i][3]) for i in range(3))


def camera_forward_z(camera: dict) -> float | None:
    frames = camera.get("frames") or []
    matrix = frames[0].get("transform_matrix") if frames else None
    if not matrix:
        return None
    # Blender camera looks along local -Z; the C2W matrix gives that axis in
    # world coordinates as the negated third column.
    return -float(matrix[2][2])


def select_camera(
    cameras: dict[str, dict],
    static_names: list[str],
) -> tuple[str, float, float]:
    candidates = []
    for name in static_names:
        if name == "CineCamera_Moving" or name not in cameras:
            continue
        position = camera_position(cameras[name])
        forward_z = camera_forward_z(cameras[name])
        if position is not None and forward_z is not None:
            candidates.append((name, position, forward_z))
    if not candidates:
        raise RuntimeError("No static camera metadata with a valid pose")
    center = tuple(
        sum(position[index] for _, position, _ in candidates) / len(candidates)
        for index in range(3)
    )
    scored = []
    for name, position, forward_z in candidates:
        horizontal_offset = math.sqrt(
            sum((position[i] - center[i]) ** 2 for i in range(2))
        )
        # Smaller |forward_z| means a more level/front-facing view.  The
        # second term keeps the choice near the center of the camera layout.
        score = abs(forward_z) + 0.01 * horizontal_offset
        scored.append((score, name, abs(forward_z), horizontal_offset))
    best = min(scored)
    return best[1], best[2], best[3]


def inspect_case(reader: RangeZip) -> dict:
    static_name = next(
        (name for name in reader.entries if name.endswith("/static_camera_list.txt")),
        None,
    )
    if static_name is None:
        raise RuntimeError("static_camera_list.txt not found")
    root = static_name.rsplit("/", 1)[0]
    trajectory_name = next(
        (
            name for name in reader.entries
            if name.startswith(root + "/")
            and name.endswith("_trajectory.json")
            and "/blender_CineCamera_" not in name
        ),
        None,
    )
    if trajectory_name is None:
        raise RuntimeError("trajectory JSON not found")
    caption_name = root + "/caption.txt"
    static_names = [
        line.strip()
        for line in reader.read(reader.entries[static_name]).decode().splitlines()
        if line.strip() and line.strip() != "CineCamera_Moving"
    ]
    camera_entries = {}
    cameras = {}
    for name in static_names:
        member = root + "/blender_" + name + ".json"
        if member in reader.entries:
            camera_entries[name] = reader.entries[member]
            cameras[name] = json.loads(reader.read(reader.entries[member]).decode())
    trajectory = json.loads(reader.read(reader.entries[trajectory_name]).decode())
    center_name, front_tilt, horizontal_offset = select_camera(cameras, static_names)
    rgb_prefix = root + "/" + center_name + "/rgb/"
    rgb_entries = sorted(
        (entry for name, entry in reader.entries.items() if name.startswith(rgb_prefix)),
        key=lambda entry: entry.name,
    )
    if not rgb_entries:
        raise RuntimeError(f"No RGB frames for {center_name}")
    return {
        "root": root,
        "trajectory_name": trajectory_name,
        "caption_name": caption_name if caption_name in reader.entries else None,
        "trajectory": trajectory,
        "center_name": center_name,
        "front_tilt": front_tilt,
        "horizontal_offset": horizontal_offset,
        "actor_scale": None,
        "camera_entry": camera_entries[center_name],
        "rgb_entries": rgb_entries,
    }


def build_url(endpoint: str, repo_id: str, filename: str) -> str:
    return (
        endpoint.rstrip("/")
        + f"/datasets/{repo_id}/resolve/main/"
        + quote(filename, safe="/")
    )


def process_case(case: dict, repo_map: dict, output_dir: Path, endpoint: str, token: str | None) -> str:
    repo_id = repo_map[case["part_id"]]
    output_case = output_dir / case["split"] / case["activity_type"] / case["case_id"]
    marker = output_case / ".complete.json"
    if marker.exists():
        return "skipped"
    if output_case.exists():
        raise RuntimeError(f"Incomplete output already exists: {output_case}")

    reader = RangeZip(build_url(endpoint, repo_id, case["hf_zip_path"]), token)
    reader.load_directory()
    info = inspect_case(reader)
    temp_case = output_case.with_name(output_case.name + ".tmp-" + uuid.uuid4().hex)
    root_out = temp_case / Path(info["root"]).name
    root_out.mkdir(parents=True, exist_ok=False)
    try:
        trajectory_name = Path(info["trajectory_name"]).name
        (root_out / trajectory_name).write_text(
            json.dumps(info["trajectory"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if info["caption_name"]:
            (root_out / "caption.txt").write_bytes(reader.read(reader.entries[info["caption_name"]]))
        selected = info["center_name"]
        (root_out / ("blender_" + selected + ".json")).write_bytes(reader.read(info["camera_entry"]))
        (root_out / "static_camera_list.txt").write_text(selected + "\n", encoding="utf-8")

        rgb = reader.read_contiguous(info["rgb_entries"])
        rgb_out = root_out / selected / "rgb"
        rgb_out.mkdir(parents=True, exist_ok=False)
        for entry in info["rgb_entries"]:
            (rgb_out / Path(entry.name).name).write_bytes(rgb[entry.name])

        metadata = {
            "case": case,
            "source_repo": repo_id,
            "source_zip": case["hf_zip_path"],
            "selected_camera": selected,
            "selection_rule": "front-center camera: smallest optical-axis vertical component, center-offset tie-break",
            "actor_coordinate_scale": None,
            "front_tilt": info["front_tilt"],
            "horizontal_offset": info["horizontal_offset"],
            "rgb_frame_count": len(info["rgb_entries"]),
        }
        (root_out / "center_camera_selection.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp_case.mkdir(parents=True, exist_ok=True)
        (temp_case / ".complete.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        output_case.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp_case, output_case)
        return "downloaded"
    except Exception:
        shutil.rmtree(temp_case, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--repo-map", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    repo_map = json.loads(args.repo_map.read_text(encoding="utf-8"))
    cases = selection.get("cases") or []
    if not cases:
        raise SystemExit("Selection contains no cases")
    missing = sorted({case["part_id"] for case in cases} - set(repo_map))
    if missing:
        raise SystemExit("Missing repo map entries: " + ", ".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or None
    failures = args.output_dir / "failures.jsonl"
    done = downloaded = skipped = failed = 0
    print(f"Selective center-camera download: {len(cases)} cases, workers={args.workers}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_cases = {
            pool.submit(process_case, case, repo_map, args.output_dir, args.endpoint, token): case
            for case in cases
        }
        with failures.open("a", encoding="utf-8") as failures_file:
            for future in as_completed(future_cases):
                case = future_cases[future]
                done += 1
                try:
                    status = future.result()
                    if status == "downloaded":
                        downloaded += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    failed += 1
                    failures_file.write(json.dumps({
                        "case": case,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }, ensure_ascii=False) + "\n")
                    failures_file.flush()
                    print(f"FAILED {case['case_id']}: {exc}", flush=True)
                if done % 10 == 0 or done == len(cases):
                    print(
                        f"Progress {done}/{len(cases)}: downloaded={downloaded}, "
                        f"skipped={skipped}, failed={failed}",
                        flush=True,
                    )
    print(
        f"Finished: downloaded={downloaded}, skipped={skipped}, failed={failed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
