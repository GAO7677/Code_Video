#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FRAMES = 13
HEIGHT = 16
WIDTH = 28


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--quantile", type=float, default=0.90)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--single-component", action="store_true")
    parser.add_argument("--removal-dilate-radius", type=int, default=0)
    parser.add_argument("--backtrack-frames", type=int, default=0)
    return parser.parse_args()


def scalar(data, key):
    return np.asarray(data[key]).item()


def read_frames(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 49:
        raise RuntimeError(f"Expected 49 frames, got {len(frames)}: {path}")
    return [frames[index] for index in range(0, 49, 4)]


def components(mask):
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return labels, [labels == index for index in range(1, count)]


def centroid(mask):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.asarray([HEIGHT / 2, WIDTH / 2], dtype=np.float32)
    return np.asarray([ys.mean(), xs.mean()], dtype=np.float32)


def choose_anchor(candidate, query_indices):
    labels, regions = components(candidate)
    if not regions:
        return candidate.copy()
    query_spatial = [int(index) % (HEIGHT * WIDTH) for index in query_indices]
    query_yx = [(index // WIDTH, index % WIDTH) for index in query_spatial]
    scores = [sum(bool(region[y, x]) for y, x in query_yx) for region in regions]
    if max(scores) > 0:
        return regions[int(np.argmax(scores))]
    target = np.asarray(query_yx, dtype=np.float32).mean(axis=0)
    distances = [float(np.linalg.norm(centroid(region) - target)) for region in regions]
    return regions[int(np.argmin(distances))]


def track(candidate, response, query_indices, radius, single_component=False):
    trajectory = np.zeros_like(candidate, dtype=bool)
    trajectory[0] = candidate[0]
    trajectory[1] = choose_anchor(candidate[1], query_indices)
    kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    for frame in range(2, FRAMES):
        _labels, regions = components(candidate[frame])
        if not regions:
            continue
        neighborhood = cv2.dilate(trajectory[frame - 1].astype(np.uint8), kernel) > 0
        continuous = [region for region in regions if np.any(region & neighborhood)]
        if continuous:
            if single_component:
                previous_center = centroid(trajectory[frame - 1])
                trajectory[frame] = max(
                    continuous,
                    key=lambda region: (
                        int(np.count_nonzero(region & neighborhood)),
                        -float(np.linalg.norm(centroid(region) - previous_center)),
                        float(response[frame][region].sum()),
                    ),
                )
            else:
                trajectory[frame] = np.logical_or.reduce(continuous)
            continue
        previous_center = centroid(trajectory[frame - 1])
        nearest = min(
            regions,
            key=lambda region: (
                float(np.linalg.norm(centroid(region) - previous_center)),
                -float(response[frame][region].sum()),
            ),
        )
        trajectory[frame] = nearest
    return trajectory


def dilate_mask(mask, radius=1):
    size = 2 * radius + 1
    kernel = np.ones((size, size), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def best_matching_component(mask, anchor, max_center_distance=2.5):
    _labels, regions = components(mask)
    if not regions:
        return None
    anchor_neighborhood = dilate_mask(anchor, radius=1)
    anchor_center = centroid(anchor)
    matches = []
    for region in regions:
        overlap = int(np.count_nonzero(region & anchor_neighborhood))
        distance = float(np.linalg.norm(centroid(region) - anchor_center))
        if overlap > 0 or distance <= max_center_distance:
            matches.append((overlap, -distance, region))
    if not matches:
        return None
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def backward_prune(candidate, forward_trajectory, backtrack_frames):
    corrected = forward_trajectory.copy()
    rejected_events = np.zeros_like(corrected, dtype=bool)
    backward_removed = np.zeros_like(corrected, dtype=bool)
    if backtrack_frames <= 0:
        return corrected, rejected_events, backward_removed

    for frame in range(2, FRAMES):
        previous = corrected[frame - 1]
        current = corrected[frame]
        if not previous.any() or not current.any():
            continue

        # Trigger only when the selected branch switches. A normally continuing
        # branch must not cause nearby rejected candidates to prune its history.
        if np.any(current & dilate_mask(previous, radius=1)):
            continue
        rejected_component = best_matching_component(
            candidate[frame] & ~current,
            previous,
        )
        if rejected_component is None:
            continue

        rejected_events[frame] |= rejected_component
        anchor = rejected_component
        earliest = max(2, frame - backtrack_frames)
        for history_frame in range(frame - 1, earliest - 1, -1):
            matched = best_matching_component(corrected[history_frame], anchor)
            if matched is None:
                break

            # A merged component may contain both the rejected branch and the
            # retained future branch. Stop rather than deleting both objects.
            retained_future = corrected[history_frame + 1]
            if retained_future.any() and np.any(
                matched & dilate_mask(retained_future, radius=1)
            ):
                break

            corrected[history_frame] &= ~matched
            backward_removed[history_frame] |= matched
            anchor = matched

    return corrected, rejected_events, backward_removed


def overlay(frame, values, binary=False, color=(30, 50, 235)):
    frame = cv2.resize(frame, (320, 183), interpolation=cv2.INTER_AREA)
    values = cv2.resize(values.astype(np.float32), (320, 183), interpolation=cv2.INTER_NEAREST)
    if binary:
        active = values > 0.5
        tint = np.empty_like(frame)
        tint[:] = color
        frame[active] = cv2.addWeighted(frame, 0.28, tint, 0.72, 0)[active]
        return frame
    vmax = max(float(values.max()), 1e-12)
    norm = np.clip(values / vmax, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    return cv2.addWeighted(frame, 0.46, heat, 0.54, 0)


def strip(frames, maps, title, binary=False, color=(30, 50, 235)):
    tiles = []
    for index in range(FRAMES):
        tile = overlay(frames[index], maps[index], binary=binary, color=color)
        cv2.rectangle(tile, (0, 0), (320, 28), (244, 240, 230), -1)
        cv2.putText(tile, f"K{index:02d}/F{index*4:02d}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .54, (25, 31, 29), 1, cv2.LINE_AA)
        tiles.append(tile)
    canvas = np.concatenate(tiles, axis=1)
    header = np.full((42, canvas.shape[1], 3), (237, 232, 219), np.uint8)
    cv2.putText(header, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .70, (22, 38, 31), 2, cv2.LINE_AA)
    return np.concatenate([header, canvas], axis=0)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.render_root.mkdir(parents=True, exist_ok=True)
    frames = read_frames(args.video)
    records = []
    for path in sorted(args.probe_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            mean = np.asarray(data["mean"], dtype=np.float32)
            query_indices = np.asarray(data["query_token_indices"], dtype=np.int64)
            region = str(scalar(data, "region_name"))
            phrase = str(scalar(data, "region_phrase"))
            branch = str(scalar(data, "cfg_branch"))
            step = int(scalar(data, "step"))
            seed = int(scalar(data, "seed"))
            num_heads = int(scalar(data, "num_heads"))
        frame_max = mean.reshape(FRAMES, -1).max(axis=1).clip(min=1e-12)
        normalized = mean / frame_max[:, None, None]
        thresholds = np.quantile(normalized.reshape(FRAMES, -1), args.quantile, axis=1)
        candidate = normalized >= thresholds[:, None, None]
        forward_trajectory = track(
            candidate, normalized, query_indices, args.radius, args.single_component
        )
        trajectory, rejected_events, backward_removed = backward_prune(
            candidate,
            forward_trajectory,
            args.backtrack_frames,
        )
        forbidden = candidate & ~trajectory
        forbidden[:2] = False
        if args.removal_dilate_radius > 0:
            dilate_kernel = np.ones(
                (2 * args.removal_dilate_radius + 1,) * 2, np.uint8
            )
            for frame in range(2, FRAMES):
                expanded = cv2.dilate(
                    forbidden[frame].astype(np.uint8), dilate_kernel
                ) > 0
                forbidden[frame] = expanded & ~trajectory[frame]
        stem = f"seed{seed:06d}__step{step:02d}__{branch}__{region}"
        mask_name = f"{stem}.npz"
        np.savez_compressed(
            args.output_root / mask_name,
            mean=mean,
            normalized=normalized,
            candidate=candidate,
            forward_trajectory=forward_trajectory,
            rejected_events=rejected_events,
            backward_removed=backward_removed,
            trajectory=trajectory,
            forbidden=forbidden,
            query_token_indices=query_indices,
            region_name=np.asarray(region),
            region_phrase=np.asarray(phrase),
            cfg_branch=np.asarray(branch),
            step=np.int32(step),
            seed=np.int32(seed),
            num_heads=np.int32(num_heads),
            quantile=np.float32(args.quantile),
            radius=np.int32(args.radius),
            single_component=np.bool_(args.single_component),
            removal_dilate_radius=np.int32(args.removal_dilate_radius),
            backtrack_frames=np.int32(args.backtrack_frames),
        )
        images = {
            "mean": f"{stem}__mean.jpg",
            "candidate": f"{stem}__candidate.jpg",
            "forward_trajectory": f"{stem}__forward_trajectory.jpg",
            "rejected_events": f"{stem}__rejected_events.jpg",
            "backward_removed": f"{stem}__backward_removed.jpg",
            "trajectory": f"{stem}__trajectory.jpg",
            "forbidden": f"{stem}__forbidden.jpg",
        }
        rendered = {
            "mean": strip(frames, normalized, f"S{step:03d} {branch} · No Intervention Top{num_heads} Mean · per-frame scale"),
            "candidate": strip(
                frames,
                candidate,
                f"All high-response regions · per-frame P{int(round(args.quantile * 100))}",
                True,
                (35, 160, 230),
            ),
            "forward_trajectory": strip(
                frames,
                forward_trajectory,
                "Forward Single-Component Trajectory · before backward correction",
                True,
                (55, 185, 80),
            ),
            "rejected_events": strip(
                frames,
                rejected_events,
                "Rejected Component · triggers backward tracing",
                True,
                (30, 135, 245),
            ),
            "backward_removed": strip(
                frames,
                backward_removed,
                f"Backward-Traced Components · max {args.backtrack_frames} latent frames",
                True,
                (180, 65, 210),
            ),
            "trajectory": strip(
                frames,
                trajectory,
                (
                    f"Corrected Single-Component Trajectory · backtrack {args.backtrack_frames}"
                    if args.single_component
                    else f"Multi-component continuous trajectory · radius {args.radius}"
                ),
                True,
                (55, 185, 80),
            ),
            "forbidden": strip(
                frames,
                forbidden,
                (
                    f"Dilated F_t · radius {args.removal_dilate_radius} · trajectory protected"
                    if args.removal_dilate_radius > 0
                    else "F_t · high response outside continuous trajectory"
                ),
                True,
                (35, 40, 235),
            ),
        }
        for key, image in rendered.items():
            cv2.imwrite(str(args.render_root / images[key]), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
        records.append({
            "seed": seed, "step": step, "cfg_branch": branch,
            "region_name": region, "region_phrase": phrase,
            "num_heads": num_heads, "quantile": args.quantile, "radius": args.radius,
            "single_component": args.single_component,
            "removal_dilate_radius": args.removal_dilate_radius,
            "backtrack_frames": args.backtrack_frames,
            "mask": mask_name, "images": images,
        })
    (args.render_root / "manifest.json").write_text(
        json.dumps({"records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
