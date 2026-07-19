from pathlib import Path

import cv2
import numpy as np


def build_contact_sheet(video_path: Path, output_path: Path, title: str) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    indices = [0, 12, 24, 36, 48]
    tiles = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Failed to read frame {idx} from {video_path}")
        frame = cv2.resize(frame, (288, 192), interpolation=cv2.INTER_AREA)
        label = np.full((48, 288, 3), 255, dtype=np.uint8)
        cv2.putText(
            label,
            f"frame {idx}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        tiles.append(np.concatenate([label, frame], axis=0))

    cap.release()

    row = np.concatenate(tiles, axis=1)
    header = np.full((60, row.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        header,
        f"{title} | frames={frame_count} | fps={fps:.2f} | {video_path.name}",
        (10, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    sheet = np.concatenate([header, row], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    root = Path("/data/gaoya/agent-data/outputs")
    build_contact_sheet(
        root / "difftrack_singlecase_redwheel_pag" / "video_0.mp4",
        root / "redwheel_inspect" / "pag_contact.png",
        "PAG",
    )
    build_contact_sheet(
        root / "difftrack_singlecase_redwheel_baseline" / "video_0.mp4",
        root / "redwheel_inspect" / "baseline_contact.png",
        "Baseline",
    )


if __name__ == "__main__":
    main()
