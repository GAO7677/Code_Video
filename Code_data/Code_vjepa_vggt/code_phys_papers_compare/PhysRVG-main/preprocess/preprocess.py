import argparse
import json
import os

import imageio
import numpy as np
import torch
from transformers import Sam2VideoModel, Sam2VideoProcessor
from transformers.video_utils import load_video

MODEL_PATH = os.path.join(os.getcwd(), "models", "sam2.1-hiera-large")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam2VideoModel.from_pretrained(MODEL_PATH).to(device, dtype=torch.bfloat16)
processor = Sam2VideoProcessor.from_pretrained(MODEL_PATH)


def track_object(video_path, point):
    """
    Track an object through a video using SAM2.

    Args:
        video_path: str, path to the input video file
        point: [x, y] single point coordinate, e.g. [220, 210]

    Returns:
        video_segments: dict, mapping frame_idx -> mask tensor
    """
    # load video
    video_frames, _ = load_video(video_path)

    # Initialize video inference session
    inference_session = processor.init_video_session(
        video=video_frames,
        inference_device=device,
        dtype=torch.bfloat16,
    )

    ann_frame_idx = 0
    ann_obj_id = 1

    # Convert to SAM2 format: [[[[x, y]]]]
    sam2_points = [[[point]]]
    sam2_labels = [[[1]]]

    processor.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
        obj_ids=ann_obj_id,
        input_points=sam2_points,
        input_labels=sam2_labels,
    )

    # Segment the object on the first frame
    outputs = model(
        inference_session=inference_session,
        frame_idx=ann_frame_idx,
    )

    video_res_masks = processor.post_process_masks(
        [outputs.pred_masks],
        original_sizes=[[inference_session.video_height, inference_session.video_width]],
        binarize=False,
    )[0]
    print(f"Segmentation shape: {video_res_masks.shape}")

    # Propagate through the entire video
    video_segments = {}
    for sam2_video_output in model.propagate_in_video_iterator(inference_session):
        video_res_masks = processor.post_process_masks(
            [sam2_video_output.pred_masks],
            original_sizes=[[inference_session.video_height, inference_session.video_width]],
            binarize=False,
        )[0]
        video_segments[sam2_video_output.frame_idx] = video_res_masks

    print(f"Tracked object through {len(video_segments)} frames")
    return video_segments


def get_video_fps(video_path):
    """Read FPS from a video file."""
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 30)
    reader.close()
    return fps


def save_mask_video(video_segments, save_path, fps):
    """
    Save tracked mask segments as an MP4 video.

    Args:
        video_segments: dict, mapping frame_idx -> mask tensor
        save_path: str, path to save the output .mp4 file
        fps: int, frames per second
    """
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    frames = []
    for i in range(len(video_segments)):
        mask = video_segments[i][0, 0, :]
        mask_numpy = mask.detach().float().cpu().numpy()
        frame = (mask_numpy > 0).astype(np.uint8) * 255
        frame_rgb = np.stack([frame, frame, frame], axis=-1)
        frames.append(frame_rgb)

    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved mask video to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="SAM2 object tracking from video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--info", type=str, required=True, help="Path to info.jsonl with object points")
    args = parser.parse_args()

    # Read info.jsonl
    with open(args.info, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            break

    video_dir = os.path.dirname(args.video) or "."
    fps = get_video_fps(args.video)

    # Track object_1
    point1 = data["object_1"]
    save_path1 = os.path.join(video_dir, "mask_object_1.mp4")
    print(f"\n=== Tracking object_1 at {point1} ===")
    segments1 = track_object(args.video, point1)
    save_mask_video(segments1, save_path1, fps=fps)

    # Track object_2
    point2 = data["object_2"]
    save_path2 = os.path.join(video_dir, "mask_object_2.mp4")
    print(f"\n=== Tracking object_2 at {point2} ===")
    segments2 = track_object(args.video, point2)
    save_mask_video(segments2, save_path2, fps=fps)


if __name__ == "__main__":
    main()
