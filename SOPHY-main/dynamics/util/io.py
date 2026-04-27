import cv2
import json
import torch
import imageio
import mediapy
import numpy as np
from PIL import Image
from pathlib import Path
from natsort import natsorted
from typing import Optional, Tuple

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def gaussian_intrin_scale(x_or_y: torch.Tensor, w_or_h: float):

    ret = ((x_or_y + 1.0) * w_or_h - 1.0) * 0.5

    return ret

def render_arrow_in_screen(viewpoint_camera, points_3d):
    """ Credit to: PhysDreamer """

    full_proj_mat = viewpoint_camera.full_proj_transform

    # [N, 4]
    pts = torch.cat([points_3d, torch.ones_like(points_3d[:, 0:1])], dim=-1)
    # [N, 1, 4] <-  [N, 1, 4] @ [1, 4, 4]
    pts_cam = pts.unsqueeze(-2) @ full_proj_mat.unsqueeze(0)  # [N, 1, 4]

    pts_cam = full_proj_mat.T.unsqueeze(0) @ pts.unsqueeze(-1)

    pts_cam = pts_cam.squeeze(-1)  # [N, 4]
    pts_cam = pts_cam[:, :3] / pts_cam[:, 3:]  # [N, 1, 3]

    pts_cam_yx_pixel = pts_cam[:, :2]

    pts_cam_x, pts_cam_y = pts_cam_yx_pixel[:, 0], pts_cam_yx_pixel[:, 1]

    w, h = viewpoint_camera.image_width, viewpoint_camera.image_height

    pts_cam_x = gaussian_intrin_scale(pts_cam_x, w)
    pts_cam_y = gaussian_intrin_scale(pts_cam_y, h)

    ret_pts_cam_xy = torch.cat(
        [pts_cam_x.unsqueeze(-1), pts_cam_y.unsqueeze(-1)], dim=-1
    )

    return ret_pts_cam_xy

def save_tensor_image(tensor, path, **kwargs):
    tensor_image = tensor.detach().contiguous().cpu().numpy().transpose(1, 2, 0)
    image = np.clip((tensor_image * 255), 0, 255).astype(np.uint8).copy()

    render_force = kwargs.get("render_force", False)
    if render_force:
        # Credit to: PhysDreamer
        means3D = kwargs["means3D"]
        sections = kwargs["sections"]
        frame = kwargs["step"]
        means3D_split = torch.split(means3D, sections, dim=0)
        infos = kwargs["render_force_infos"]
        for info in infos:
            if info["start_frame"] <= frame < info["start_frame"] + info["num_frames"]:
                means3D_selected = means3D_split[info["object_idx"]]
                closest_kernel = means3D_selected[info["closest_kernel_idx"]]
                force = info["force"]
                force = force / force.norm() * 0.1
                two_points = torch.stack([closest_kernel, closest_kernel + force], dim=0)
                arrow_2d = render_arrow_in_screen(kwargs["viewpoint_camera"], two_points)
                arrow_2d = arrow_2d.cpu().numpy()

                start, vec_2d = arrow_2d[0], arrow_2d[1] - arrow_2d[0]
                vec_2d = vec_2d / np.linalg.norm(vec_2d)

                start = start  # + np.array([540.0, 288.0])
                image = cv2.circle(
                    image, (int(start[0]), int(start[1])), 20, (205, 209, 211), 4
                )

                # draw arrow in img
                end = start + vec_2d * 40 # force_in_2d_scale
                end = end.astype(np.int32)
                start = start.astype(np.int32)
                image = cv2.arrowedLine(
                    image, (start[0], start[1]), (end[0], end[1]), (0, 0, 255), 4
                )

    imageio.imwrite(path, image)

def save_tensor_image_v2(tensor, path, **kwargs):
    tensor_image = tensor.detach().contiguous().cpu().numpy()
    image = np.clip((tensor_image * 255), 0, 255).astype(np.uint8).copy()

    imageio.imwrite(path, image)

def save_gif_imageio(
    frame_dir: Path,
    frame_name: str,
    output_path: Path,
    resize: Optional[Tuple[int, int]] = None,
    skip_frame: int = 1,
    fps: int = 30,
    white_bg: bool = False,
):
    np_frames = list()
    image_paths = [i for i in frame_dir.glob(frame_name)]
    image_paths = natsorted(image_paths)[::skip_frame]

    for image_path in image_paths:
        image = Image.open(image_path)
        if resize is not None:
            image = image.resize(size=resize)
        if image.mode == "RGBA":
            background_color = np.array([1, 1, 1]) if white_bg else np.array([0, 0, 0])
            image_rgba = np.array(image)
            norm_rgba = image_rgba / 255.0
            norm_rgba = norm_rgba[:, :, :3] * norm_rgba[:, :, 3:] + (1 - norm_rgba[:, :, 3:]) * background_color
            image_arr = np.array(norm_rgba*255.0, dtype=np.uint8)
        elif image.mode == "RGB":
            image_arr = np.array(image)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        np_frames.append(image_arr)

    with imageio.get_writer(output_path, mode='I', fps=fps, loop=0) as writer:
        for frame in np_frames:
            writer.append_data(frame)

    print(f"GIF saved to {output_path} with skip frame {skip_frame} and fps {fps}")

def save_video_mediapy(
    frame_dir: Path,
    frame_name: str,
    output_path: Path,
    skip_frame: int = 1,
    fps: int = 30,
    white_bg: bool = False,
):
    np_frames = list()
    image_paths = [i for i in frame_dir.glob(frame_name)]
    image_paths = natsorted(image_paths)[::skip_frame]

    for image_path in image_paths:
        image = Image.open(image_path)
        # make sure the image height and width are even
        height_res = image.height % 2
        weight_res = image.width % 2
        if height_res != 0 or weight_res != 0:
            new_height = image.height + height_res
            new_width = image.width + weight_res
            image = image.resize((new_width, new_height))
        if image.mode == "RGBA":
            background_color = np.array([1, 1, 1]) if white_bg else np.array([0, 0, 0])
            image_rgba = np.array(image)
            norm_rgba = image_rgba / 255.0
            norm_rgba = norm_rgba[:, :, :3] * norm_rgba[:, :, 3:] + (1 - norm_rgba[:, :, 3:]) * background_color
            image_arr = np.array(norm_rgba*255.0, dtype=np.uint8)
        elif image.mode == "RGB":
            image_arr = np.array(image)
        else:
            raise ValueError(f"Unsupported image mode: {image.mode}")
        np_frames.append(image_arr)
    
    mediapy.write_video(output_path, np_frames, fps=fps, qp=18)
    print(f"Video saved to {output_path} with skip frame {skip_frame} and fps {fps}")

def safe_symlink(source: Path, link_path: Path):
    if not link_path.exists():
        try:
            link_path.symlink_to(source)
        except Exception as e:
            print(f"Failed to create symlink {link_path} -> {source}: {e}")
