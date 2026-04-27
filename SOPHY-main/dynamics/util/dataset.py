import os
import math
import torch
import numpy as np
import kaolin as kal


import sys
sys.path.append("/home/gaoya/Code_Video/SOPHY-main/")
# -------------------------------------- gaoya -------------------------------------- #
from dynamics.util.io import read_json
from torch.utils.data import Dataset
from omegaconf import OmegaConf, DictConfig


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


class ViewDataset(Dataset):
    def __init__(self, config: DictConfig, readCameras: bool=True):
        self.eval = config.eval
        self.cameras = {}
        if readCameras:
            self.readCameras(config)

    def readViews(
        self,
        path,
        transformsfile,
        image_width,
        image_height,
        init_frame=None,
        exclude_steps=[-1],
        used_views=None,
        fov_multiplier=1.,
        **kwargs
    ):
        cam_infos = list()
        subfolder = transformsfile.split(".")[0]
        print(f"Reading Synthetic View [{subfolder}] with init_frame={init_frame}...")

        # check how many views do we have automatically
        idx = 0
        contents = read_json(os.path.join(path, transformsfile))
        meta_info = dict()
        for entry in contents:
            file_path = entry.pop("file_path")
            file_idx = file_path.split("/")[-1].split(".")[0]
            meta_info[file_idx] = entry

        views = set()
        steps = set()
        for m in meta_info:
            view = str(m.rsplit("_", 1)[0])
            if used_views is None or view in used_views:
                views.add(view)
            step = int(m.rsplit("_", 1)[1])
            if step not in exclude_steps:
                steps.add(step)
        views = sorted(list(views))
        steps = sorted(list(steps))
        print(f"Views found: {views if len(views) < 20 else views[:20]} {'' if len(views) < 20 else f'#all: {len(views)} ...'}\n"
            f"Steps found: {steps if len(steps) < 20 else steps[:20]} {'' if len(steps) < 20 else f'#all: {len(steps)} ...'}")

        # only read the first frame if `init_frame` is set
        steps = [init_frame] if init_frame is not None else steps
        for view in views:
            for step in steps:
                file_idx_to_fetch = f"{view}_{step:03d}"

                assert file_idx_to_fetch in meta_info, f"File {file_idx_to_fetch} not found in meta_info!"

                # NeRF 'transform_matrix' is a camera-to-world transform
                c2w = np.array(meta_info[file_idx_to_fetch]["c2w"]).astype(np.float32)

                w2c = np.linalg.inv(c2w)

                intrinsics = meta_info[file_idx_to_fetch]["intrinsic"]
                focalx = intrinsics[0][0] / fov_multiplier
                fovx = focal2fov(focalx, image_width) * fov_multiplier

                cam_infos.append({
                    "view": view,
                    "step": step,
                    "view_matrix": w2c,
                    "fov": fovx,
                    "width": image_width,
                    "height": image_height,
                })

                idx += 1

        return {'cam_infos': cam_infos, 'views': views, 'steps': steps}

    def readCameras(self, config: DictConfig):
        self.cameras = {}
        mode = "Training" if not self.eval else "Testing"
        print(f"Reading {mode} Data")
        info = self.readViews(**config.data)
        self.views = info["views"]  # sorted
        self.steps = info["steps"]  # sorted
        self.length = len(self.views) * len(self.steps)

        print(f"Loading {mode} Cameras")
        if config.camera.get("data_device") is None:
            config.camera.data_device = config.device
        print(f'Setting default device for camera data to [{config.camera.data_device}]')
        for cam in info["cam_infos"]:
            # build camera
            camera = kal.render.camera.Camera.from_args(
                view_matrix=torch.tensor(cam["view_matrix"]).to(config.camera.data_device),
                fov=cam["fov"],
                width=cam["width"],
                height=cam["height"]
            )
            if cam["view"] in self.cameras:
                self.cameras[cam["view"]].update({
                    cam["step"]: camera
                })
            else:
                self.cameras.update({
                    cam["view"]: {cam["step"]: camera}
                })

        print(f"Loaded the Camera Set with {len(self.cameras)} views and {len(self.steps)} steps")
        if len(self.views) < 20:
            print(f"    Views: {self.views}")
        if len(self.steps) < 20:
            print(f"    Steps: {self.steps}")

    def getCameras(self, view, step) -> kal.render.camera.Camera:
        if isinstance(view, int):
            view = self.views[view]
        elif isinstance(view, str):
            pass
        else:
            raise ValueError(f"view must be an integer or a string, but got {view} ({type(view)})")
        return self.cameras[view][step]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # modulate idx to get view and step
        idx = idx % self.length
        view_id = idx // len(self.steps)
        view = self.views[view_id]
        step = idx % len(self.steps)
        
        return self.cameras[view][step]
