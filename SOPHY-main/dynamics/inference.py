import os
import torch
import random
import trimesh
import argparse
import nvdiffrast
import subprocess
import warp as wp
import numpy as np
import kaolin as kal
from PIL import Image
from typing import List
from pathlib import Path
from natsort import natsorted
import matplotlib.pyplot as plt
from tqdm.autonotebook import trange
from omegaconf import DictConfig, OmegaConf
from mpm.object_utils import (
    PhysObject,
    prepare_boundary_conditions_given_mesh,
    prepare_simulation_environment_given_mesh
)
from mpm.mpm_data_structure import (
    denormalize_points,
    denormalize_points_helper_func
)

from safetensors import safe_open
from safetensors.torch import save_file
from util.io import save_tensor_image_v2, save_gif_imageio, save_video_mediapy
from util.dataset import ViewDataset
from util.dataprep import mesh_rasterization
# from utils.dataprep import mesh_rasterization_v2 as mesh_rasterization

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", "-c", type=str, required=True,
        help="Path to the config file."
    )
    parser.add_argument(
        "--eval_steps", "-s", type=int, default=600,
        help="Number of simulation steps."
    )
    parser.add_argument(
        "--skip_frames", "-f", type=int, default=1,
        help="Number of skip frames when packing the video."
    )
    parser.add_argument(
        "--remove_images", "-ri", action="store_true",
        help="Whether to remove images after packing video."
    )
    parser.add_argument(
        "--video_name", "-vn", type=str, default=None,
        help="Save video name."
    )
    parser.add_argument(
        "--annotated_name", "-an", type=str, default=None,
        help="Annotation for the video name."
    )
    parser.add_argument(
        "--vis_bbox", "-vb", action="store_true",
        help="Visualize bounding box."
    )
    parser.add_argument(
        "--debug_views", "-dv", nargs='+', default=[],
        help="Views for rendering."
    )
    parser.add_argument(
        "--save_with_mp4", "-mp4", action="store_true",
        help="Save video in mp4 format."
    )
    parser.add_argument(
        "--save_particles", "-sp", action="store_true",
        help="Save state name for evaluation."
    )
    parser.add_argument(
        "--save_meshes", "-sm", action="store_true",
        help="Save gaussians for visualization."
    )
    parser.add_argument(
        "--reuse_meshes", "-rm", action="store_true",
        help="Reuse gaussians for visualization."
    )
    parser.add_argument(
        "--pack_illustration", "-pi", action="store_true",
        help="Pack illustration."
    )
    parser.add_argument(
        "--dataset_path", type=str, default=None,
        help="Rewrite video dataset path."
    )
    parser.add_argument(
        "--image_size", type=int, default=256,
        help="Square image size."
    )

    args = parser.parse_args()
    return args


def load_render_pack(load_path: Path, device):
    verts = list()
    with safe_open(load_path, framework="pt") as f:
        for i in range(int(f.metadata()['num_meshes'])):
            verts.append(f.get_tensor(f'vertices_{i}').to(device))
    return verts


def save_render_pack(meshes: List[kal.rep.SurfaceMesh], save_path: Path):
    tensors = dict()
    profile = dict()
    for i, mesh in enumerate(meshes):
        tensors[f'vertices_{i}'] = mesh.vertices.cpu()
        profile[f'n_verts_{i}'] = str(mesh.vertices.shape[0])
        profile[f'n_faces_{i}'] = str(mesh.faces.shape[0])
    profile['num_meshes'] = str(len(meshes))
    
    save_file(tensors, save_path, metadata=profile)


def pack_images(image_paths: List[str], video_name: str, view: str, overlap_fraction=0.5):
    images = list()
    for path in image_paths:
        rgba = Image.open(path)
        rgba = rgba.convert("RGBA")
        images.append(rgba)

    total_width = 0
    max_height = 0

    for img in images:
        img_width, img_height = img.size
        total_width += int(img_width * (1 - overlap_fraction))
        max_height = max(max_height, img_height)

    total_width += int(img_width * (1 - overlap_fraction))
    packed_image = Image.new("RGBA", (total_width, max_height))

    current_x = 0
    for img in images:  # Reverse the order of images to ensure greater indices are in the back
        img_width, img_height = img.size
        overlap_width = int(img_width * overlap_fraction)

        image_to_paste = img

        # Blend the image with the packed image (preserving non-transparent pixels)
        for x in range(image_to_paste.width):
            for y in range(image_to_paste.height):
                pixel = image_to_paste.getpixel((x, y))
                # If the pixel is non-transparent, copy it to the packed image
                if pixel[3] > 0:  # Check if alpha channel is non-transparent
                    packed_image.putpixel((current_x + x, y), pixel)

        # Move the current position by the image width minus the overlap
        current_x += img_width - overlap_width

    packed_image = np.array(packed_image).astype(np.float32)
    packed_image /= 255.0
    white_bg = np.ones_like(packed_image[..., :3])
    packed_image = white_bg * (1 - packed_image[..., 3:]) + packed_image[..., :3] * packed_image[..., 3:]
    tmp_path = Path(image_paths[0])
    plt.imsave(tmp_path.parent.parent.parent / f"{video_name}_{view}.png", packed_image)
    print(f'illustration saved to {tmp_path.parent.parent.parent / f"{video_name}_{view}.png"}')


def pack_illustration(image_dir: Path, view: str, video_name: str):
    tmp_dir = image_dir / view
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not (tmp_dir.parent / f'{tmp_dir.name}_sod').exists():
        desired_images = natsorted(list(image_dir.glob(f'{view}_*.png')))
        for i, file in enumerate(desired_images):
            if i % 16 == 0 or i == len(desired_images) - 1:
                file.rename(tmp_dir / f'{file.name}')

        subprocess.run([
            'transparent-background',
            '--source', tmp_dir.as_posix(),
            '--dest', f'{tmp_dir.as_posix()}_sod',
            '--type', 'rgba',
        ])

    pack_images(
        natsorted(list((tmp_dir.parent / f'{tmp_dir.name}_sod').glob('*.png'))),
        video_name,
        view,
        overlap_fraction=0.2
    )


def debug_points(vertices):
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.numpy()
    print(vertices[:, 0].min(), vertices[:, 0].max())
    print(vertices[:, 1].min(), vertices[:, 1].max())
    print(vertices[:, 2].min(), vertices[:, 2].max())


def load_mesh(
    path, x_min, x_max, y_min, y_max, z_min, z_max, normal
) -> kal.rep.SurfaceMesh:
    default_settings = {
        'with_materials': True, 
        'with_normals': False, 
        "triangulate": True,
        'heterogeneous_mesh_handler': kal.io.utils.mesh_handler_naive_triangulate
    }
    mesh = kal.io.obj.import_mesh(
        path, **default_settings, raw_materials=True
    )
    v0 = mesh.vertices[:, 0].clone()
    v1 = mesh.vertices[:, 1].clone()
    v2 = mesh.vertices[:, 2].clone()
    if normal == 0:
        mesh.vertices[:, 0] = v1
        mesh.vertices[:, 1] = v0
        mesh.vertices[:, 2] = v2
    elif normal == 1:
        pass
    else:
        mesh.vertices[:, 0] = v0
        mesh.vertices[:, 1] = v2
        mesh.vertices[:, 2] = v1
    # scaling the vertices along each direction specified by min max
    if normal != 0:
        mxr = mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min()
        ratio = (x_max - x_min) / mxr
        mesh.vertices[:, 0] = mesh.vertices[:, 0] * ratio + (x_max + x_min) / 2
    if normal != 1:
        myr = mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min()
        ratio = (y_max - y_min) / myr
        mesh.vertices[:, 1] = mesh.vertices[:, 1] * ratio + (y_max + y_min) / 2
    if normal != 2:
        mzr = mesh.vertices[:, 2].max() - mesh.vertices[:, 2].min()
        ratio = (z_max - z_min) / mzr
        mesh.vertices[:, 2] = mesh.vertices[:, 2] * ratio + (z_max + z_min) / 2
    info = [x_min, y_min, z_min]
    mesh.vertices[:, normal] = info[normal]
    return mesh


def load_mesh_ori(
    path
) -> kal.rep.SurfaceMesh:
    default_settings = {
        'with_materials': True, 
        'with_normals': False, 
        "triangulate": True,
        'heterogeneous_mesh_handler': kal.io.utils.mesh_handler_naive_triangulate
    }
    mesh = kal.io.obj.import_mesh(
        path, **default_settings, raw_materials=True
    )
    return mesh


def evaluate(config: DictConfig):

    # init

    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    wp.init()
    wp_device = wp.get_device(config.device)
    wp.ScopedTimer.enabled = False
    wp.set_module_options({'fast_math': False})

    torch_device = torch.device(config.device)
    torch.backends.cudnn.benchmark = True

    requires_grad = False

    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device=torch_device)
        if config.get("white_background", False)   # default to black background
        else torch.tensor([0, 0, 0], dtype=torch.float32, device=torch_device)
    )

    nvctx = nvdiffrast.torch.RasterizeCudaContext(device='cuda')

    # path

    output_dir = config.get("output_dir", "results")
    output_dir = Path(output_dir)

    video_dir: Path = output_dir

    image_dir: Path = output_dir / f'images_{config.video_name}'
    image_dir.mkdir(parents=True, exist_ok=True)

    if config.save_particles:
        state_dir: Path = output_dir / f'states_{config.video_name}'
        state_dir.mkdir(parents=True, exist_ok=True)

    if config.save_meshes:
        meshes_dir: Path = output_dir / f'meshes_{config.video_name}'
        meshes_dir.mkdir(parents=True, exist_ok=True)

    if config.reuse_meshes:
        meshes_dir: Path = output_dir / f'meshes_{config.video_name}'
        assert meshes_dir.exists(), f"Reuse meshes but meshes dir not found: {meshes_dir}"

    # env

    if config.dataset_path is not None:
        config.video_data.data.path = config.dataset_path
        print(f'Rewrite video dataset path to\n\t{config.dataset_path}')
    if len(config.get("debug_views", list())) > 0:
        config.video_data.data.used_views = config.debug_views
    config.video_data.data.white_background = config.get("white_background", False)
    dataset = ViewDataset(config.video_data)
    first_step = dataset.steps[0]

    eval_steps = config.eval_steps
    phys_objects: List[PhysObject] = list()

    for obj_config in config.objects:
        phys_object = PhysObject(
            config = obj_config,
            device = config.device,
        )
        phys_objects.append(phys_object)

    env_pack = prepare_simulation_environment_given_mesh(
        num_grids=config.sim.num_grids,
        gravity=config.sim.gravity,
        objects=phys_objects,
        device=config.device,
        max_e_order=config.sim.get('max_e_order', 9),
        requires_grad=requires_grad,
    )

    mpm_solver = env_pack['mpm_solver']
    mpm_model = env_pack['mpm_model']
    mpm_state = env_pack['mpm_state']
    sections = env_pack['sections']
    state_initializer = env_pack['state_initializer']
    obj_kal_meshes = env_pack['obj_kal_meshes']
    obj_mv_indices = env_pack['obj_mv_indices']

    bc_outputs = dict()
    if config.get('boundary_conditions') is not None:
        bc_outputs = prepare_boundary_conditions_given_mesh(
            obj_mv_indices=obj_mv_indices,
            state_initializer=state_initializer,
            bc_configs=config.boundary_conditions,
            mpm_state=mpm_state,
            mpm_solver=mpm_solver,
            num_grids=config.sim.num_grids,
            dt=config.sim.dt,
            device=config.device,
        )

    if config.vis_bbox:
        # bounding box params
        tmp_group = state_initializer.groups[0]
        # adjust floor if modified in boundary conditions
        tmp_min = [0., 0., 0.]
        if 'floor_info' in bc_outputs:
            floor_point = bc_outputs['floor_info']['point']
            floor_normal = bc_outputs['floor_info']['normal']
            axis = np.where(floor_normal==1)[0].item()
            tmp_min[axis] = floor_point[axis]
        if config.get('denormalize', False):
            denorm_bb_min = denormalize_points_helper_func(
                torch.tensor(tmp_min), tmp_group.size, tmp_group.center
            )
            denorm_bb_max = denormalize_points_helper_func(
                torch.tensor([1., 1., 1.]), tmp_group.size, tmp_group.center
            )
        else:
            denorm_bb_min = torch.tensor(tmp_min)
            denorm_bb_max = torch.tensor([1., 1., 1.])

        denorm_bb_min = denorm_bb_min.numpy().tolist()
        denorm_bb_max = denorm_bb_max.numpy().tolist()
        print(denorm_bb_min, denorm_bb_max)

        plane_path = "asset/floor.obj"
        floor_mesh = load_mesh_ori(plane_path)
        floor_mesh.vertices[:, 1] = denorm_bb_min[1]
        debug_points(floor_mesh.vertices)
        floor_mesh = floor_mesh.to(torch_device)
        bg_mesh = [floor_mesh]

    prev_state = mpm_state

    # simulation

    # -- first step
    if not config.get('denormalize', False):
        # if not denormalize, this means the visualization is performed in the simulation bounds,
        # so we need to transform the gaussians from the original bounds to the simulation bounds
        # for visualization
        for i, group_data in enumerate(state_initializer.groups):
            # tmp_pos = torch.from_numpy(group_data.pos[obj_mv_indices[i]].clone()).to(torch_device)
            tmp_pos = torch.from_numpy(group_data.pos).float().to(torch_device)
            assert tmp_pos[obj_mv_indices[i]].shape[0] == obj_kal_meshes[i].vertices.shape[0], \
                f'{tmp_pos[obj_mv_indices[i]].shape[0]} != {obj_kal_meshes[i].vertices.shape[0]}'
            obj_kal_meshes[i].vertices = tmp_pos[obj_mv_indices[i]].clone()

    # FIXME: this is a hack to make the mesh visible
    _, render = mesh_rasterization(
        meshes=obj_kal_meshes,
        camera=dataset.getCameras(0, first_step).to(torch_device),
        background=background,
        nvctx=nvctx,
        return_alpha=True,
        background_mesh=bg_mesh if config.vis_bbox else None,
    )

    for view in dataset.views:
        if view in config.get("debug_views", list()):
            # rasterize deformed meshes
            _, render = mesh_rasterization(
                meshes=obj_kal_meshes,
                camera=dataset.getCameras(view, first_step).to(torch_device),
                background=background,
                nvctx=nvctx,
                return_alpha=True,
                background_mesh=bg_mesh if config.vis_bbox else None,
            )
            save_tensor_image_v2(render, image_dir / f'{view}_{first_step:03d}.png')
    
    for step in trange(1, eval_steps + 1):
        if not config.reuse_meshes:
            next_state = prev_state.partial_clone(requires_grad=False)
            mpm_solver.p2g2p_differentiable(
                mpm_model,
                prev_state,
                next_state,
                dt=config.sim.dt,
                device=wp_device
            )

            x = wp.to_torch(next_state.particle_x).clone()
            de_x = denormalize_points(x, sections, state_initializer) if config.get('denormalize', False) else x

            prev_state = next_state

            # update mesh vertices
            for i, sec_x in enumerate(torch.split(de_x, sections, dim=0)):
                assert sec_x[obj_mv_indices[i]].shape[0] == obj_kal_meshes[i].vertices.shape[0], \
                    f'{sec_x[obj_mv_indices[i]].shape[0]} != {obj_kal_meshes[i].vertices.shape[0]}'
                obj_kal_meshes[i].vertices = sec_x[obj_mv_indices[i]].clone()

        if step % config.skip_frames == 0:
            if config.save_particles:
                t = trimesh.PointCloud(vertices=x.clone().detach().cpu())
                t.export(state_dir / f'{first_step + step:03d}.ply')

            if config.save_meshes:
                save_render_pack(obj_kal_meshes, meshes_dir / f'{first_step + step:03d}.safetensors')

            if config.reuse_meshes:
                verts = load_render_pack(meshes_dir / f'{first_step + step:03d}.safetensors', device=torch_device)
                assert len(verts) == len(obj_kal_meshes), \
                    f'Length of stored meshes misaligns with the running ones: {len(verts)} != {len(obj_kal_meshes)}'
                for i, mesh in enumerate(verts):
                    obj_kal_meshes[i].vertices = mesh

            for view in dataset.views:
                if view in config.get("debug_views", list()):
                    # rasterize deformed meshes
                    _, render = mesh_rasterization(
                        meshes=obj_kal_meshes,
                        camera=dataset.getCameras(view, first_step).to(torch_device),
                        background=background,
                        nvctx=nvctx,
                        return_alpha=True,
                        background_mesh=bg_mesh if config.vis_bbox else None,
                    )
                    save_tensor_image_v2(render, image_dir / f'{view}_{first_step + step:03d}.png')

    # pack video

    fps = 30

    for view in dataset.views:
        if view in config.get("debug_views", list()):
            if config.save_with_mp4:
                save_video_mediapy(
                    image_dir, f"{view}_*.png",
                    video_dir / f"{config.video_name}_{view}.mp4",
                    skip_frame=1, fps=fps, white_bg=True,
                )
            else:
                save_gif_imageio(
                    image_dir, f"{view}_*.png", 
                    video_dir / f"{config.video_name}_{view}.gif",
                    (256, 256), 1, fps=fps, white_bg=True,
                )                
            
    if config.remove_images:
        # os.system(f"rm -f {image_root}/{view}_*.png")
        os.system(f"rm -rf {image_dir}")


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config)
    config.update(vars(args))

    # resolve default video name
    if config.video_name is None:
        config.video_name = config.config.split('/')[-1].split('.')[0]
    # resolve annotated video name
    if config.annotated_name is not None:
        config.video_name = f"{config.video_name}_{config.pop('annotated_name')}"

    config = DictConfig(config)

    with torch.no_grad():
        evaluate(config)



'''

python /home/gaoya/Code_Video/SOPHY-main/geometry/generate_image_cond.py \
    --ae-pth /data/gaoya/ckpt/SOPHY/output/ae/shared/checkpoint-0.pth \
    --dm-pth /data/gaoya/ckpt/SOPHY/output/dm/shared_image/checkpoint-0.pth \
    --num_samples_per_cond 1 \
    --cond_version eval \
    --cond_path v2



'''