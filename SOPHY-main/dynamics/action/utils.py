import os
import subprocess
from glob import glob
from os.path import join
from omegaconf import OmegaConf
# -------------------------------------- gaoya -------------------------------------- #
import sys
sys.path.append("/home/gaoya/Code_Video/SOPHY-main/")


from dynamics.util.io import read_json

# -------------------------------------- gaoya -------------------------------------- #
from dynamics.util.constants import METADATA

RESULT = "results"
SHAPE_LIST = read_json(os.path.join(METADATA, 'classes.json'))
MAT_COLORS = read_json(os.path.join(METADATA, 'material_colors.json'))
SUB_TYPES = read_json(os.path.join(METADATA, 'sub_materials_fine.json'))


def subprocess_eval(
    ver_dir,
    sim_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, inference_path="inference.py"):

    config_path = join(ver_dir, f'{sim_args["video_name"]}.yaml')

    arguments = [
        'python', inference_path,
        '-c', config_path,
        '-s', str(sim_args["eval_steps"]),
        '-f', str(sim_args["skip_frames"]),
        '-vn', sim_args["video_name"],
    ]

    if sim_args["remove_images"]:
        arguments.append('-ri')
    if sim_args["vis_bbox"]:
        arguments.append('-vb')
    if sim_args["pack_illustration"]:
        arguments.append('-pi')
    if sim_args["save_particles"]:
        arguments.append('-sp')
    if sim_args["save_meshes"]:
        arguments.append('-sm')
    if sim_args["reuse_meshes"]:
        arguments.append('-rm')
    if sim_args["save_with_mp4"]:
        arguments.append('-mp4')
    if len(sim_args["debug_views"]) > 0:
        arguments.extend(['-dv', *sim_args["debug_views"]])

    subprocess.run(arguments, env=os.environ, stdout=stdout, stderr=stderr)


def prepare_config(
    target_path: str,
    config: str,
    split: str,
    category: str,
    shape_id: str,
    style_id: str,
    **kwargs
):
    config = OmegaConf.load(config)
    img_dir = join(target_path, "data", split, category, f'{shape_id}__{style_id}')
    sim_dir = join(target_path, "simulation_data", split, category, f'{shape_id}__{style_id}')
    obj_dir = join(target_path, "object_data", split, category, f'{shape_id}__{style_id}')

    # output dir
    config.output_dir = kwargs.get("output_dir", None)

    # video data

    config.video_data.data.path = config.video_data.data.path or img_dir

    # sim

    config.sim.gravity = config.sim.gravity or kwargs["gravity"]

    # objects

    config.objects[0].name = config.objects[0].name or f'{shape_id}__{style_id}'
    config.objects[0].path = config.objects[0].path or obj_dir

    config.objects[0].kernels.path = config.objects[0].kernels.path or join(img_dir, 'point_cloud.ply')

    config.objects[0].particles.shape.ori_bounds = config.objects[0].particles.shape.ori_bounds or kwargs["ori_bounds"]
    config.objects[0].particles.shape.sim_bounds = config.objects[0].particles.shape.sim_bounds or kwargs["sim_bounds"]

    config.objects[0].particles.vel.lin_vel = config.objects[0].particles.vel.lin_vel or kwargs["lin_vel"]
    config.objects[0].particles.vel.ang_vel = config.objects[0].particles.vel.ang_vel or kwargs["ang_vel"]

    config.objects[0].particles.particles_path = join(sim_dir, "sampled_points.ply")
    config.objects[0].particles.downsample_factor = kwargs["downsample_factor"]

    config.objects[0].material.mat_info_dir = config.objects[0].material.mat_info_dir or sim_dir

    # transforms

    if "transforms" in kwargs:
        if config.objects[0].get("transforms") is None:
            config.objects[0].transforms = kwargs["transforms"]

    # boundary conditions

    if "boundary_conditions" in kwargs:
        if config.get("boundary_conditions") is None:
            config.boundary_conditions = kwargs["boundary_conditions"]

    return config

def prepare_config_for_mesh(
    target_path: str,
    config: str,
    split: str,
    category: str,
    shape_id: str,
    style_id: str,
    **kwargs
):
    config = OmegaConf.load(config)
    img_dir = join(target_path, "data", split, category, f'{shape_id}__{style_id}')
    sim_dir = join(target_path, "simulation_data", split, category, f'{shape_id}__{style_id}')
    obj_dir = join(target_path, "object_data", split, category, f'{shape_id}__{style_id}')

    # output dir
    config.output_dir = kwargs.get("output_dir", None)

    # video data

    config.video_data.data.path = config.video_data.data.path or img_dir

    # sim

    config.sim.gravity = config.sim.gravity or kwargs["gravity"]

    # objects

    config.objects[0].name = config.objects[0].name or f'{shape_id}__{style_id}'
    config.objects[0].path = config.objects[0].path or obj_dir

    config.objects[0].particles.shape.ori_bounds = config.objects[0].particles.shape.ori_bounds or kwargs["ori_bounds"]
    config.objects[0].particles.shape.sim_bounds = config.objects[0].particles.shape.sim_bounds or kwargs["sim_bounds"]

    config.objects[0].particles.vel.lin_vel = config.objects[0].particles.vel.lin_vel or kwargs["lin_vel"]
    config.objects[0].particles.vel.ang_vel = config.objects[0].particles.vel.ang_vel or kwargs["ang_vel"]

    config.objects[0].particles.particles_path = join(sim_dir, "sampled_points.ply")
    config.objects[0].particles.downsample_factor = kwargs["downsample_factor"]

    mesh_paths = glob(join(sim_dir, f"*{sim_dir.split('/')[-1]}.obj"))
    assert len(mesh_paths) == 1, f"Found {len(mesh_paths)} mesh files in {sim_dir}, expected 1."
    config.objects[0].particles.mesh_path = mesh_paths[0]

    config.objects[0].material.mat_info_dir = config.objects[0].material.mat_info_dir or sim_dir

    # transforms

    if "transforms" in kwargs:
        if config.objects[0].get("transforms") is None:
            config.objects[0].transforms = kwargs["transforms"]

    # boundary conditions

    if "boundary_conditions" in kwargs:
        if config.get("boundary_conditions") is None:
            config.boundary_conditions = kwargs["boundary_conditions"]

    return config

def animation_exists(ver_dir, shape_id, style_id, test_case, ver_identifier, debug_views, ext="gif"):
    for view in debug_views:
        if len(glob(join(ver_dir, f'{shape_id}__{style_id}-{test_case}_{ver_identifier}_{view}.{ext}'))) == 0:
            return False
    return True
