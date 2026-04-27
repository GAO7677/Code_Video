import os
import json
import argparse
from glob import glob
from os.path import join
from util.constants import DATA_PATH

TEMPLATE_CAMERA_JSON = "output/obj/data_static.json"

with open(join("dataset", "metadata", "classes.json"), 'r') as f:
    SHAPE_LIST = json.load(f)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['dm', 'ae'], required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--identifier', '-id', type=str, default=None)
    return parser.parse_args()


def soft_link_one_object(shape_id, style_id, source_dir, mode, identifier, split='test'):
    # 0. resolve the target directory
    if shape_id.startswith("xx"):
        shape_name = "any"
    else:
        shape_name = SHAPE_LIST[int(shape_id.split("_")[0], 16)]
    target_sim_dir = join(DATA_PATH, "generated_cache", f'{mode}-{identifier}', 'simulation_data', split, shape_name, f"{shape_id}__{style_id}")
    os.makedirs(target_sim_dir, exist_ok=True)
    target_data_dir = join(DATA_PATH, "generated_cache", f'{mode}-{identifier}', 'data', split, shape_name, f"{shape_id}__{style_id}")
    os.makedirs(target_data_dir, exist_ok=True)

    # 1. simulation data
    mmid_src = join(source_dir, f"{shape_id}__{style_id}", "mmid.ply")
    part_src = join(source_dir, f"{shape_id}__{style_id}", "part.ply")
    sampled_info_src = join(source_dir, f"{shape_id}__{style_id}", "sampled_points_info.npz")
    sampled_points = join(source_dir, f"{shape_id}__{style_id}", "sampled_points.ply")
    mesh_paths = glob(join(source_dir.replace("mats", "meshes"), f"{split}_*_{shape_id}__{style_id}.obj"))  # NOTE: here
    # mesh_paths = glob(join(source_dir.replace("mats", "meshes"), f"{shape_id}__{style_id}.obj"))  # NOTE: here
    assert len(mesh_paths) == 1, f'Found {len(mesh_paths)} mesh files for {shape_id}__{style_id}'
    mesh_src = mesh_paths[0]
    mesh_texture_src = mesh_src.replace(".obj", ".png")
    mesh_mtl_src = mesh_src.replace(".obj", ".mtl")

    for src in [mmid_src, part_src, mesh_src, mesh_texture_src, mesh_mtl_src, sampled_info_src, sampled_points]:
        if os.path.exists(src) and not os.path.exists(join(target_sim_dir, src.split("/")[-1])):
            os.system(f'ln -s {os.path.abspath(src)} {target_sim_dir}')

    # 2. data
    data_json_src = TEMPLATE_CAMERA_JSON
    data_cond_image = join(source_dir, f"{shape_id}__{style_id}", "cond.png")
    data_cond_text = join(source_dir, f"{shape_id}__{style_id}", "cond.txt")

    for src in [data_json_src, data_cond_image, data_cond_text]:
        if os.path.exists(src) and not os.path.exists(join(target_data_dir, src.split("/")[-1])):
            os.system(f'cp {os.path.abspath(src)} {target_data_dir}')


def soft_link(mats_dir, mode, identifier, split='test'):
    print(f'Soft linking from {mats_dir} to {DATA_PATH}/generated_cache/{mode}-{identifier} ...')
    for model_code in os.listdir(mats_dir):
        shape_id, style_id = model_code.split("__")
        soft_link_one_object(shape_id, style_id, mats_dir, mode, identifier, split)


def process_data(args):
    mesh_dir = args.mesh_dir
    assert os.path.exists(mesh_dir), f"{mesh_dir} not found"
    assert os.path.isdir(mesh_dir), f"{mesh_dir} is not a directory"

    if args.identifier is None:
        if args.mode == "ae":
            ckpt_idx = mesh_dir.split("/")[-1].split("-")[-1]
            identifier = f'{mesh_dir.split("/")[-2]}-{ckpt_idx}'
        elif args.mode == "dm":
            ckpt_idx = f'{mesh_dir.split("/")[-1].split("-")[-2]}_{mesh_dir.split("/")[-1].split("-")[-1]}'
            identifier = f'{mesh_dir.split("/")[-3]}-{mesh_dir.split("/")[-2]}-{ckpt_idx}'
        else:
            ckpt_idx = mesh_dir.split("/")[-1].split("-")[-1]
            identifier = f'{mesh_dir.split("/")[-3]}_{mesh_dir.split("/")[-2]}-{ckpt_idx}'
    else:
        identifier = args.identifier
    soft_link(
        mesh_dir.replace("meshes", "mats"),
        mode=args.mode,
        identifier=identifier,
        split=args.split
    )


if __name__ == "__main__":
    args = arg_parser()
    process_data(args)
