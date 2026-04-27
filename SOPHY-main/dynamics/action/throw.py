import os
import sys
import time
import argparse
from glob import glob
from tqdm import tqdm
from os.path import join
from typing import List, Optional, Literal
from omegaconf import DictConfig, OmegaConf
from action.utils import (
    prepare_config_for_mesh,
    subprocess_eval,
    animation_exists,
)
from util.constants import DATA_PATH


TEST_CASE = "throw"
DEBUG_VIEWS = ["s_34"]
VER_DIR = "gen_data"


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocess 3DCoMPaT200 dataset')
    parser.add_argument('--target_name', type=str, required=True)
    parser.add_argument('--config', type=str, default='config/mesh.yaml')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    parser.add_argument('--indices', '-ids', type=str, nargs='+', default=None)
    parser.add_argument('--shape_start', type=int, default=None)
    parser.add_argument('--shape_end', type=int, default=None)
    parser.add_argument('--lin_vel', type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument('--save_mp4', '-mp4', action='store_true')
    parser.add_argument('--pack_illustration', '-pi', action='store_true')
    parser.add_argument('--save_particles', '-sp', action='store_true')
    parser.add_argument('--save_meshes', '-sm', action='store_true')
    parser.add_argument('--reuse_meshes', '-rm', action='store_true')
    parser.add_argument('--vis_bbox', '-vb', action='store_true')
    parser.add_argument('--ver_identifier', '-ver', type=str, default="default")
    return parser.parse_args()


def get_additional_args_default(lin_vel):
    additional_args = dict()

    gravity = [0.0, -9.8, 0.0]
    ori_bounds = [
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5]
    ]
    sim_bounds = [
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75]
    ]
    # lin_vel = [1.0, 0.0, 0.0]
    ang_vel = [0.0, 0.0, 0.0]
    boundary_conditions = list()
    boundary_conditions.append({
        "type": "floor",
        "object_idx": 0,
        "surface": "slip",
        "friction": 1.0,
        "normal": [0.0, 1.0, 0.0],
        "height": 0.1,
        "start_frame": 0,
        "num_frames": 1e9
    })

    additional_args["gravity"] = gravity
    additional_args["ori_bounds"] = ori_bounds
    additional_args["sim_bounds"] = sim_bounds
    additional_args["lin_vel"] = lin_vel
    additional_args["ang_vel"] = ang_vel
    additional_args["downsample_factor"] = 10
    additional_args["boundary_conditions"] = boundary_conditions

    return additional_args


def get_sim_args_default(
    save_mp4: bool=False,
    pack_illustration: bool=False,
    save_particles: bool=False,
    save_meshes: bool=False,
    reuse_meshes: bool=False,
    vis_bbox: bool=False,
):
    sim_args = dict()

    sim_args["eval_steps"] = 40000 * 2
    sim_args["skip_frames"] = 400 * 2
    sim_args["remove_images"] = True
    sim_args["debug_views"] = DEBUG_VIEWS
    sim_args["dataset_path"] = None

    sim_args["save_with_mp4"] = save_mp4
    sim_args["pack_illustration"] = pack_illustration
    sim_args["save_particles"] = save_particles
    sim_args["save_meshes"] = save_meshes
    sim_args["reuse_meshes"] = reuse_meshes
    sim_args["vis_bbox"] = vis_bbox

    return sim_args


def preprocess_simulation_video(
    config_path: str,
    split: Literal['train', 'valid', 'test']='train',
    categories: Optional[List[str]] = None,
    start_idx: Optional[int]=None,
    end_idx: Optional[int]=None,
    ver_identifier: str="default",
    lin_vel: List[float]=[1.0, 0.0, 0.0],
    save_mp4: bool=False,
    pack_illustration: bool=False,
    save_particles: bool=False,
    save_meshes: bool=False,
    reuse_meshes: bool=False,
    vis_bbox: bool=False,
):
    if categories is None:
        categories = os.listdir(join(TARGET_PATH, "data", split))

    for cat in categories:

        cat_ids = os.listdir(join(TARGET_PATH, "data", split, cat))
        start_idx = start_idx if start_idx is not None else 0
        end_idx = end_idx if end_idx is not None else len(cat_ids)
        tracker = tqdm(cat_ids[start_idx:end_idx])

        for idx in tracker:
            shape_id, style_id = idx.split("__")
            tracker.set_postfix_str(f"{split}-{cat}-{idx}")

            sim_dir = join(TARGET_PATH, "simulation_data", split, cat, f'{shape_id}__{style_id}')
            if not os.path.exists(sim_dir):
                continue

            ver_dir = join(TARGET_PATH, VER_DIR, split, cat, f'{shape_id}__{style_id}')
            if (
                os.path.exists(ver_dir) and 
                not reuse_meshes and 
                animation_exists(
                    ver_dir, shape_id, style_id, TEST_CASE, ver_identifier, DEBUG_VIEWS, 
                    ext="mp4" if save_mp4 else "gif"
                )
            ):
                continue
            os.makedirs(ver_dir, exist_ok=True)

            additional_args = get_additional_args_default(line_vel=lin_vel)
            sim_config = prepare_config_for_mesh(
                target_path=TARGET_PATH,
                config=config_path,
                split=split,
                category=cat,
                shape_id=shape_id,
                style_id=style_id,
                output_dir=ver_dir,
                **additional_args
            )

            sim_args = get_sim_args_default(
                save_mp4=save_mp4,
                pack_illustration=pack_illustration,
                save_particles=save_particles,
                save_meshes=save_meshes,
                reuse_meshes=reuse_meshes,
                vis_bbox=vis_bbox
            )
            sim_args["video_name"] = f"{shape_id}__{style_id}-{TEST_CASE}_{ver_identifier}"

            sim_config.update(sim_args)
            sim_config = DictConfig(sim_config)

            OmegaConf.save(sim_config, join(ver_dir, f'{sim_args["video_name"]}.yaml'))

            try:
                subprocess_eval(ver_dir, sim_args, inference_path="inference.py")
            except Exception as e:
                print(f"Error when processing {sim_dir}: {e}")
                with open("throw_error.txt", "a") as f:
                    f.write(f"{split}-{cat}-{idx}-{shape_id}-{style_id}\n{e}\n\n")

                time.sleep(10)


def preprocess_simulation_video_by_style_ids(
    config_path: str,
    split: Literal['train', 'valid', 'test']='train',
    indices: List[int]=None,
    ver_identifier: str="default",
    lin_vel: List[float]=[1.0, 0.0, 0.0],
    save_mp4: bool=False,
    pack_illustration: bool=False,
    save_particles: bool=False,
    save_meshes: bool=False,
    reuse_meshes: bool=False,
    vis_bbox: bool=False,
):
    for i in indices:
        # available_paths = glob(join(TARGET_PATH, "simulation_data", split, "*", f'{i}__*'))
        available_paths = glob(join(TARGET_PATH, "simulation_data", split, "*", f'{i}*'))
        assert len(available_paths) == 1
        sim_dir = available_paths[0]
        shape_id, style_id = os.path.basename(sim_dir).split("__")
        cat = os.path.basename(os.path.dirname(sim_dir))
        print(f"{split}-{cat}-{shape_id}_{style_id}")

        ver_dir = join(TARGET_PATH, VER_DIR, split, cat, f'{shape_id}__{style_id}')
        if (
            os.path.exists(ver_dir) and 
            not reuse_meshes and 
            animation_exists(
                ver_dir, shape_id, style_id, TEST_CASE, ver_identifier, DEBUG_VIEWS, 
                ext="mp4" if save_mp4 else "gif"
            )
        ):
            continue
        os.makedirs(ver_dir, exist_ok=True)

        additional_args = get_additional_args_default(lin_vel=lin_vel)
        sim_config = prepare_config_for_mesh(
            target_path=TARGET_PATH,
            config=config_path,
            split=split,
            category=cat,
            shape_id=shape_id,
            style_id=style_id,
            output_dir=ver_dir,
            **additional_args
        )

        sim_args = get_sim_args_default(
            save_mp4=save_mp4,
            pack_illustration=pack_illustration,
            save_particles=save_particles,
            save_meshes=save_meshes,
            reuse_meshes=reuse_meshes,
            vis_bbox=vis_bbox
        )
        sim_args["video_name"] = f"{shape_id}__{style_id}-{TEST_CASE}_{ver_identifier}"

        sim_config.update(sim_args)
        sim_config = DictConfig(sim_config)

        OmegaConf.save(sim_config, join(ver_dir, f'{sim_args["video_name"]}.yaml'))

        subprocess_eval(ver_dir, sim_args, None, None)
        sys.stdout.flush()

if __name__ == "__main__":
    args = parse_args()

    TARGET_PATH = os.path.join(DATA_PATH, "generated_cache", args.target_name)
    print(f'target_dir: {TARGET_PATH}')

    if args.indices is not None:
        preprocess_simulation_video_by_style_ids(
            config_path=args.config,
            split=args.split,
            indices=args.indices,
            ver_identifier=args.ver_identifier,
            lin_vel=args.lin_vel,
            save_mp4=args.save_mp4,
            pack_illustration=args.pack_illustration,
            save_particles=args.save_particles,
            save_meshes=args.save_meshes,
            reuse_meshes=args.reuse_meshes,
            vis_bbox=args.vis_bbox,
        )
    else:
        preprocess_simulation_video(
            config_path=args.config,
            split=args.split,
            categories=args.categories,
            start_idx=args.shape_start,
            end_idx=args.shape_end,
            ver_identifier=args.ver_identifier,
            lin_vel=args.lin_vel,
            save_mp4=args.save_mp4,
            pack_illustration=args.pack_illustration,
            save_particles=args.save_particles,
            save_meshes=args.save_meshes,
            reuse_meshes=args.reuse_meshes,
            vis_bbox=args.vis_bbox,
        )
