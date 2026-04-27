import os
import sys
import time
import json
import shutil
import trimesh
import argparse
import transformers
import huggingface_hub
from pathlib import Path
from util.constants import DATA_PATH
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.rembg import BackgroundRemover
from hy3dgen.text2image import HunyuanDiTPipeline


TGT = Path(DATA_PATH)


model_path = 'tencent/Hunyuan3D-2'
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled', device='cuda')

huggingface_hub.logging.set_verbosity_error()
transformers.logging.set_verbosity(transformers.logging.CRITICAL)
transformers.logging.disable_progress_bar()


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_dir', '-d', type=str, help='Path to the directory containing meshes to be sampled')
    return parser.parse_args()


def resolve_conditioned_input(mesh_dir: Path):
    info_paths = list(mesh_dir.parent.glob('*info*.json'))
    assert len(info_paths) == 1, f"len(info_paths) = {len(info_paths)}"

    info_path = info_paths[0]
    with open(info_path, 'r') as f:
        info = json.load(f)

    mats_dir = Path(mesh_dir.as_posix().replace('meshes', 'mats'))

    i = 0
    for mesh_path in mesh_dir.glob('*.ply'):
        mesh_name = mesh_path.stem
        ftmp, stmp = mesh_name.split('__')

        model_idx = f'{ftmp.split("_")[2]}_{ftmp.split("_")[3]}'
        info_idx = f'{ftmp.split("_")[2]}_{ftmp.split("_")[3]}__{stmp.split("_")[0]}_{stmp.split("_")[1]}_{stmp.split("_")[2]}'

        if 'image' in mesh_dir.as_posix():
            img_dirs = list((TGT / 'data' / 'test').glob(f"*/{model_idx}__0"))
            assert len(img_dirs) == 1, f"len(img_dirs) = {len(img_dirs)}"

            img_dir = img_dirs[0]
            img_path = img_dir / 'data_static' / info[info_idx]['image_path']
            assert img_path.exists(), f"{img_path} not found"

            # copy image
            sampled_img_path = mats_dir / f'{info_idx}_0' / 'cond.png'
            shutil.copy(img_path, sampled_img_path)
            print(sampled_img_path)
        elif 'text' in mesh_dir.as_posix():
            text = info[info_idx]['text']
            text_path = mats_dir / f'{info_idx}_0' / 'cond.txt'
            with open(text_path, 'w') as f:
                f.write(text + '\n')
        else:
            raise ValueError(f'Unknown data type: {mesh_dir}')

        i += 1

    print(f'Resolvegeometry/output/obj/0116_image/backup/mats-2430/0c_3cf__0_image_6_0d {i} conditioned inputs in {mesh_dir}.')


def log_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        if elapsed_time < 60:
            time_str = f"{elapsed_time:.2f} seconds"
        elif elapsed_time < 3600:
            time_str = f"{elapsed_time / 60:.2f} minutes"
        else:
            time_str = f"{elapsed_time / 3600:.2f} hours"

        print(f"  **TIMING: {func.__name__} took {time_str}")
        return result
    return wrapper


def process_text_condition(text: str):
    image = t2i_worker(text)
    if image.mode == 'RGB':
        rembg = BackgroundRemover()
        image = rembg(image)
    return image


@log_runtime
def process_textured_mesh(mesh_path: Path):
    """
    Process textured mesh
    """
    if not mesh_path.exists():
        print(f'{mesh_path} does not exist.')
        return

    if (
        os.path.exists(mesh_path.parent / (mesh_path.stem + '.obj')) and
        os.path.exists(mesh_path.parent / (mesh_path.stem + '.mtl')) and
        os.path.exists(mesh_path.parent / (mesh_path.stem + '.png'))
    ):
        print(f'Textured already processed for {mesh_path}.')
        return

    assert mesh_path.suffix == '.ply', f"Mesh path {mesh_path} is not a .ply file."

    mesh_cache_dir = mesh_path.parent / (mesh_path.stem)
    mesh_cache_dir.mkdir(parents=True, exist_ok=True)

    print(f'Processing textured mesh for {mesh_cache_dir / mesh_path.name} ...')

    mat_folder = mesh_path.parent.parent / ('mats-' + mesh_path.parent.name.split('-')[-1])
    tmp_0, tmp_1 = mesh_path.stem.split('__')
    cat, idx = tmp_0.split('_')[2], tmp_0.split('_')[3]
    index = f'{cat}_{idx}'
    # index = mesh_path.stem
    mat_dir = list(mat_folder.glob(f'*{index}*'))[0]

    mesh = trimesh.load(mesh_path)
    image_path = mat_dir / 'cond.png'
    if not image_path.exists():
        print(f'{image_path} does not exist. Try text mode ...')
        text_path = mat_dir / 'cond.txt'
        if not text_path.exists():
            print(f'{text_path} does not exist. Skipping ...')
            return
        with open(text_path, 'r') as f:
            text_prompt = f.readlines()[0].strip()
        image = process_text_condition(text_prompt)
        mesh = pipeline_texgen(mesh, image=image)
    else:
        mesh = pipeline_texgen(mesh, image=image_path.as_posix())

    final_mesh_path = mesh_cache_dir / (mesh_path.stem + '.obj')
    mesh.export(final_mesh_path)

    mesh_exists = final_mesh_path.exists()
    mtl_exists = (mesh_cache_dir / 'material.mtl').exists()
    png_exists = (mesh_cache_dir / 'material_0.png').exists()

    print(f'mesh path: {mesh_exists}, mtl path: {mtl_exists}, png path: {png_exists}')

    if mesh_exists and mtl_exists and png_exists:
        # Fix obj name
        fixed_obj = list()
        with open(final_mesh_path, 'r') as f:
            obj_content = f.readlines()
            for line in obj_content:
                if line.startswith('mtllib'):
                    line = f'mtllib {mesh_path.stem}.mtl\n'
                fixed_obj.append(line)
        with open(mesh_path.parent / (mesh_path.stem + '.obj'), 'w') as f:
            f.writelines(fixed_obj)
        
        # Fix mtl name
        fixed_mtl = list()
        with open(mesh_cache_dir / 'material.mtl', 'r') as f:
            mtl_content = f.readlines()
            for line in mtl_content:
                if line.startswith('map_Kd'):
                    line = f'map_Kd {mesh_path.stem}.png\n'
                fixed_mtl.append(line)
        with open(mesh_path.parent / (mesh_path.stem + '.mtl'), 'w') as f:
            f.writelines(fixed_mtl)

        # Fix png name
        os.rename(mesh_cache_dir / 'material_0.png', mesh_path.parent / (mesh_path.stem + '.png'))

        # move to cache dir
        os.rename(mesh_path, mesh_cache_dir / mesh_path.name)
        
        # Clean up
        os.system(f'rm -rf {mesh_cache_dir.as_posix()}')
    else:
        print(f'Failed to generate mesh, mtl or png for {mesh_path}.')
    

def main(mesh_dir: Path):
    """
    Process UV and texture for all meshes in the directory
    """
    # resolve conditioned inputs
    resolve_conditioned_input(mesh_dir)

    # start processing textured meshes
    to_do = list(mesh_dir.glob('*.ply'))
    print(f'Found {len(to_do)} meshes to process in {mesh_dir}.')

    for i, mesh_path in enumerate(to_do):
        print(f'[{i+1}/{len(to_do)}] Processing {mesh_path.name} ...')
        sys.stdout.flush()
        process_textured_mesh(mesh_path)


if __name__ == '__main__':
    args = arg_parser()
    mesh_dir = Path(args.mesh_dir)
    main(mesh_dir)
