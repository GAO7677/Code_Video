import json
import torch
import torch.nn as nn
import random
import mcubes
import trimesh
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from safetensors import safe_open
from typing import List, Optional, Literal
from torchvision.transforms import Compose, Resize, InterpolationMode, Normalize

from model import models_cond, models_ae
from util.misc import to_hex_3_chars
from util.constants import DATA_PATH, MAT_COLORS, MMID_COLORS, PART_COLORS, category_n2i

TGT = Path(DATA_PATH)


def arg_parser():
    parser = argparse.ArgumentParser('', add_help=False)
    parser.add_argument('--ae-pth', type=str, required=True) # 'output/ae/kl_d512_m512_l16/checkpoint-199.pth'
    parser.add_argument('--dm-pth', type=str, required=True) # 'output/uncond_dm/kl_d512_m512_l16_edm/checkpoint-999.pth'
    parser.add_argument('--num_samples_per_cond', type=int, default=1)
    parser.add_argument('--cond_version', type=str, required=True, choices=['custom', 'eval'])
    parser.add_argument('--cond_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--note', type=str, default="")
    parser.add_argument('--image_model', type=str, default='dinov2_vitb14_reg')
    args = parser.parse_args()
    print(args)
    return args

class ImageEmbedder(nn.Module):
    def __init__(self, model_name: str, device: str='cuda'):
        super(ImageEmbedder, self).__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', model_name)
        self.model.to(device)
        self.model.eval()
        for _, param in self.model.named_parameters():
            param.requires_grad = False

        self.preprocess = Compose([
            # Resize(self.model.patch_embed.img_size[0], interpolation=InterpolationMode.BICUBIC),
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.device = device

    @staticmethod
    def read_rgba_image(path, background_color: Optional[Literal['white', 'black']]=None):
        image_rgb = Image.open(path)
        if background_color is not None:
            im_data = np.array(image_rgb.convert("RGBA"))
            bg = np.array([1, 1, 1]) if background_color == 'white' else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:,:,3:4] + bg * (1 - norm_data[:,:,3:4])
            image_rgb = Image.fromarray((arr * 255.0).astype(np.uint8), "RGB")
        return image_rgb

    @torch.no_grad()
    def forward(self, image_path: List[str]) -> torch.Tensor:
        images = list()
        for ip in image_path:
            image = self.read_rgba_image(ip, background_color='white')
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.
            images.append(self.preprocess(image))
        x = torch.stack(images, dim=0).to(self.device)
        outs = self.model.forward_features(x)
        rets = torch.cat([
            outs["x_norm_clstoken"].unsqueeze(dim=1),
            outs["x_norm_patchtokens"],
        ], dim=1)
        return rets


def get_image_embedder(args, device='cpu'):
    return ImageEmbedder(args.image_model, device=device)


def eval_image_cond_dm(args):
    dm_path = Path(args.dm_pth)
    ckpt_idx = dm_path.stem.split('-')[-1]
    dm_exp_id = dm_path.parent.name
    note = "" if args.note == "" else f"-{args.note}"
    obj_path = Path(f"/data/gaoya/ckpt/SOPHY/output/obj/{dm_exp_id}/{args.cond_path}{note}")
    mat_dir = obj_path / f"mats-{ckpt_idx}"
    mat_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = obj_path / f"meshes-{ckpt_idx}"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    print(f'Will save files to {obj_path.as_posix()}')

    device = torch.device(args.device)
    image_embedder = get_image_embedder(args) # NOTE: Use cpu

    ae_ckpt = torch.load(args.ae_pth, map_location='cpu', weights_only=False)
    ae_args = ae_ckpt['args']
    args.rgb_embed_dim = ae_args.rgb_embed_dim
    args.part_embed_dim = ae_args.part_embed_dim
    ae = models_ae.__dict__[ae_args.model](
        N=ae_args.point_cloud_size,
        decoder_ff=ae_args.decoder_ff,
        num_mat_classes=ae_args.num_mat_classes,
        mat_decoder_ff=ae_args.mat_decoder_ff,
        mat_layers=ae_args.mat_layers,
        rgb_embed_dim=ae_args.rgb_embed_dim,
        part_embed_dim=ae_args.part_embed_dim,
        mat_embed_dim=ae_args.mat_embed_dim,
        num_part_classes=ae_args.num_part_classes,
        part_loss=ae_args.part_loss,
        mat_start_depth=ae_args.mat_start_depth,
        requires_e=ae_args.requires_e,
        requires_nu=ae_args.requires_nu,
        requires_sigma=ae_args.requires_sigma,
        requires_phi=ae_args.requires_phi,
        requires_rho=ae_args.requires_rho,
        num_mmid=ae_args.num_mmid,
        rgb_layers=ae_args.rgb_layers,
        rgb_decoder_ff=ae_args.rgb_decoder_ff,
    )
    ae.eval()
    print(f"Loading autoencoder from {args.ae_pth} ...")
    ae.load_state_dict(ae_ckpt['model'])
    ae.to(device)

    dm_ckpt = torch.load(args.dm_pth, map_location='cpu')
    dm_args = dm_ckpt['args']

    model = models_cond.__dict__[dm_args.model](dm_args.conditional_signal, dm_args.conditional_arg)
    model.eval()
    print(f"Loading diffusion model from {args.dm_pth} ...")
    model.load_state_dict(dm_ckpt['model'])
    model.to(device)

    density = 128
    gap = 2. / density
    x = np.linspace(-1, 1, density+1)
    y = np.linspace(-1, 1, density+1)
    z = np.linspace(-1, 1, density+1)
    xv, yv, zv = np.meshgrid(x, y, z)
    grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

    kid = 588
    num_samples_pc = args.num_samples_per_cond

    colors = torch.tensor(MAT_COLORS)
    colors = colors[1:]
    mcolors = torch.tensor(MMID_COLORS)
    pcolors = torch.tensor(PART_COLORS)
    part_colors = pcolors[1:]

    with torch.no_grad():

        image_cond_info = prepare_image_features(args)

        print(f'Start sampling ...')
        tracker = tqdm(list(image_cond_info.keys()))
        for k in tracker:

            # [1, n_ctx, d_model]
            features = image_cond_info[k].pop('features', None)
            if features is None:
                ip = image_cond_info[k]['image_path']
                features = image_embedder([ip])

            features = torch.cat([features] * num_samples_pc, dim=0).to(device)
            seeds = torch.arange(kid * num_samples_pc, (kid + 1) * num_samples_pc).to(device)

            sampled_array = model.sample(cond=features, batch_seeds=seeds).float()

            tracker.set_postfix_str(f'{k}, mean={sampled_array.mean().item():.4f}, std={sampled_array.std().item():.4f}')

            for i in range(sampled_array.shape[0]):

                sub_mat_dir = mat_dir / f'{k}_{i}'
                sub_mat_dir.mkdir(parents=True, exist_ok=True)

                torch.save(sampled_array[i:i+1], sub_mat_dir / 'sampled_array.pt')

                with torch.cuda.amp.autocast(enabled=True):
                    outputs = ae.decode(sampled_array[i:i+1], grid, wo_rgb=True)

                occ_outputs = outputs['logits'].cpu() if 'logits' in outputs else None
                mat_outputs = outputs['mat_logits'].cpu() if 'mat_logits' in outputs else None
                part_outputs = outputs['part_logits'].cpu() if 'part_logits' in outputs else None
                mmid_outputs = outputs['mat_mmid_logits'].cpu() if 'mat_mmid_logits' in outputs else None
                E_outputs = outputs['mat_Es'].cpu() if 'mat_Es' in outputs else None
                nu_outputs = outputs['mat_nus'].cpu() if 'mat_nus' in outputs else None
                sigma_outputs = outputs['mat_sigmas'].cpu() if 'mat_sigmas' in outputs else None
                phi_outputs = outputs['mat_phis'].cpu() if 'mat_phis' in outputs else None
                rho_outputs = outputs['mat_rhos'].cpu() if 'mat_rhos' in outputs else None
                # rgb_outputs = outputs['rgb_logits'].cpu() if 'rgb_logits' in outputs else None
                assert 'rgb_logits' not in outputs

                del outputs

                occ_volume = occ_outputs.view(density+1, density+1, density+1).permute(1, 0, 2).numpy()
                verts, faces = mcubes.marching_cubes(occ_volume.astype(np.float32), 0)

                verts *= gap
                verts -= 1

                if ae.num_rgb_layers > 0:
                    verts_ = torch.tensor(verts).float().to(device, non_blocking=True)
                    rgb_verts_outputs = ae.decode_rgb(sampled_array[i:i+1], verts_.unsqueeze(0))
                    rgb_verts_outputs = rgb_verts_outputs.cpu().view(-1, 3)
                    rgb_verts_outputs = rgb_verts_outputs * 0.5 + 0.5
                    rgb_verts_outputs = torch.clamp(rgb_verts_outputs, 0, 1)
                    assert rgb_verts_outputs.shape[0] == verts.shape[0]
                    # NOTE: scale to [-0.5, 0.5]
                    m = trimesh.Trimesh(vertices=verts * 0.5, faces=faces, vertex_colors=rgb_verts_outputs.numpy())
                else:
                    m = trimesh.Trimesh(vertices=verts, faces=faces)

                if args.cond_version == 'eval':
                    m_file = mesh_dir / f"test_{image_cond_info[k]['category']}_{k}_{i}.ply"
                else:
                    m_file = mesh_dir / f"{k}_{i}.ply"
                m.export(m_file.as_posix())

                grid_cpu = grid.cpu()

                points_info = dict()

                threshold = 0

                occ_pred = torch.zeros_like(occ_outputs)
                occ_pred[occ_outputs>=threshold] = 1
                occ_pred = occ_pred.flatten()

                occ_points = grid_cpu.flatten(0, 1)[occ_pred==1]
                # NOTE: scale to [-0.5, 0.5]
                occ_points = occ_points * 0.5
                occ_points = occ_points.numpy()

                if mat_outputs is not None:
                    if ae_args.use_empty_for_mat:
                        mat_outputs = mat_outputs[:, :, 1:]
                    mat_pred = mat_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
                    points_info['point_mat_labels'] = mat_pred.numpy().astype(np.int32)

                    mat_colors = colors[mat_pred]
                    mat_colors = mat_colors.numpy()
                    assert occ_points.shape[0] == mat_pred.shape[0]

                    pcd = trimesh.PointCloud(occ_points, colors=mat_colors)

                    pcd_file = sub_mat_dir / 'mat.ply'
                    pcd.export(pcd_file)

                if part_outputs is not None:
                    if ae_args.use_empty_for_part:
                        part_outputs = part_outputs[:, :, 1:]
                    part_pred = part_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
                    points_info['point_part_labels'] = part_pred.numpy().astype(np.int32)

                    part_colors = pcolors[part_pred]
                    part_colors = part_colors.numpy()
                    assert occ_points.shape[0] == part_pred.shape[0]

                    pcd = trimesh.PointCloud(occ_points, colors=part_colors)

                    pcd_file = sub_mat_dir / 'part.ply'
                    pcd.export(pcd_file)

                if mmid_outputs is not None:
                    mmid_pred = mmid_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
                    points_info['mmid'] = mmid_pred.numpy().astype(np.int32)

                    mmid_colors = mcolors[mmid_pred]
                    mmid_colors = mmid_colors.numpy()
                    assert occ_points.shape[0] == mmid_pred.shape[0]

                    pcd = trimesh.PointCloud(occ_points, colors=mmid_colors)

                    pcd_file = sub_mat_dir / 'mmid.ply'
                    pcd.export(pcd_file)

                if E_outputs is not None:
                    E_pred = E_outputs.flatten(0, 1)[occ_pred==1]
                    points_info['raw_E'] = E_pred.numpy().astype(np.float32)
                    assert occ_points.shape[0] == E_pred.shape[0]

                if nu_outputs is not None:
                    nu_pred = nu_outputs.flatten(0, 1)[occ_pred==1]
                    points_info['raw_nu'] = nu_pred.numpy().astype(np.float32)
                    assert occ_points.shape[0] == nu_pred.shape[0]

                if sigma_outputs is not None:
                    sigma_pred = sigma_outputs.flatten(0, 1)[occ_pred==1]
                    points_info['raw_sigma'] = sigma_pred.numpy().astype(np.float32)
                    assert occ_points.shape[0] == sigma_pred.shape[0]

                if phi_outputs is not None:
                    phi_pred = phi_outputs.flatten(0, 1)[occ_pred==1]
                    points_info['raw_phi'] = phi_pred.numpy().astype(np.float32)
                    assert occ_points.shape[0] == phi_pred.shape[0]

                if rho_outputs is not None:
                    rho_pred = rho_outputs.flatten(0, 1)[occ_pred==1]
                    points_info['raw_rho'] = rho_pred.numpy().astype(np.float32)
                    assert occ_points.shape[0] == rho_pred.shape[0]

                if len(points_info) > 0:
                    pcd = trimesh.PointCloud(occ_points)
                    pcd.export((sub_mat_dir / 'sampled_points.ply').as_posix())
                    if 'point_mat_labels' not in points_info:
                        # placeholder -1
                        points_info['point_mat_labels'] = (np.zeros(occ_points.shape[0]) - 1).astype(np.int32)
                    np.savez_compressed(sub_mat_dir / 'sampled_points_info.npz', **points_info)

            kid += 1

        with open(obj_path / f'image_info-{ckpt_idx}.json', 'w') as f:
            json.dump(image_cond_info, f, indent=4)


def prepare_image_features(args):
    print(f'Preparing image features for {args.cond_version} ...')
    print(f'Loading image from {args.cond_path} ...')

    with open("output/sampled-image.json", "r") as f:
        sampled_info = json.load(f)

    all_sampled_info = set()
    for k, v in sampled_info.items():
        # v: list of strings
        all_sampled_info.update(v)

    out = dict()
    if args.cond_version == 'custom':
        with open(args.cond_path, 'r') as f:
            context = f.readlines()
        for i, line in enumerate(context):
            l = line.strip()
            if l == '': continue
            # FIXME: currently accepts at most 4096 image prompts (see to_hex_3_chars)
            out[f'xx_{to_hex_3_chars(i)}__0_image_0'] = {'image_path': l}
    elif args.cond_version == 'eval':
        tracker = tqdm(list((TGT / 'data' / 'test').iterdir()))
        print(f'Using {args.cond_path}_image.safetensors ...')
        for category in tracker:
            tracker.set_postfix_str(category.stem)
            for obj_dir in natsorted(list(category.iterdir())):
                if obj_dir.stem.split("__")[0] in all_sampled_info:
                    cond_signal_path = obj_dir / 'cond_signal' / f'{args.cond_path}_image.safetensors'
                    with safe_open(cond_signal_path, framework='pt') as f:
                        image_features = f.get_tensor('image')

                    i = np.random.choice([1, 4, 6, 9])
                    out[f'{obj_dir.stem}_image_{i}'] = {
                        'features': image_features[i:i+1],
                        'category': category_n2i[category.stem],
                        'image_path': f's_{i + 30}_000.png'
                    }

    return out

if __name__ == '__main__':
    args = arg_parser()

    random_seed = 2025
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

    eval_image_cond_dm(args)




'''
python geometry/generate_image_cond.py \
    --ae-pth /data/gaoya/ckpt/SOPHY/output/ae/shared/checkpoint-0.pth \
    --dm-pth /data/gaoya/ckpt/SOPHY/output/dm/shared_image/checkpoint-0.pth \
    --num_samples_per_cond 1 \
    --cond_version eval \
    --cond_path v2


'''