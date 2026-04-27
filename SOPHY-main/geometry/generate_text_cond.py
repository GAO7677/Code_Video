import json
import torch
import torch.nn as nn
import random
import mcubes
import trimesh
import argparse
import open_clip
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from safetensors import safe_open
from typing import List

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
    parser.add_argument('--text_model', type=str, default='ViT-L-14')
    parser.add_argument('--text_model_pretrained', '-tmp', type=str, default="dfn2b_s39b")
    args = parser.parse_args()
    print(args)
    return args


class TextEmbedder(nn.Module):
    def __init__(self, model_name: str, model_pretrained: str, force_quick_gelu: bool = False, device: str='cuda'):
        super(TextEmbedder, self).__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=model_pretrained, force_quick_gelu=force_quick_gelu, device=device
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.device = device

    @torch.no_grad()
    def forward(self, text: List[str]) -> torch.Tensor:
        text = self.tokenizer(text).to(self.device)

        cast_dtype = self.model.transformer.get_cast_dtype()

        x = self.model.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.model.positional_embedding.to(cast_dtype)
        x = self.model.transformer(x, attn_mask=self.model.attn_mask)
        x = self.model.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        return x


def get_text_embedder(args, device='cpu'):
    return TextEmbedder(args.text_model, args.text_model_pretrained, device=device)


def eval_text_cond_dm(args):
    dm_path = Path(args.dm_pth)
    ckpt_idx = dm_path.stem.split('-')[-1]
    dm_exp_id = dm_path.parent.name
    note = "" if args.note == "" else f"-{args.note}"
    obj_path = Path(f"output/obj/{dm_exp_id}/{args.cond_path}{note}")
    mat_dir = obj_path / f"mats-{ckpt_idx}"
    mat_dir.mkdir(parents=True, exist_ok=True)
    mesh_dir = obj_path / f"meshes-{ckpt_idx}"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    print(f'Will save files to {obj_path.as_posix()}')

    device = torch.device(args.device)
    text_embedder = get_text_embedder(args) # NOTE: Use cpu

    ae_ckpt = torch.load(args.ae_pth, map_location='cpu')
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

        text_cond_info = prepare_text_features(args)

        print(f'Start sampling ...')
        tracker = tqdm(sorted(list(text_cond_info.keys())))
        for k in tracker:

            # [1, n_ctx, d_model]
            features = text_cond_info[k].pop('features', None)
            if features is None:
                text = text_cond_info[k]['text']
                features = text_embedder([text])

            features = torch.cat([features] * num_samples_pc, dim=0).to(device)
            seeds = torch.arange(kid * num_samples_pc, (kid + 1) * num_samples_pc).to(device)
            text_cond_info[k]['seed'] = kid

            sampled_array = model.sample(cond=features, batch_seeds=seeds).float()

            tracker.set_postfix_str(f'{k}, mean={sampled_array.mean().item():.4f}, std={sampled_array.std().item():.4f}')

            for i in range(sampled_array.shape[0]):

                sub_mat_dir = mat_dir / f'{k}_{i}'
                sub_mat_dir.mkdir(parents=True, exist_ok=True)

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
                    m_file = mesh_dir / f"test_{text_cond_info[k]['category']}_{k}_{i}.ply"
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

        with open(obj_path / f'text_info-{ckpt_idx}.json', 'w') as f:
            json.dump(text_cond_info, f, indent=4)


def prepare_text_features(args):
    print(f'Preparing text features for {args.cond_version} ...')
    print(f'Loading text from {args.cond_path} ...')

    with open("output/sampled-text.json", "r") as f:
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
            # FIXME: currently accepts at most 4096 text prompts (see to_hex_3_chars)
            out[f'xx_{to_hex_3_chars(i)}__0_text_0'] = {'text': l}
    elif args.cond_version == 'eval':
        tracker = tqdm(list((TGT / 'data' / 'test').iterdir()))
        print(f'Using {args.cond_path}_text.safetensors ...')
        for category in tracker:
            tracker.set_postfix_str(category.stem)
            for obj_dir in natsorted(list(category.iterdir())):
                if obj_dir.stem.split("__")[0] in all_sampled_info:
                    cond_signal_path = obj_dir / 'cond_signal' / f'{args.cond_path}_text.safetensors'
                    text_strs = list()
                    with safe_open(cond_signal_path, framework='pt') as f:
                        text_features = f.get_tensor('text')
                        metadata = f.metadata()
                        for i in range(text_features.shape[0]):
                            text_strs.append(metadata[f'text-{i}'])

                    i = np.random.choice([0, 1, 4])
                    out[f'{obj_dir.stem}_text_{i}'] = {
                        'text': text_strs[i],
                        'features': text_features[i:i+1],
                        'category': category_n2i[category.stem]
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

    eval_text_cond_dm(args)
