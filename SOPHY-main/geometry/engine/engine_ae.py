# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import os
import sys
from typing import Iterable

import torch
import mcubes
import trimesh
import xlsxwriter
import numpy as np
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from util.constants import MAT_COLORS, MMID_COLORS, PART_COLORS
from model.misc import (
    prepare_mat_inputs,
    consistency_loss,
    material_loss,
    material_accuracy,
    material_statistics,
)

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, 
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    criterion_mat: torch.nn.Module = torch.nn.CrossEntropyLoss(),
                    criterion_mmid: torch.nn.Module = torch.nn.CrossEntropyLoss(),
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    mat_weight = args.mat_weight
    mat_emp_weight = args.mat_emp_weight
    kl_weight = args.kl_weight
    part_weight = args.part_weight
    rgb_weight = args.rgb_weight
    half_pc_size = args.point_cloud_size // 2

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (
            points, labels, mats, mat_Es, mat_nus, mat_sigmas, mat_phis, mat_rhos, mat_mmid, parts, rgbs, _,
            surface, surf_rgbs, surf_parts, surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, _, _
        ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mats = mats.to(device, non_blocking=True)
        mat_Es = mat_Es.to(device, non_blocking=True).log10_()              # log10(E)
        mat_nus = mat_nus.to(device, non_blocking=True) * 2.                # 2 * nu
        mat_sigmas = mat_sigmas.to(device, non_blocking=True).log10_()      # log10(sigma)
        mat_phis = mat_phis.to(device, non_blocking=True).deg2rad_()        # rad(phi)
        mat_rhos = mat_rhos.to(device, non_blocking=True).float() * 1e-3    # rho in g/cm^3
        mat_mmid = mat_mmid.to(device, non_blocking=True)
        pb_mats = None
        parts = parts.to(device, non_blocking=True)
        rgbs = rgbs.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        surf_rgbs = surf_rgbs.to(device, non_blocking=True)
        surf_parts = surf_parts.to(device, non_blocking=True)
        surf_mat_Es = surf_mat_Es.to(device, non_blocking=True).log10_()            # log10(E)
        surf_mat_nus = surf_mat_nus.to(device, non_blocking=True) * 2.              # 2 * nu
        surf_mat_sigmas = surf_mat_sigmas.to(device, non_blocking=True).log10_()    # log10(sigma)
        surf_mat_phis = surf_mat_phis.to(device, non_blocking=True).deg2rad_()      # rad(phi)
        surf_mat_rhos = surf_mat_rhos.to(device, non_blocking=True).float() * 1e-3  # rho in g/cm^3
        surf_mat_mmid = surf_mat_mmid.to(device, non_blocking=True).float()
        surf_mat_features = prepare_mat_inputs(surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, args)

        # queries for rgb: concat surface and near points
        queries_for_rgb = torch.cat([surface, points[:, half_pc_size:]], dim=1)
        queries_rgb = torch.cat([surf_rgbs, rgbs[:, half_pc_size:]], dim=1)

        mat_mmid_flat = mat_mmid.reshape(-1)
        mask_sigma = torch.logical_or(mat_mmid_flat == 1, mat_mmid_flat == 2)
        mask_phi = mat_mmid_flat == 3

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(
                surface, points,
                pc_rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                pc_part=surf_parts if args.part_embed_dim > 0 else None,
                pc_mat=surf_mat_features if args.mat_embed_dim > 0 else None,
                queries_for_rgb=queries_for_rgb,    # use surface locations for rgb prediction
            )

            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            labels_vol = labels[:, :half_pc_size]
            labels_near = labels[:, half_pc_size:]

            if 'logits' in outputs:
                occ_outputs = outputs['logits']
                loss_vol = criterion(occ_outputs[:, :half_pc_size], labels_vol)
                loss_near = criterion(occ_outputs[:, half_pc_size:], labels_near)
            else:
                loss_vol = torch.tensor(0.).to(device)
                loss_near = torch.tensor(0.).to(device)

            if 'mat_logits' in outputs:
                mat_outputs = outputs['mat_logits']
                num_mat_classes = mat_outputs.shape[-1]

                mat_outputs_vol = mat_outputs[:, :half_pc_size]                         # (B, half_pc_size, num_mat_classes)
                loss_vol_mat_occ, loss_vol_mat_emp = material_loss(
                    mat_outputs_vol,
                    mats[:, :half_pc_size],
                    labels_vol,
                    use_empty=args.use_empty_for_mat,
                    occ_version=args.mat_loss_occ_version,
                    targets_all_possible=pb_mats[:, :half_pc_size] if args.mat_loss_occ_version != 'cross_entropy' else None,
                )

                mat_outputs_near = mat_outputs[:, half_pc_size:]                        # (B, half_pc_size, num_mat_classes)
                loss_near_mat_occ, loss_near_mat_emp = material_loss(
                    mat_outputs_near,
                    mats[:, half_pc_size:],
                    labels_near,
                    use_empty=args.use_empty_for_mat,
                    occ_version=args.mat_loss_occ_version,
                    targets_all_possible=pb_mats[:, half_pc_size:] if args.mat_loss_occ_version != 'cross_entropy' else None,
                )

                loss_vol_mat = (loss_vol_mat_occ + mat_emp_weight * loss_vol_mat_emp) / (1. + mat_emp_weight)
                loss_near_mat = (loss_near_mat_occ + mat_emp_weight * loss_near_mat_emp) / (1. + mat_emp_weight)

            if 'mat_Es' in outputs:
                mat_E_outputs = outputs['mat_Es'].squeeze(-1)
                loss_mat_E = F.l1_loss(mat_E_outputs[labels==1], mat_Es[labels==1], reduction='mean')
            else:
                loss_mat_E = None

            if 'mat_nus' in outputs:
                mat_nu_outputs = outputs['mat_nus'].squeeze(-1)
                loss_mat_nu = F.l1_loss(mat_nu_outputs[labels==1], mat_nus[labels==1], reduction='mean')
            else:
                loss_mat_nu = None
            
            if 'mat_sigmas' in outputs:
                if mask_sigma.sum() > 0:
                    mat_sigma_outputs = outputs['mat_sigmas'].squeeze(-1)   # (B, N)
                    mat_sigma_outputs_flat = mat_sigma_outputs.reshape(-1)
                    loss_mat_sigma = F.l1_loss(mat_sigma_outputs_flat[mask_sigma], mat_sigmas.reshape(-1)[mask_sigma], reduction='mean')
                else:
                    loss_mat_sigma = F.l1_loss(outputs['mat_sigmas'], outputs['mat_sigmas'], reduction='mean')  # dummy loss
            else:
                loss_mat_sigma = None

            if 'mat_phis' in outputs:
                if mask_phi.sum() > 0:
                    mat_phi_outputs = outputs['mat_phis'].squeeze(-1)   # (B, N)
                    mat_phi_outputs_flat = mat_phi_outputs.reshape(-1)
                    loss_mat_phi = F.l1_loss(mat_phi_outputs_flat[mask_phi], mat_phis.reshape(-1)[mask_phi], reduction='mean')
                else:
                    loss_mat_phi = F.l1_loss(outputs['mat_phis'], outputs['mat_phis'], reduction='mean')        # dummy loss
            else:
                loss_mat_phi = None

            if 'mat_rhos' in outputs:
                mat_rho_outputs = outputs['mat_rhos'].squeeze(-1)
                loss_mat_rho = F.l1_loss(mat_rho_outputs[labels==1], mat_rhos[labels==1], reduction='mean')
            else:
                loss_mat_rho = None

            if 'mat_mmid_logits' in outputs:
                mat_mmid_outputs = outputs['mat_mmid_logits']
                num_mmid_classes = mat_mmid_outputs.shape[-1]

                mat_mmid_outputs = mat_mmid_outputs.reshape(-1, num_mmid_classes)
                loss_mat_mmid = criterion_mmid(mat_mmid_outputs[labels.reshape(-1)==1], mat_mmid.reshape(-1)[labels.reshape(-1)==1])
            else:
                loss_mat_mmid = None

            if 'part_logits' in outputs:
                part_outputs = outputs['part_logits']
                num_part_classes = part_outputs.shape[-1]

                part_outputs_vol = part_outputs[:, :half_pc_size]                       # (B, half_pc_size, num_part_classes)
                part_outputs_vol = part_outputs_vol.reshape(-1, num_part_classes)       # (B*half_pc_size, num_part_classes)
                part_outputs_vol_occ = part_outputs_vol
                loss_vol_part = criterion_mat(part_outputs_vol_occ, parts[:, :half_pc_size].reshape(-1))

                part_outputs_near = part_outputs[:, half_pc_size:]                      # (B, half_pc_size, num_part_classes)
                part_outputs_near = part_outputs_near.reshape(-1, num_part_classes)     # (B*half_pc_size, num_part_classes)
                part_outputs_near_occ = part_outputs_near
                loss_near_part = criterion_mat(part_outputs_near_occ, parts[:, half_pc_size:].reshape(-1))

            if args.part_loss:
                loss_part = 0.

                mat_latents = outputs['mat_latents']

                for batch_mat_latents, batch_parts, batch_labels in zip(mat_latents, parts, labels):
                    loss_part += consistency_loss(batch_mat_latents[batch_labels==1], batch_parts[batch_labels==1], normalization=True)

                loss_part /= len(parts)

                loss_surf_part = 0.

                surf_mat_latents = outputs.get('surf_mat_latents')
                if surf_mat_latents is None:
                    loss_surf_part = None
                else:
                    surf_mat_idx = outputs['surf_mat_idx']                          # downsampled indices
                    surf_parts_sampled = surf_parts.view(-1, 1)[surf_mat_idx]
                    surf_parts_sampled = surf_parts_sampled.view(surf_parts.shape[0], -1)
                    for batch_mat_latents, batch_surf_parts in zip(surf_mat_latents, surf_parts_sampled):
                        loss_surf_part += consistency_loss(batch_mat_latents, batch_surf_parts, normalization=True)

                    loss_surf_part /= len(surf_parts_sampled)
            else:
                loss_part = None
                loss_surf_part = None

            if 'rgb_logits' in outputs:
                rgb_outputs = outputs['rgb_logits']
                loss_rgb = F.l1_loss(rgb_outputs, queries_rgb, reduction='mean')
            else:
                loss_rgb = None

            loss = loss_vol + 0.1 * loss_near

            if 'mat_logits' in outputs:
                loss += mat_weight * (loss_vol_mat + 0.1 * loss_near_mat)
            if loss_mat_E is not None:
                loss += mat_weight * 0.1 * loss_mat_E
            if loss_mat_nu is not None:
                loss += mat_weight * loss_mat_nu
            if loss_mat_sigma is not None:
                loss += mat_weight * 0.1 * loss_mat_sigma
            if loss_mat_phi is not None:
                loss += mat_weight * loss_mat_phi
            if loss_mat_rho is not None:
                loss += mat_weight * loss_mat_rho
            if loss_mat_mmid is not None:
                loss += mat_weight * loss_mat_mmid

            if 'part_logits' in outputs:
                loss += part_weight * (loss_vol_part + 0.1 * loss_near_part)

            if loss_kl is not None:
                loss += kl_weight * loss_kl

            if loss_part is not None:
                loss += part_weight * loss_part

            if loss_surf_part is not None:
                loss += part_weight * 0.1 * loss_surf_part

            if loss_rgb is not None:
                loss += rgb_weight * loss_rgb

        loss_value = loss.item()

        threshold = 0

        if 'logits' in outputs:
            pred = torch.zeros_like(occ_outputs[:, :half_pc_size])
            pred[occ_outputs[:, :half_pc_size]>=threshold] = 1

            accuracy = (pred==labels[:, :half_pc_size]).float().sum(dim=1) / labels[:, :half_pc_size].shape[1]
            accuracy = accuracy.mean()
            intersection = (pred * labels[:, :half_pc_size]).sum(dim=1)
            union = (pred + labels[:, :half_pc_size]).gt(0).sum(dim=1) + 1e-5
            iou = intersection * 1.0 / union
            iou = iou.mean()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(f"loss_vol: {loss_vol.item()}, loss_near: {loss_near.item()}")
            if 'mat_logits' in outputs:
                print(f"loss_vol_mat: {loss_vol_mat.item()}, loss_near_mat: {loss_near_mat.item()}")
            if loss_mat_E is not None:
                print(f"loss_mat_E: {loss_mat_E.item()}")
            if loss_mat_nu is not None:
                print(f"loss_mat_nu: {loss_mat_nu.item()}")
            if loss_mat_sigma is not None:
                print(f"loss_mat_sigma: {loss_mat_sigma.item()}")
            if loss_mat_phi is not None:
                print(f"loss_mat_phi: {loss_mat_phi.item()}")
            if loss_mat_rho is not None:
                print(f"loss_mat_rho: {loss_mat_rho.item()}")
            if loss_mat_mmid is not None:
                print(f"loss_mat_mmid: {loss_mat_mmid.item()}")
            print(f"loss_kl: {loss_kl.item()}")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        metric_logger.update(loss_vol=loss_vol.item())
        metric_logger.update(loss_near=loss_near.item())

        if 'logits' in outputs:
            metric_logger.update(iou=iou.item())

        if 'mat_logits' in outputs:
            metric_logger.update(loss_vol_mat=loss_vol_mat.item())
            metric_logger.update(loss_near_mat=loss_near_mat.item())

        if loss_mat_E is not None:
            metric_logger.update(loss_mat_E=loss_mat_E.item())
        if loss_mat_nu is not None:
            metric_logger.update(loss_mat_nu=loss_mat_nu.item())
        if loss_mat_sigma is not None:
            metric_logger.update(loss_mat_sigma=loss_mat_sigma.item())
        if loss_mat_phi is not None:
            metric_logger.update(loss_mat_phi=loss_mat_phi.item())
        if loss_mat_rho is not None:
            metric_logger.update(loss_mat_rho=loss_mat_rho.item())
        if loss_mat_mmid is not None:
            metric_logger.update(loss_mat_mmid=loss_mat_mmid.item())

        if 'part_logits' in outputs:
            metric_logger.update(loss_vol_part=loss_vol_part.item())
            metric_logger.update(loss_near_part=loss_near_part.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

        if loss_part is not None:
            metric_logger.update(loss_part=loss_part.item())

        if loss_surf_part is not None:
            metric_logger.update(loss_surf_part=loss_surf_part.item())

        if loss_rgb is not None:
            metric_logger.update(loss_rgb=loss_rgb.item())

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_vol_reduce = misc.all_reduce_mean(loss_vol.item())
        loss_near_reduce = misc.all_reduce_mean(loss_near.item())
        if 'mat_logits' in outputs:
            loss_vol_mat_reduce = misc.all_reduce_mean(loss_vol_mat.item())
            loss_near_mat_reduce = misc.all_reduce_mean(loss_near_mat.item())
        if loss_mat_E is not None:
            loss_mat_E_reduce = misc.all_reduce_mean(loss_mat_E.item())
        if loss_mat_nu is not None:
            loss_mat_nu_reduce = misc.all_reduce_mean(loss_mat_nu.item())
        if loss_mat_sigma is not None:
            loss_mat_sigma_reduce = misc.all_reduce_mean(loss_mat_sigma.item())
        if loss_mat_phi is not None:
            loss_mat_phi_reduce = misc.all_reduce_mean(loss_mat_phi.item())
        if loss_mat_rho is not None:
            loss_mat_rho_reduce = misc.all_reduce_mean(loss_mat_rho.item())
        if loss_mat_mmid is not None:
            loss_mat_mmid_reduce = misc.all_reduce_mean(loss_mat_mmid.item())
        if 'part_logits' in outputs:
            loss_vol_part_reduce = misc.all_reduce_mean(loss_vol_part.item())
            loss_near_part_reduce = misc.all_reduce_mean(loss_near_part.item())
        if loss_kl is not None:
            loss_kl_reduce = misc.all_reduce_mean(loss_kl.item())
        if loss_part is not None:
            loss_part_reduce = misc.all_reduce_mean(loss_part.item())
        if loss_surf_part is not None:
            loss_surf_part_reduce = misc.all_reduce_mean(loss_surf_part.item())
        if loss_rgb is not None:
            loss_rgb_reduce = misc.all_reduce_mean(loss_rgb.item())
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/loss_vol', loss_vol_reduce, epoch_1000x)
            log_writer.add_scalar('train/loss_near', loss_near_reduce, epoch_1000x)
            if 'mat_logits' in outputs:
                log_writer.add_scalar('train/loss_vol_mat', loss_vol_mat_reduce, epoch_1000x)
                log_writer.add_scalar('train/loss_near_mat', loss_near_mat_reduce, epoch_1000x)
            if loss_mat_E is not None:
                log_writer.add_scalar('train/loss_mat_E', loss_mat_E_reduce, epoch_1000x)
            if loss_mat_nu is not None:
                log_writer.add_scalar('train/loss_mat_nu', loss_mat_nu_reduce, epoch_1000x)
            if loss_mat_sigma is not None:
                log_writer.add_scalar('train/loss_mat_sigma', loss_mat_sigma_reduce, epoch_1000x)
            if loss_mat_phi is not None:
                log_writer.add_scalar('train/loss_mat_phi', loss_mat_phi_reduce, epoch_1000x)
            if loss_mat_rho is not None:
                log_writer.add_scalar('train/loss_mat_rho', loss_mat_rho_reduce, epoch_1000x)
            if loss_mat_mmid is not None:
                log_writer.add_scalar('train/loss_mat_mmid', loss_mat_mmid_reduce, epoch_1000x)
            if 'part_logits' in outputs:
                log_writer.add_scalar('train/loss_vol_part', loss_vol_part_reduce, epoch_1000x)
                log_writer.add_scalar('train/loss_near_part', loss_near_part_reduce, epoch_1000x)
            if loss_kl is not None:
                log_writer.add_scalar('train/loss_kl', loss_kl_reduce, epoch_1000x)
            if loss_part is not None:
                log_writer.add_scalar('train/loss_part', loss_part_reduce, epoch_1000x)
            if loss_surf_part is not None:
                log_writer.add_scalar('train/loss_surf_part', loss_surf_part_reduce, epoch_1000x)
            if loss_rgb is not None:
                log_writer.add_scalar('train/loss_rgb', loss_rgb_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, header='Test:'):
    criterion = torch.nn.BCEWithLogitsLoss()
    criterion_mat = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")

    if args.eval_save_xlsx:
        ckpt_idx = args.resume.split('/')[-1].split('-')[-1].split('.')[0]
        result_file = os.path.join(args.output_dir, f'valid-{ckpt_idx}-v2.xlsx')
        result_wb = xlsxwriter.Workbook(result_file)
        result_ws = result_wb.add_worksheet()
        result_ws.write(0, 0, 'category')
        result_ws.write(0, 1, 'idx')
        result_ws.write(0, 2, 'iou')
        result_ws.write(0, 3, 'acc_mat')
        row = 1

    # switch to evaluation mode
    model.eval()
    occ_model = None
    metrics = dict()
    mat_i2n = data_loader.dataset.mat_i2n
    cat_i2n = data_loader.dataset.category_i2n

    mat_logits_list = list()
    mat_targets_list = list()

    for points, labels, mats, mat_Es, mat_nus, mat_sigmas, mat_phis, mat_rhos, mat_mmid, parts, rgbs, _, \
        surface, surf_rgbs, surf_parts, surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, c, idx in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        mats = mats.to(device, non_blocking=True)
        mat_Es = mat_Es.to(device, non_blocking=True).log10_()              # log10(E)
        mat_nus = mat_nus.to(device, non_blocking=True) * 2.                # 2 * nu
        mat_sigmas = mat_sigmas.to(device, non_blocking=True).log10_()      # log10(sigma)
        mat_phis = mat_phis.to(device, non_blocking=True).deg2rad_()        # rad(phi)
        mat_rhos = mat_rhos.to(device, non_blocking=True).float() * 1e-3    # rho in g/cm^3
        mat_mmid = mat_mmid.to(device, non_blocking=True)
        pb_mats = None
        parts = parts.to(device, non_blocking=True)
        rgbs = rgbs.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        surf_rgbs = surf_rgbs.to(device, non_blocking=True)
        surf_parts = surf_parts.to(device, non_blocking=True)
        surf_mat_Es = surf_mat_Es.to(device, non_blocking=True).log10_()            # log10(E)
        surf_mat_nus = surf_mat_nus.to(device, non_blocking=True) * 2.              # 2 * nu
        surf_mat_sigmas = surf_mat_sigmas.to(device, non_blocking=True).log10_()    # log10(sigma)
        surf_mat_phis = surf_mat_phis.to(device, non_blocking=True).deg2rad_()      # rad(phi)
        surf_mat_rhos = surf_mat_rhos.to(device, non_blocking=True).float() * 1e-3  # rho in g/cm^3
        surf_mat_mmid = surf_mat_mmid.to(device, non_blocking=True).float()
        surf_mat_features = prepare_mat_inputs(surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, args)

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            outputs = model(
                surface, points,
                pc_rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                pc_part=surf_parts if args.part_embed_dim > 0 else None,
                pc_mat=surf_mat_features if args.mat_embed_dim > 0 else None,
                queries_for_rgb=surface,    # use surface locations for rgb prediction
            )
            if 'kl' in outputs:
                loss_kl = outputs['kl']
                loss_kl = torch.sum(loss_kl) / loss_kl.shape[0]
            else:
                loss_kl = None

            occ_outputs = outputs['logits'] if 'logits' in outputs else None

            loss = criterion(occ_outputs, labels) if occ_outputs is not None else torch.tensor(0.).to(device)

            if 'mat_logits' in outputs:
                mat_outputs = outputs['mat_logits']
                num_mat_classes = mat_outputs.shape[-1]
                loss_mat_occ, loss_mat_emp = material_loss(
                    mat_outputs,
                    mats,
                    labels,
                    use_empty=args.use_empty_for_mat,
                    occ_version=args.mat_loss_occ_version,
                    targets_all_possible=pb_mats if args.mat_loss_occ_version != 'cross_entropy' else None,
                )

            if 'mat_Es' in outputs:
                mat_E_outputs = outputs['mat_Es'].squeeze(-1)
                loss_mat_E = F.l1_loss(mat_E_outputs[labels==1], mat_Es[labels==1], reduction='mean')

            if 'mat_nus' in outputs:
                mat_nu_outputs = outputs['mat_nus'].squeeze(-1)
                loss_mat_nu = F.l1_loss(mat_nu_outputs[labels==1], mat_nus[labels==1], reduction='mean')

            if 'mat_sigmas' in outputs:
                mat_mmid_flat = mat_mmid.reshape(-1)
                mask = torch.logical_or(mat_mmid_flat == 1, torch.logical_or(mat_mmid_flat == 4, torch.logical_or(mat_mmid_flat == 6, mat_mmid_flat == 7)))
                if mask.sum() > 0:
                    mat_sigma_outputs = outputs['mat_sigmas'].squeeze(-1)
                    mat_sigma_outputs_flat = mat_sigma_outputs.reshape(-1)
                    loss_mat_sigma = F.l1_loss(mat_sigma_outputs_flat[mask], mat_sigmas.reshape(-1)[mask], reduction='mean')
                else:
                    loss_mat_sigma = torch.zeros(1).to(device)

            if 'mat_phis' in outputs:
                mat_mmid_flat = mat_mmid.reshape(-1)
                mask = torch.logical_or(mat_mmid_flat == 2, mat_mmid_flat == 5)
                if mask.sum() > 0:
                    mat_phi_outputs = outputs['mat_phis'].squeeze(-1)
                    mat_phi_outputs_flat = mat_phi_outputs.reshape(-1)
                    loss_mat_phi = F.l1_loss(mat_phi_outputs_flat[mask], mat_phis.reshape(-1)[mask], reduction='mean')
                else:
                    loss_mat_phi = torch.zeros(1).to(device)

            if 'mat_rhos' in outputs:
                mat_rho_outputs = outputs['mat_rhos'].squeeze(-1)
                loss_mat_rho = F.l1_loss(mat_rho_outputs[labels==1], mat_rhos[labels==1], reduction='mean')

            if 'mat_mmid_logits' in outputs:
                mat_mmid_outputs = outputs['mat_mmid_logits']
                num_mmid_classes = mat_mmid_outputs.shape[-1]
                loss_mat_mmid = criterion_mat(mat_mmid_outputs.reshape(-1, num_mmid_classes)[labels.reshape(-1)==1], mat_mmid.reshape(-1)[labels.reshape(-1)==1])

            if 'part_logits' in outputs:
                part_outputs = outputs['part_logits']
                num_part_classes = part_outputs.shape[-1]
                loss_part = criterion_mat(part_outputs.reshape(-1, num_part_classes)[labels.reshape(-1)==1], parts.reshape(-1)[labels.reshape(-1)==1])

            if 'rgb_logits' in outputs:
                rgb_outputs = outputs['rgb_logits']
                loss_rgb = F.l1_loss(rgb_outputs, surf_rgbs, reduction='mean')

        threshold = 0
        batch_size = points.shape[0]

        if 'logits' in outputs:
            pred = torch.zeros_like(occ_outputs)
            pred[occ_outputs>=threshold] = 1

            accuracy = (pred==labels).float().sum(dim=1) / labels.shape[1]
            accuracy = accuracy.mean()
            intersection = (pred * labels).sum(dim=1)
            union = (pred + labels).gt(0).sum(dim=1)
            iou = intersection * 1.0 / union + 1e-5

            # category_iou
            category_iou = dict()
            for i in torch.unique(c):
                category_iou[i.item()] = iou[c==i].mean()

            iou = iou.mean()

        if 'mat_logits' in outputs:
            if args.use_empty_for_mat:
                post_mat_outputs = mat_outputs[:, :, 1:]
                tgt_mats = pb_mats[:, :, 1:]
            else:
                post_mat_outputs = mat_outputs
                tgt_mats = pb_mats
            pred_mats = post_mat_outputs.argmax(dim=-1)
            post_num_mat_classes = num_mat_classes - 1 if args.use_empty_for_mat else num_mat_classes
            accuracy_mat = material_accuracy(
                F.one_hot(pred_mats, num_classes=num_mat_classes - 1 if args.use_empty_for_mat else num_mat_classes),
                tgt_mats,
                labels, reduction='none'
            )

            # category_acc_mat
            category_acc_mat = dict()
            for i in torch.unique(c):
                category_acc_mat[i.item()] = accuracy_mat[c==i].mean()

            accuracy_mat = accuracy_mat.mean()

        if 'mat_mmid_logits' in outputs:
            pred_mmid = mat_mmid_outputs.argmax(dim=-1)
            accuracy_mmid = torch.logical_and(pred_mmid==mat_mmid, labels==1).float().sum(dim=1) / labels.sum(dim=1)
            
            # category_acc_mmid
            category_acc_mmid = dict()
            for i in torch.unique(c):
                category_acc_mmid[i.item()] = accuracy_mmid[c==i].mean()
            
            accuracy_mmid = accuracy_mmid.mean()

        if 'part_logits' in outputs:
            # only calculate accuracy for occupied space
            if args.use_empty_for_part:
                pred_parts = part_outputs[:, :, 1:].argmax(dim=-1)
                parts -= 1
            else:
                pred_parts = part_outputs.argmax(dim=-1)
            accuracy_part = torch.logical_and(pred_parts==parts, labels==1).float().sum(dim=1) / labels.sum(dim=1)

            # category_acc_part
            category_acc_part = dict()
            for i in torch.unique(c):
                category_acc_part[i.item()] = accuracy_part[c==i].mean()

            accuracy_part = accuracy_part.mean()

        if args.eval_save_xlsx:
            result_ws.write(row, 0, c.item())
            result_ws.write(row, 1, idx[0] if isinstance(idx, tuple) else idx.item())
            if 'logits' in outputs:
                result_ws.write(row, 2, iou.item())
            if 'mat_logits' in outputs:
                result_ws.write(row, 3, accuracy_mat.item())
            row += 1

        if 'logits' in outputs:
            metric_logger.update(loss=loss.item())
            metric_logger.meters['iou'].update(iou.item(), n=batch_size)
            for k, v in category_iou.items():
                kk = cat_i2n[k]
                metric_logger.meters[f'category_iou_{kk}'].update(v.item(), n=(c==k).sum().item())

        if 'mat_logits' in outputs:
            metric_logger.update(loss_mat=loss_mat_occ.item())      # NOTE: only consider occupied space
            metric_logger.meters['accuracy_mat'].update(accuracy_mat.item(), n=batch_size)
            for k, v in category_acc_mat.items():
                kk = cat_i2n[k]
                metric_logger.meters[f'category_acc_mat_{kk}'].update(v.item(), n=(c==k).sum().item())

        if 'mat_Es' in outputs:
            metric_logger.update(loss_mat_E=loss_mat_E.item())
        if 'mat_nus' in outputs:
            metric_logger.update(loss_mat_nu=loss_mat_nu.item())
        if 'mat_sigmas' in outputs:
            metric_logger.update(loss_mat_sigma=loss_mat_sigma.item())
        if 'mat_phis' in outputs:
            metric_logger.update(loss_mat_phi=loss_mat_phi.item())
        if 'mat_rhos' in outputs:
            metric_logger.update(loss_mat_rho=loss_mat_rho.item())
        if 'mat_mmid_logits' in outputs:
            metric_logger.update(loss_mat_mmid=loss_mat_mmid.item())
            metric_logger.meters['accuracy_mmid'].update(accuracy_mmid.item(), n=batch_size)
            for k, v in category_acc_mmid.items():
                kk = cat_i2n[k]
                metric_logger.meters[f'category_acc_mmid_{kk}'].update(v.item(), n=(c==k).sum().item())

        if 'part_logits' in outputs:
            metric_logger.update(loss_part=loss_part.item())
            metric_logger.meters['accuracy_part'].update(accuracy_part.item(), n=batch_size)
            for k, v in category_acc_part.items():
                kk = cat_i2n[k]
                metric_logger.meters[f'category_acc_part_{kk}'].update(v.item(), n=(c==k).sum().item())

        if 'rgb_logits' in outputs:
            metric_logger.update(loss_rgb=loss_rgb.item())

        if loss_kl is not None:
            metric_logger.update(loss_kl=loss_kl.item())

    if args.eval_save_xlsx:
        result_wb.close()

    if len(mat_logits_list) > 0:
        mat_targets = np.concatenate(mat_targets_list, axis=0)
        mat_logits = np.concatenate(mat_logits_list, axis=0)

        for i in range(mat_targets.shape[1]):
            if mat_targets[:, i].sum() == 0:
                continue                        # skip when not presented
            ii = i + 1 if args.use_empty_for_mat else i
            threshold = getattr(args, f'mat_thre_{mat_i2n[ii]}', 'auto')
            sub_metrics = material_statistics(mat_targets[:, i], mat_logits[:, i], threshold=threshold)
            for k, v in sub_metrics.items():
                if k in ['accuracy', 'precision', 'recall', 'thre']:
                    metrics[f'mat_{k}_{mat_i2n[ii]}'] = v

    avg_precision = 0.
    avg_recall = 0.

    for k, v in metrics.items():
        if 'precision' in k:
            avg_precision += v
        if 'recall' in k:
            avg_recall += v

    if avg_precision > 0:
        avg_precision /= len([k for k in metrics.keys() if 'precision' in k])
    if avg_recall > 0:
        avg_recall /= len([k for k in metrics.keys() if 'recall' in k])

    metrics['mat_precision'] = avg_precision
    metrics['mat_recall'] = avg_recall

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    log = "* "
    if 'logits' in outputs:
        log += f"iou {metric_logger.iou.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f} "
    if 'mat_logits' in outputs:
        log += f"accuracy_mat {metric_logger.accuracy_mat.global_avg:.3f} "
        if 'mat_mmid_logits' in outputs:
            log += f"accuracy_mmid {metric_logger.accuracy_mmid.global_avg:.3f} "
    if 'part_logits' in outputs:
        log += f"accuracy_part {metric_logger.accuracy_part.global_avg:.3f} "
    print(log)

    metrics.update({k: meter.global_avg for k, meter in metric_logger.meters.items()})

    return metrics


@torch.no_grad()
def evaluate_mesh(data_loader, model, device, args):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test Mesh:'

    ckpt_idx = args.resume.split('/')[-1].split('-')[-1].split('.')[0]

    mesh_dir = os.path.join(args.output_dir, f'meshes-{ckpt_idx}')
    os.makedirs(mesh_dir, exist_ok=True)
    mat_dir = os.path.join(args.output_dir, f'mats-{ckpt_idx}')
    os.makedirs(mat_dir, exist_ok=True)
    colors = torch.tensor(MAT_COLORS)
    # if not args.use_empty_for_mat:
    colors = colors[1:]
    mcolors = torch.tensor(MMID_COLORS)
    pcolors = torch.tensor(PART_COLORS)
    part_colors = pcolors[1:]

    # switch to evaluation mode
    model.eval()

    for _, labels, _, _, _, _, _, _, _, _, _, _, surface, surf_rgbs, surf_parts, \
        surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, c, idx in metric_logger.log_every(data_loader, 50, header):

        labels = labels.to(device, non_blocking=True)
        surface = surface.to(device, non_blocking=True)
        surf_rgbs = surf_rgbs.to(device, non_blocking=True)
        surf_parts = surf_parts.to(device, non_blocking=True)
        surf_mat_Es = surf_mat_Es.to(device, non_blocking=True).log10_()            # log10(E)
        surf_mat_nus = surf_mat_nus.to(device, non_blocking=True) * 2.              # 2 * nu
        surf_mat_sigmas = surf_mat_sigmas.to(device, non_blocking=True).log10_()    # log10(sigma)
        surf_mat_phis = surf_mat_phis.to(device, non_blocking=True).deg2rad_()      # rad(phi)
        surf_mat_rhos = surf_mat_rhos.to(device, non_blocking=True).float() * 1e-3  # rho in g/cm^3
        surf_mat_mmid = surf_mat_mmid.to(device, non_blocking=True).float()
        surf_mat_features = prepare_mat_inputs(surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, args)

        density = 128
        gap = 2. / density
        x = np.linspace(-1, 1, density+1)
        y = np.linspace(-1, 1, density+1)
        z = np.linspace(-1, 1, density+1)
        xv, yv, zv = np.meshgrid(x, y, z)
        grid = torch.from_numpy(np.stack([xv, yv, zv]).astype(np.float32)).view(3, -1).transpose(0, 1)[None].to(device, non_blocking=True)

        sub_mat_dir = os.path.join(mat_dir, f'{idx[0] if isinstance(idx, tuple) else idx.item()}')
        os.makedirs(sub_mat_dir, exist_ok=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(
                surface, grid,
                pc_rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                pc_part=surf_parts if args.part_embed_dim > 0 else None,
                pc_mat=surf_mat_features if args.mat_embed_dim > 0 else None,
            )

            occ_outputs = outputs['logits'].detach().cpu() if 'logits' in outputs else None
            mat_outputs = outputs['mat_logits'].detach().cpu() if 'mat_logits' in outputs else None
            part_outputs = outputs['part_logits'].detach().cpu() if 'part_logits' in outputs else None
            mmid_outputs = outputs['mat_mmid_logits'].detach().cpu() if 'mat_mmid_logits' in outputs else None
            E_outputs = outputs['mat_Es'].detach().cpu() if 'mat_Es' in outputs else None
            nu_outputs = outputs['mat_nus'].detach().cpu() if 'mat_nus' in outputs else None
            sigma_outputs = outputs['mat_sigmas'].detach().cpu() if 'mat_sigmas' in outputs else None
            phi_outputs = outputs['mat_phis'].detach().cpu() if 'mat_phis' in outputs else None
            rho_outputs = outputs['mat_rhos'].detach().cpu() if 'mat_rhos' in outputs else None
            rgb_outputs = outputs['rgb_logits'].detach().cpu() if 'rgb_logits' in outputs else None

            del outputs

        occ_volume = occ_outputs.view(density+1, density+1, density+1).permute(1, 0, 2).numpy()
        verts, faces = mcubes.marching_cubes(occ_volume.astype(np.float32), 0)

        verts *= gap
        verts -= 1

        if rgb_outputs is not None:
            # the model outputs rgb
            verts_ = torch.tensor(verts).float().to(device, non_blocking=True)
            outputs_tmp = model(
                surface, verts_.unsqueeze(0),
                pc_rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                pc_part=surf_parts if args.part_embed_dim > 0 else None,
                pc_mat=surf_mat_features if args.mat_embed_dim > 0 else None,
            )
            rgb_verts_outputs = outputs_tmp['rgb_logits'].detach().cpu()
            rgb_verts_outputs = rgb_verts_outputs.view(-1, 3)
            rgb_verts_outputs = rgb_verts_outputs * 0.5 + 0.5
            rgb_verts_outputs = torch.clamp(rgb_verts_outputs, 0, 1)
            assert rgb_verts_outputs.shape[0] == verts.shape[0]
            # NOTE: scale to [-0.5, 0.5]
            m = trimesh.Trimesh(vertices=verts * 0.5, faces=faces, vertex_colors=rgb_verts_outputs.numpy())

            del outputs_tmp
        else:
            m = trimesh.Trimesh(vertices=verts * 0.5, faces=faces)

        # if idx is tuple, it is a length-1 'str' tuple
        m_file = os.path.join(mesh_dir, f'test_{c.item()}_{idx[0] if isinstance(idx, tuple) else idx.item()}.ply')
        m.export(m_file)

        grid = grid.cpu()

        points_info = dict()

        threshold = 0

        occ_pred = torch.zeros_like(occ_outputs)
        occ_pred[occ_outputs>=threshold] = 1
        occ_pred = occ_pred.flatten()

        occ_points = grid.flatten(0, 1)[occ_pred==1]
        # NOTE: scale to [-0.5, 0.5]
        occ_points = occ_points * 0.5
        occ_points = occ_points.numpy()

        if mat_outputs is not None:
            if args.use_empty_for_mat:
                mat_outputs = mat_outputs[:, :, 1:]
            mat_pred = mat_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
            points_info['point_mat_labels'] = mat_pred.numpy().astype(np.int32)

            mat_colors = colors[mat_pred]
            mat_colors = mat_colors.numpy()
            assert occ_points.shape[0] == mat_pred.shape[0]

            pcd = trimesh.PointCloud(occ_points, colors=mat_colors)

            pcd_file = os.path.join(sub_mat_dir, 'mat.ply')
            pcd.export(pcd_file)

        if part_outputs is not None:
            if args.use_empty_for_part:
                part_outputs = part_outputs[:, :, 1:]
            part_pred = part_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
            points_info['point_part_labels'] = part_pred.numpy().astype(np.int32)

            part_colors = pcolors[part_pred]
            part_colors = part_colors.numpy()
            assert occ_points.shape[0] == part_pred.shape[0]

            pcd = trimesh.PointCloud(occ_points, colors=part_colors)

            pcd_file = os.path.join(sub_mat_dir, 'part.ply')
            pcd.export(pcd_file)

        if mmid_outputs is not None:
            mmid_pred = mmid_outputs.argmax(dim=-1).flatten(0, 1)[occ_pred==1]
            points_info['mmid'] = mmid_pred.numpy().astype(np.int32)

            mmid_colors = mcolors[mmid_pred]
            mmid_colors = mmid_colors.numpy()
            assert occ_points.shape[0] == mmid_pred.shape[0]

            pcd = trimesh.PointCloud(occ_points, colors=mmid_colors)

            pcd_file = os.path.join(sub_mat_dir, f'mmid.ply')
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
            pcd.export(os.path.join(sub_mat_dir, 'sampled_points.ply'))
            if 'point_mat_labels' not in points_info:
                # placeholder -1
                points_info['point_mat_labels'] = (np.zeros(occ_points.shape[0]) - 1).astype(np.int32)
            np.savez_compressed(os.path.join(sub_mat_dir, 'sampled_points_info.npz'), **points_info)
