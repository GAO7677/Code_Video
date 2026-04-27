# --------------------------------------------------------
# References:
# MAE: https://github.com/facebookresearch/mae
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable

import torch
import torch.nn.functional as F

import util.misc as misc
import util.lr_sched as lr_sched
from model.misc import prepare_mat_inputs


def train_one_epoch(model: torch.nn.Module, ae: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (
            points, labels, _, _, _, _, _, _, _, _, _, cond_signal,
            surface, surf_rgbs, surf_parts, surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, categories, _
    ) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
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
        if args.conditional_signal is None:
            conds = None
        elif args.conditional_signal in ['text', 'image']:
            conds = cond_signal.to(device, non_blocking=True)
        elif args.conditional_signal == 'category':
            conds = categories.to(device, non_blocking=True)
        else:
            raise ValueError('Unknown conditional signal: {}'.format(args.conditional_signal))

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                enc_out = ae.encode(
                    surface,
                    rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                    part=surf_parts if args.part_embed_dim > 0 else None,
                    mat=surf_mat_features if args.mat_embed_dim > 0 else None,
                )

                x = enc_out['latents']

            loss = criterion(model, x, conds)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, ae, criterion, device, args, header='Test:'):

    metric_logger = misc.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    for (
        points, labels, _, _, _, _, _, _, _, _, _, cond_signal,
        surface, surf_rgbs, surf_parts, surf_mat_Es, surf_mat_nus, surf_mat_sigmas, surf_mat_phis, surf_mat_rhos, surf_mat_mmid, categories, _
    ) in metric_logger.log_every(data_loader, 50, header):

        points = points.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).long()
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
        if args.conditional_signal is None:
            conds = None
        elif args.conditional_signal in ['text', 'image']:
            conds = cond_signal.to(device, non_blocking=True)
        elif args.conditional_signal == 'category':
            conds = categories.to(device, non_blocking=True)
        else:
            raise ValueError('Unknown conditional signal: {}'.format(args.conditional_signal))

        # compute output
        with torch.cuda.amp.autocast(enabled=False):
            with torch.no_grad():

                enc_out = ae.encode(
                    surface,
                    rgb=surf_rgbs if args.rgb_embed_dim > 0 else None,
                    part=surf_parts if args.part_embed_dim > 0 else None,
                    mat=surf_mat_features if args.mat_embed_dim > 0 else None,
                )

                x = enc_out['latents']

            loss = criterion(model, x, conds)

        metric_logger.update(loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* loss {losses.global_avg:.3f}'
          .format(losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
