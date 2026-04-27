import argparse
import datetime
import json
import numpy as np
import os
import time
import psutil
import random
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

torch.set_num_threads(8)

import util.lr_decay as lrd
import util.misc as misc
from dataset import build_3dcompat200_occupancy_dataset as build_dataset
from timm.optim.optim_factory import param_groups_weight_decay
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model import models_ae

from engine.engine_ae import train_one_epoch, evaluate, evaluate_mesh

def get_args_parser():
    parser = argparse.ArgumentParser('Autoencoder', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='ae_blob64', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--point_cloud_size', default=2048, type=int,
                        help='input size')
    parser.add_argument('--mat_start_depth', default=None, type=int,
                        help='start depth for material prediction')
    parser.add_argument('--mat_layers', default=0, type=int,
                        help='number of layers for material prediction')
    parser.add_argument('--rgb_embed_dim', default=0, type=int,
                        help='dimension of rgb embedding')
    parser.add_argument('--part_embed_dim', default=0, type=int,
                        help='dimension of part embedding')
    parser.add_argument('--mat_embed_dim', default=0, type=int,
                        help='dimension of material embedding')
    parser.add_argument('--num_part_classes', default=0, type=int,
                        help='number of part classes')
    parser.add_argument('--part_loss', action='store_true',
                        help='use part consistency loss for regularization')

    parser.add_argument('--num_mat_classes', default=None, type=int,
                        help='number of material classes')
    parser.add_argument('--decoder_ff', action='store_true',
                        help='use feedforward decoder')
    parser.add_argument('--mat_decoder_ff', action='store_true',
                        help='use feedforward decoder for material prediction')

    parser.add_argument('--requires_e', action='store_true',
                        help='requires Young\'s modulus')
    parser.add_argument('--requires_nu', action='store_true',
                        help='requires Poisson\'s ratio')
    parser.add_argument('--requires_sigma', action='store_true',
                        help='requires yield stress')
    parser.add_argument('--requires_phi', action='store_true',
                        help='requires friction angle')
    parser.add_argument('--requires_rho', action='store_true',
                        help='requires density')
    parser.add_argument('--num_mmid', default=0, type=int,
                        help='number of material models')

    parser.add_argument('--rgb_layers', default=0, type=int,
                        help='number of layers for rgb prediction')
    parser.add_argument('--rgb_decoder_ff', action='store_true',
                        help='use feedforward decoder for rgb prediction')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    parser.add_argument('--kl_weight', type=float, default=1e-3,
                        help='weight for kl divergence')
    parser.add_argument('--mat_weight', type=float, default=0.1,
                        help='weight for material prediction loss')
    parser.add_argument('--mat_emp_weight', type=float, default=1.0,
                        help='weight for material prediction loss for empty volume')
    parser.add_argument('--part_weight', type=float, default=0.1,
                        help='weight for part consistency loss')
    parser.add_argument('--rgb_weight', type=float, default=0.1,
                        help='weight for rgb prediction loss')

    # Dataset parameters
    parser.add_argument('--data_path', default='/ibex/scratch/projects/c2168/diffusion-shapes/datasets', type=str,
                        help='dataset path')
    parser.add_argument('--use_empty_for_mat', action='store_true',
                        help='use empty class for material')
    parser.add_argument('--use_empty_for_part', action='store_true',
                        help='use empty class for part')

    parser.add_argument('--output_dir', default='./output/',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output/',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform statistics evaluation only')
    parser.add_argument('--eval_save_xlsx', action='store_true',
                        help='Save evaluation results to xlsx file')
    parser.add_argument('--eval_mesh', action='store_true',
                        help='Perform mesh evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=False)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser

def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True

    dataset_train = build_dataset('train', replica=misc.get_world_size(), args=args)
    dataset_val = build_dataset('valid', replica=misc.get_world_size() if args.dist_eval else 1, args=args)
    dataset_test = build_dataset('test', replica=misc.get_world_size() if args.dist_eval else 1, args=args)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
            if len(dataset_test) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True)   # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval and not args.eval_mesh:
        os.makedirs(args.log_dir, exist_ok=True)
        file_dir = os.path.join(args.log_dir, 'files')
        os.makedirs(file_dir, exist_ok=True)
        model_file = 'model/models_ae.py'
        misc_file = 'util/misc.py'
        model_common_file = 'model/common_ae.py'
        engine_file = 'engine/engine_ae.py'        # AE 200 with Mat Params
        dataset_file = 'dataset/compat200.py'      # AE 200 with Mat Params
        to_cp_list = [model_file, misc_file, model_common_file, engine_file, dataset_file]
        for file in to_cp_list:
            if not os.path.exists(os.path.join(file_dir, file.split('/')[-1])):
                os.system(f'cp {file} {file_dir}')
        training_command_file = os.path.join(args.log_dir, 'train.sh')
        if not os.path.exists(training_command_file):
            cmd = psutil.Process(os.getpid()).cmdline()
            cmd = ' '.join(cmd)
            cmd = cmd.replace(' -', ' \\\n\t-')
            with open(training_command_file, 'w') as f:
                f.write('#!/bin/bash\n\n')
                f.write(cmd)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        prefetch_factor=2,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=1,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    models_file = models_ae
    model = models_file.__dict__[args.model](
        N=args.point_cloud_size,
        decoder_ff=args.decoder_ff,
        num_mat_classes=args.num_mat_classes,
        mat_decoder_ff=args.mat_decoder_ff,
        mat_layers=args.mat_layers,
        rgb_embed_dim=args.rgb_embed_dim,
        part_embed_dim=args.part_embed_dim,
        mat_embed_dim=args.mat_embed_dim,
        num_part_classes=args.num_part_classes,
        part_loss=args.part_loss,
        mat_start_depth=args.mat_start_depth,
        requires_e=args.requires_e,
        requires_nu=args.requires_nu,
        requires_sigma=args.requires_sigma,
        requires_phi=args.requires_phi,
        requires_rho=args.requires_rho,
        num_mmid=args.num_mmid,
        rgb_layers=args.rgb_layers,
        rgb_decoder_ff=args.rgb_decoder_ff,
    )
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    print("Model class = %s" % str(model.__class__.__name__))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    # # build optimizer with layer-wise lr decay (lrd)
    # param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
    #     no_weight_decay_list=model_without_ddp.no_weight_decay(),
    #     layer_decay=args.layer_decay
    # )
    # optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr)
    param_groups = param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.BCEWithLogitsLoss()

    print("criterion = %s" % str(criterion))

    if args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        ckpt_idx = args.resume.split('/')[-1].split('-')[-1].split('.')[0]

        test_stats = evaluate(data_loader_test, model, device, args)
        print(f'==============test results==============')
        print(f'#data: {len(dataset_test)}')

        for k, v in test_stats.items():
            print(f"{k}: {v:.3f}")
        with open(os.path.join(args.output_dir, f"test_stats-{ckpt_idx}.json"), "w") as f:
            json.dump(dict(sorted(test_stats.items())), f, indent=4)
        print(f'==============test results==============')
        exit(0)

    if args.eval_mesh:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
        evaluate_mesh(data_loader_test, model, device, args)
        exit(0)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    criterion_mmid = torch.nn.CrossEntropyLoss()
    print("Using cross entropy for material model id prediction")

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_iou = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            criterion_mmid=criterion_mmid,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch
            )

        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate(data_loader_val, model, device, args, header='Valid:')

            if log_writer is not None:
                if 'iou' in val_stats:
                    log_writer.add_scalar('val/iou', val_stats['iou'], epoch)
                if 'loss' in val_stats:
                    log_writer.add_scalar('val/loss', val_stats['loss'], epoch)
                if 'accuracy_mat' in val_stats:
                    log_writer.add_scalar('val/acc_mat', val_stats['accuracy_mat'], epoch)
                if 'loss_mat' in val_stats:
                    log_writer.add_scalar('val/loss_mat', val_stats['loss_mat'], epoch)
                if 'accuracy_part' in val_stats:
                    log_writer.add_scalar('val/acc_part', val_stats['accuracy_part'], epoch)
                if 'loss_part' in val_stats:
                    log_writer.add_scalar('val/loss_part', val_stats['loss_part'], epoch)
                if 'loss_mat_E' in val_stats:
                    log_writer.add_scalar('val/loss_mat_E', val_stats['loss_mat_E'], epoch)
                if 'loss_mat_nu' in val_stats:
                    log_writer.add_scalar('val/loss_mat_nu', val_stats['loss_mat_nu'], epoch)
                if 'accuracy_mmid' in val_stats:
                    log_writer.add_scalar('val/acc_mmid', val_stats['accuracy_mmid'], epoch)
                if 'loss_rgb' in val_stats:
                    log_writer.add_scalar('val/loss_rgb', val_stats['loss_rgb'], epoch)

            if 'iou' in val_stats:
                print(f"iou of the network on the {len(data_loader_val)} val images: {val_stats['iou']:.3f}")
                max_iou = max(max_iou, val_stats["iou"])
                print(f'Max iou: {max_iou:.2f}%')

            if log_writer is not None:
                if 'iou' in val_stats:
                    log_writer.add_scalar('val/iou', val_stats['iou'], epoch)
                if 'loss' in val_stats:
                    log_writer.add_scalar('val/loss', val_stats['loss'], epoch)
                if 'accuracy_mat' in val_stats:
                    log_writer.add_scalar('val/acc_mat', val_stats['accuracy_mat'], epoch)
                if 'loss_mat' in val_stats:
                    log_writer.add_scalar('val/loss_mat', val_stats['loss_mat'], epoch)
                if 'accuracy_part' in val_stats:
                    log_writer.add_scalar('val/acc_part', val_stats['accuracy_part'], epoch)
                if 'loss_part' in val_stats:
                    log_writer.add_scalar('val/loss_part', val_stats['loss_part'], epoch)
                if 'loss_mat_E' in val_stats:
                    log_writer.add_scalar('val/loss_mat_E', val_stats['loss_mat_E'], epoch)
                if 'loss_mat_nu' in val_stats:
                    log_writer.add_scalar('val/loss_mat_nu', val_stats['loss_mat_nu'], epoch)
                if 'accuracy_mmid' in val_stats:
                    log_writer.add_scalar('val/acc_mmid', val_stats['accuracy_mmid'], epoch)
                if 'loss_rgb' in val_stats:
                    log_writer.add_scalar('val/loss_rgb', val_stats['loss_rgb'], epoch)

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }
        else:
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    finally:
        misc.destroy_distributed_mode()
