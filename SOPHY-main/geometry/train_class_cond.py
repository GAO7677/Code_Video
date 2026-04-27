import argparse
import datetime
import json
import numpy as np
import os
import time
import psutil
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import util.lr_decay as lrd
import util.misc as misc
from dataset import build_3dcompat200_occupancy_dataset as build_dataset
from timm.optim.optim_factory import param_groups_weight_decay
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from model import models_ae
from model import models_cond

from engine.engine_cond import train_one_epoch, evaluate

def get_args_parser():
    parser = argparse.ArgumentParser('Latent Diffusion', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='kl_d512_m512_l8_edm', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--ae', default='kl_d512_m512_l8', type=str, metavar='MODEL',
                        help='Name of autoencoder')
    parser.add_argument('--ae-pth', required=True, help='Autoencoder checkpoint')
    parser.add_argument('--point_cloud_size', default=2048, type=int,
                        help='input size')
    parser.add_argument('--conditional_signal', default=None, type=str,
                        choices=['category', 'text', 'image'],
                        help='conditional signal')
    parser.add_argument('--conditional_signal_ver', default='v2', type=str,
                        help='conditional signal')
    parser.add_argument('--conditional_arg', default=None, type=int,
                        help='conditional arg')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-4, metavar='LR', # 2e-4
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')


    # Dataset parameters
    parser.add_argument('--data_path', default='/ibex/ai/home/zhanb0b/data', type=str,
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
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=60, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

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
    torch.manual_seed(seed)
    np.random.seed(seed)

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
                print('Warning: Enabling distributed evaluation with an test dataset not divisible by process number. '
                      'This will slightly alter test results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=True) # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if global_rank == 0 and args.log_dir is not None and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        file_dir = os.path.join(args.log_dir, 'files')
        os.makedirs(file_dir, exist_ok=True)
        model_file = 'model/models_cond.py'
        engine_file = 'engine/engine_cond.py'
        dataset_file = 'dataset/compat200.py'
        to_cp_list = [model_file, engine_file, dataset_file]
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
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, sampler=sampler_test,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # load hyper-parameter from checkpoint
    ae_ckpt = torch.load(args.ae_pth, map_location='cpu', weights_only=False)
    ae_args = ae_ckpt['args']
    args.rgb_embed_dim = ae_args.rgb_embed_dim
    args.part_embed_dim = ae_args.part_embed_dim
    args.requires_e = ae_args.requires_e
    args.requires_nu = ae_args.requires_nu
    args.requires_sigma = ae_args.requires_sigma
    args.requires_phi = ae_args.requires_phi
    args.requires_rho = ae_args.requires_rho
    args.num_mmid = ae_args.num_mmid
    args.mat_embed_dim = getattr(ae_args, 'mat_embed_dim', 0)
    prev_use_empty_for_mat = getattr(ae_args, 'use_empty_for_mat', False)
    prev_use_empty_for_part = getattr(ae_args, 'use_empty_for_part', False)
    assert args.use_empty_for_mat == prev_use_empty_for_mat, f'use empty for mat: args {args.use_empty_for_mat} != ae_args {prev_use_empty_for_mat}'
    assert args.use_empty_for_part == prev_use_empty_for_part, f'use empty for part: args {args.use_empty_for_part} != ae_args {prev_use_empty_for_part}'
    ae = models_ae.__dict__[args.ae](
        N=ae_args.point_cloud_size,
        decoder_ff=ae_args.decoder_ff,
        num_mat_classes=ae_args.num_mat_classes,
        mat_decoder_ff=ae_args.mat_decoder_ff,
        mat_layers=ae_args.mat_layers,
        rgb_embed_dim=args.rgb_embed_dim,
        part_embed_dim=args.part_embed_dim,
        mat_embed_dim=args.mat_embed_dim,
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
    print(f"Loading autoencoder {args.ae_pth}, rgb embed dim: {args.rgb_embed_dim}, part embed dim: {args.part_embed_dim}, mat embed dim: {args.mat_embed_dim}")
    ae.load_state_dict(ae_ckpt['model'])

    ae.to(device)

    model = models_cond.__dict__[args.model](args.conditional_signal, args.conditional_arg)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

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

    criterion = models_cond.__dict__['EDMLoss']()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        ckpt_idx = args.resume.split('/')[-1].split('-')[-1].split('.')[0]

        test_stats = evaluate(data_loader_test, model, ae, criterion, device, args, header='Test:')
        print(f'==============test results==============')
        print(f'#data: {len(dataset_test)}')

        for k, v in test_stats.items():
            print(f"{k}: {v:.3f}")
        with open(os.path.join(args.output_dir, f"test_stats-{ckpt_idx}.json"), "w") as f:
            json.dump(dict(sorted(test_stats.items())), f, indent=4)
        print(f'==============test results==============')
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, ae, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )
        if args.output_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        if epoch % 10 == 0 or epoch + 1 == args.epochs:
            val_stats = evaluate(data_loader_val, model, ae, criterion, device, args=args, header='Valid:')

            if log_writer is not None:
                log_writer.add_scalar('valid/loss', val_stats['loss'], epoch)
            
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'val_{k}': v for k, v in val_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            'epoch': epoch,
                            'n_parameters': n_parameters}

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
    main(args)
