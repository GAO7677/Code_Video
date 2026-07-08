import argparse
import math
import os
import time
import sys
import time
from collections import deque
from tqdm.auto import tqdm
from email.policy import strict
from pathlib import Path
from datetime import datetime
from functools import partial
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.video_processor import VideoProcessor
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
from safetensors.torch import load_file, save_file
from transformers import Sam2VideoModel, Sam2VideoProcessor

from fastvideo.utils.logging_ import main_print
from fastvideo.dataset.latent_rl_datasets import WanV2V5BDataset
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.utils import print_model,save_cfg
from fastvideo.reward.reward import reward_position_v2v_metric
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline
from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel

from utils.utils import clear_dir

def gather_tensor(tensor):
    if not dist.is_initialized():
        return tensor
    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

                
def main(args):

    # dist init
    torch.backends.cuda.matmul.allow_tf32 = True
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device_id = torch.cuda.current_device()
    device = torch.device(f"cuda:{device_id}")
    print(f"== local_rank {local_rank} rank {rank} world_size {world_size} device {device} ==")

    # seed
    if args.seed is not None:
        set_seed(args.seed+rank)
    
    # repository 
    now = datetime.now()
    current_time = f'{args.exp_name}-{now.year}-{now.month}-{now.day}-{now.hour}'
    exp_dir = os.path.join(args.output_dir,current_time) # main exp root dir 

    ckpt_dir = os.path.join(exp_dir,"ckpt")
    log_dir = os.path.join(exp_dir,"log")
    video_cache_dir = os.path.join(exp_dir,"videos")
    args.video_cache_dir = video_cache_dir
    args.log_dir = log_dir
    if rank <= 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
    
    # config
    if rank<=0:
        save_cfg(exp_dir,args)    

    # model
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(args.model_id,subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_id, transformer= transformer, vae=vae, torch_dtype=torch.bfloat16)    

    # gradient_checkpoint
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    # lora
    target_modules = ["to_q","to_k","to_v","to_out.0","net.0.proj","net.2"]
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    if args.resume_from_checkpoint:
        from safetensors.torch import load_file
        resume_pt = load_file(args.resume_from_checkpoint)
        pipe.transformer.load_state_dict(resume_pt)
        main_print(f"--> Resume from {args.resume_from_checkpoint}")  
    if args.use_lora and args.resume_from_lora_checkpoint:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.resume_from_lora_checkpoint)
        pipe.transformer.set_adapter("default")  
    elif args.use_lora:
        pipe.transformer = get_peft_model(pipe.transformer, lora_config) 

    pipe.to(device) 
    pipe.to(torch.bfloat16)
    print("==== text_encoder.device== ",pipe.text_encoder.device)
    
    # DDP
    pipe.transformer = DDP(pipe.transformer, device_ids=[local_rank])
    pipe.transformer.train()
    transformer = pipe.transformer

    # # Metric model
    # model = Sam2VideoModel.from_pretrained(args.reward_model_path).to(device, dtype=torch.bfloat16)
    # processor = Sam2VideoProcessor.from_pretrained(args.reward_model_path)
    # modules_config = {"model":model,
    #                 "processor":processor,
    #                 "mask_file_suffix":"_mask"}  

    # info
    main_print(f"gradient_checkpointing {pipe.transformer.module.gradient_checkpointing}")
    if rank <= 0:
        print_model(pipe.transformer,"transformer")
    main_print(f"--> model loaded")
    main_print(f"  Total training parameters = {sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad) / 1e6} M")


    # optimizer
    params_to_optimize = pipe.transformer.parameters()
    params_to_optimize = list(filter(lambda p: p.requires_grad, params_to_optimize))
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )
    init_steps = args.init_steps
    global_step =init_steps
    main_print(f"optimizer: {optimizer}")
    num_train_params = sum(p.numel() for p in params_to_optimize)
    print(f"Trainable param count: {num_train_params/1e6:.2f} M")

    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=1000000,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
        # last_epoch=init_steps - 1,
    )

    # dataset
    train_dataset = WanV2V5BDataset(json_path = args.data_json_path, 
                                    width = args.width, 
                                    height = args.height,
                                    num_frames = args.num_frames,
                                    start_max = args.start_max,
                                    need_mask = False,
                                    data_repeat = args.data_repeat)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  collate_fn=train_dataset.collate_fn,
                                  pin_memory=True,
                                  batch_size=args.train_batch_size, # 1
                                  num_workers=args.dataloader_num_workers,
                                  drop_last=False,)

    # tensorboard
    if rank <= 0:
        writer = SummaryWriter(log_dir)

    # log!
    total_batch_size = (
        args.train_batch_size
        * world_size
        * args.gradient_accumulation_steps
        / args.sp_size
        * args.train_sp_batch_size
    )
    main_print("***** Running training *****")
    main_print(f"  Num examples = {len(train_dataset)}")
    main_print(f"  Dataloader size = {len(train_dataloader)}")
    main_print(f"  Resume training from step {init_steps}")
    main_print(f"  Instantaneous batch size per device = {args.train_batch_size}")
    main_print(f"  Total train batch size (w. data & sequence parallel, accumulation) = {total_batch_size}")
    main_print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    main_print(f"  Total optimization steps per epoch = {args.max_train_steps}")
    main_print(f"  Total training parameters per FSDP shard = {sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad) / 1e9} B")
    main_print(f"  Master weight dtype: {pipe.transformer.parameters().__next__().dtype}")

    # progress bar
    progress_bar = tqdm(
        range(0, 100000),
        initial=init_steps,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=local_rank > 0,
    )

    step_times = deque(maxlen=100)

    # train
    for epoch in range(args.train_epoch):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch+2) # Crucial for distributed shuffling per epoch
            
        if global_step > args.max_train_steps:
            break

        for step,data in enumerate(train_dataloader):

            # resume
            if epoch == 0 and step < init_steps:
                continue


            start_time = time.time()

            # save weight
            if global_step % args.checkpointing_steps == 0 and rank<=0 and global_step>0:
                save_dir = os.path.join(ckpt_dir, f"checkpoint-{global_step}-{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                # save model
                if args.use_lora:
                    main_print(f"============ Saving checkpoint lora ============")
                    pipe.transformer.module.save_pretrained(save_dir)
                else:

                    main_print("============ Saving Model ==========")
                    state_dict = pipe.transformer.module.state_dict()
                    weight_path = os.path.join(save_dir, "diffusion_pytorch_model.safetensors")
                    save_file(state_dict, weight_path)
                    main_print(f"============ Saving Model to {weight_path}==========")

                clear_dir(ckpt_dir,args.checkpoints_total_limit)
            
            # data
            texts = data["text"] # List[str]
            videos = data["video"] # List[List[Image.Image]]
            
            # train
            pipe.scheduler.set_timesteps(1000,device)
            timestep_id = torch.randint(0, 1000, (1,))
            timestep = pipe.scheduler.timesteps[timestep_id].to( device=device)

            loss = pipe.forward_loss(
                video = videos[0],
                timestep = timestep,
                device = device,
                prompt = texts[0],
                height = args.height,
                width = args.width,
                num_frames = args.num_frames,
            )

            # backward
            final_loss = loss / args.gradient_accumulation_steps
            final_loss.backward()
            avg_loss = loss.detach().clone()
            dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)

            # log
            if rank<=0:
                with open(os.path.join(args.log_dir,'loss.txt'), 'a') as f:  
                    f.write(f"{avg_loss.mean().item()}\n")

            # step
            if (step+1)%args.gradient_accumulation_steps==0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # log
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    "epoch": f"{epoch}",
                    "loss": f"{avg_loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                }
            )
            progress_bar.update(1)
            # if rank <= 0:
            #     writer.add_scalar('Loss/train', loss, step)
            global_step += 1



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train 
    parser.add_argument("--train_epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_start_timestep_idx", type=int, default=0)
    parser.add_argument("--init_steps", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="finetune", help="exp name")
    parser.add_argument("--start_max", type=int, default=None)
    parser.add_argument("--data_repeat", type=int, default=1)

    # dataset & dataloader
    parser.add_argument("--data_json_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=10,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--sample_step", type=int, default=1,help="video sample step") 
    parser.add_argument("--height",type=int, default=None,help="video height",)
    parser.add_argument("--width",type=int,default=None, help="video width",)

    # models
    parser.add_argument("--model_id",required = True, type=str)


    # exp
    parser.add_argument("--output_dir",type=str,required = True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--checkpoints_total_limit",type=int,default=None, help=("Max number of checkpoints to store."),)
    parser.add_argument("--checkpointing_steps",type=int,default=500,help=("Save a checkpoint of the training state every X updates."),)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--resume_from_lora_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous lora checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )

    # optimizer & scheduler & Training
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--is_decay", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=10,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--max_grad_norm", default=2.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument("--selective_checkpointing", type=float, default=1.0)
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--use_cpu_offload",
        action="store_true",
        help="Whether to use CPU offload for param & gradient & optimizer states.",
    )

    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument(
        "--train_sp_batch_size",
        type=int,
        default=1,
        help="Batch size for sequence parallel training",
    )
    parser.add_argument("--fsdp_sharding_startegy", default="full")
    # lr_scheduler
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of cycles in the learning rate scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay to apply."
    )
    parser.add_argument(
        "--master_weight_type",
        type=str,
        default="fp32",
        help="Weight type to use - fp32 or bf16.",
    )
    parser.add_argument(
        "--sampling_steps",
        type=int,
        default=None,   
        help="sampling steps",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=None,   
        help="noise eta",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,   
        help="fps of stored video",
    )
    parser.add_argument(
        "--sampler_seed",
        type=int,
        default=0,   
        help="seed of sampler",
    )
    parser.add_argument(
        "--use_group",
        action="store_true",
        default=False,
        help="whether to use group",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=16,   
        help="num_generations per prompt",
    )
    parser.add_argument(
        "--use_same_noise",
        action="store_true",
        default=False,
        help="whether to use same noise",
    )
    parser.add_argument(
        "--timestep_fraction",
        type = float,
        default=1.0,
        help="timestep_fraction",
    )
    parser.add_argument(
        "--shift",
        type = float,
        default=1.0,
        help="shift value",
    )
    parser.add_argument(
        "--bestofn",
        type = int,
        default=8,
        help="the chosen samples in best-of-n",
    )

    args = parser.parse_args()
    main(args)
