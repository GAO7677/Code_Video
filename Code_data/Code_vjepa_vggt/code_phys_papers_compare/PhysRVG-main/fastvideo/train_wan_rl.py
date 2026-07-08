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

from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from diffusers.video_processor import VideoProcessor
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import AutoencoderKLWan, ModularPipeline
from diffusers.utils import export_to_video, load_image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel

from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_checkpoint_lora,
    save_lora_checkpoint,
    resume_lora_optimizer,
)
from fastvideo.utils.logging_ import main_print
from fastvideo.dataset.latent_rl_datasets import WanV2V5BDataset
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.utils import print_model,save_cfg
from fastvideo.reward.reward import reward_vjepa2,reward_pixel,reward_position,reward_position_v2v_metric
from fastvideo.sample.wanv2v_pipeline_with_logprob import wanv2v_train_onestep_with_logprob,wanv2v_sample_with_logprob_fast
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

def sample_reference_model(
    args,
    pipe,
    texts,  # List[str]
    videos, # List[List[Image.Image]]
    mask1, # List[List[Image.Image]]
    mask2, # List[List[Image.Image]]
    reward_fn,
    modules_config, # dict
    device,
):  
    # cache
    rank = int(os.environ["RANK"])
    all_latents = [] # length = steps + 1 ; each shape B,S+1,C,T,H,W
    all_log_probs = [] # length = steps ; each shape B,S,1
    all_rewards = [] # each shape B,
    all_loss = []
    all_prompt_embeds = []
    all_negative_prompt_embeds = []
    all_condition = []
    negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    # main sample <== 1. sample [num_generations] videos for 1 prompt  2. save video 3.get rewards  

    for i,(text,video,mask_1,mask_2) in enumerate(zip(texts,videos,mask1,mask2)):
        gt_path : str = None
        samples_path : list[str] = []
        prompt_embeds_i = None
        negative_prompt_embeds_i = None

        # sde_step_idx
        if args.full_traj_sde:
            sde_idxs = list(range(args.sampling_steps - 1))
        else:
            sde_idx = torch.randint(0, args.sampling_steps//4+1, (1,)).item()
            sde_idxs = [sde_idx,sde_idx+1] # 6,7
        # sde_idxs = list(range(24))
        main_print(f" ============ sde_idxs,{sde_idxs}")

        # same_noise
        if args.use_same_noise:
            latent_height = args.height // pipe.vae_scale_factor_spatial
            latent_width = args.width // pipe.vae_scale_factor_spatial            
            num_latent_frames = (args.num_frames - 1) // pipe.vae_scale_factor_temporal + 1
            num_channels_latents = pipe.vae.config.z_dim
            shape = (1, num_channels_latents, num_latent_frames, latent_height, latent_width)
            init_noise = torch.randn(shape,device=device, dtype=torch.float32)
        else:
            init_noise = None


        # sample j video for 1 prompt
        for j in range(args.num_generations): 
            main_print(f"=========== Begin Sample Text{i} Video{j} =============")

            # denoise
            # pipe.do_classifier_free_guidance = False
            sample_video,latents,log_probs,prompt_embeds,\
            negative_prompt_embeds, condition = wanv2v_sample_with_logprob_fast(self=pipe, 
                                                                        args = args,
                                                                        video = video, 
                                                                        device = device,
                                                                        sde_idxs = sde_idxs,
                                                                        prompt = text,
                                                                        negative_prompt=negative_prompt,
                                                                        prompt_embeds = prompt_embeds_i,
                                                                        negative_prompt_embeds = negative_prompt_embeds_i,
                                                                        height = args.height,
                                                                        width = args.width,
                                                                        num_frames = args.num_frames,
                                                                        num_inference_steps = args.sampling_steps,
                                                                        guidance_scale = 5,
                                                                        latents = init_noise,
                                                                        do_cfg = args.do_cfg)

            prompt_embeds_i = prompt_embeds
            negative_prompt_embeds_i = negative_prompt_embeds
            text = None
            negative_prompt = None


            # record
            s,e = sde_idxs[0],sde_idxs[0]+len(sde_idxs) # if sample_step=25, s=0,e=24
            latents = latents[:,s:e+1,:] # B,len(sde_idxs)+1,C,T,H,W 
            log_probs = log_probs[:,s:e]
            # assert torch.tensor(0.) not in log_probs , "wrong idx, log_probs != 0"

            all_latents.append(latents) # latents : B(S+1)CTHW
            all_log_probs.append(log_probs) # log_probs : BS1
            all_prompt_embeds.append(prompt_embeds) # prompt_embeds : BND
            all_negative_prompt_embeds.append(negative_prompt_embeds) # prompt_embeds : BND
            all_condition.append(condition)

            # save video
            if j == 0:
                save_path = os.path.join(args.video_cache_dir,f"rank{rank}_prompt{i}_gt.mp4")
                gt_path = save_path
                export_to_video(video, save_path, fps=args.fps, macro_block_size=4)
                mask1_save_path = save_path.replace(".mp4","_mask_1.mp4")
                mask2_save_path = save_path.replace(".mp4","_mask_2.mp4")
                export_to_video(mask_1, mask1_save_path, fps=args.fps, macro_block_size=4)
                export_to_video(mask_2, mask2_save_path, fps=args.fps, macro_block_size=4)

            save_path = os.path.join(args.video_cache_dir,f"rank{rank}_prompt{i}_sample{j}.mp4")
            samples_path.append(save_path)
            export_to_video(sample_video[0], save_path, fps=args.fps, macro_block_size=4)

        # reward 
        main_print(f"----------->> waiting for reward!")
        loss_ft,loss_fg,reward_ft,reward_fg =  reward_fn(gt_video_path = gt_path,
                                                        sample_videos_path = samples_path,
                                                        modules_config = modules_config,
                                                        device = device,
                                                        is_rank = args.is_rank,
                                                        collision_loss_weight = args.collision_loss_weight,
                                                        ignore_static = args.ignore_static) # keys "model" "transform"
        # change name for visual
        for i,sample_path in enumerate(samples_path):
            float_ft = round(loss_ft[i].item(),2)
            float_fg = round(reward_fg[i].item(),2)
            old_path = sample_path
            new_path = sample_path.split(".mp4")[0] + f"_loss{float_ft}_fg{float_fg}.mp4"
            os.rename(old_path,new_path)

        all_loss.append(loss_ft)
        all_rewards.append(reward_ft)


    
    # cat
    all_latents = torch.cat(all_latents) # B(S+1)CTHW <-- B = num_prompt * num_generations
    all_log_probs = torch.cat(all_log_probs) # BS1
    all_rewards = torch.cat(all_rewards) # B
    all_loss = torch.cat(all_loss) # B
    all_prompt_embeds = torch.cat(all_prompt_embeds) # BSD
    all_negative_prompt_embeds = torch.cat(all_negative_prompt_embeds) # BSD
    all_condition = torch.cat(all_condition) # BC1HW



    return all_latents, all_log_probs, all_prompt_embeds, all_negative_prompt_embeds, all_condition, all_loss, all_rewards, sde_idxs


def train_one_step(
    args,
    device,
    pipe,
    transformer,
    reward_fn,
    modules_config,
    optimizer,
    lr_scheduler,
    loader,
    max_grad_norm,
    step,
):

    # clear video cache dir
    video_files = os.listdir(args.video_cache_dir)
    video_paths = [os.path.join(args.video_cache_dir,f) for f in video_files]
    rank = int(os.environ["RANK"])
    if rank<=0:
        for p in video_paths:
            os.remove(p)

    # init
    total_loss = 0.0
    optimizer.zero_grad()


    # data
    data = next(loader)
    texts = data["text"] # List[str]
    videos = data["video"] # List[List[Image.Image]]
    mask1 = data["mask1"]
    mask2 = data["mask2"]


    main_print("================ Begin Sampleing Phase ==============")

    # sample

    all_latents, all_log_probs, all_prompt_embeds, all_negative_prompt_embeds, all_condition, all_loss, all_rewards, sde_idxs\
     = sample_reference_model(
                args = args,
                pipe = pipe,
                texts = texts,  # List[str]
                videos = videos, # List[List[Image.Image]]
                mask1 = mask1,
                mask2 = mask2,
                reward_fn = reward_fn,
                modules_config = modules_config,
                device = device,) # dict)

    # record timesteps
    batch_size = all_latents.shape[0]
    timesteps = pipe.scheduler.timesteps[sde_idxs].to(all_latents.device)
    timesteps = timesteps.repeat((batch_size,1)) # (16,16)

    # check shape
    assert len(pipe.scheduler.timesteps) == args.sampling_steps , f"len(pipe.scheduler.timesteps) {len(pipe.scheduler.timesteps)} != args.sampling_steps {args.sampling_steps}"
    assert timesteps.shape[1] == all_latents.shape[1]-1 , f"timesteps.shape[1] != all_latents.shape[1]-1 ; timesteps:{timesteps.shape},all_latents:{all_latents.shape}"
    assert timesteps.shape[1] == all_log_probs.shape[1], f"timesteps.shape[1] != all_log_probs.shape[1] ; timesteps:{timesteps.shape},all_log_probs:{all_log_probs.shape}"

    # samples
    samples = {
        "timesteps":    timesteps.detach().clone(), # BS (16,15)
        "latents":      all_latents[:, :-1],  # B,step,c,t,h,w , torch.Size([16, 15, 16, 14, 60, 60])
        "next_latents": all_latents[:, 1:],  # each entry is the latent after timestep t , torch.Size([16, 15, 16, 14, 60, 60])
        'prompt_embeds': all_prompt_embeds, # BSD
        'negative_prompt_embeds': all_negative_prompt_embeds, # BSD
        'condition': all_condition, # BC1HW
        "log_probs":    all_log_probs, # torch.Size([16, 15])
        "rewards":      all_rewards.to(torch.float32), # torch.Size([16])
        'loss':         all_loss.to(torch.float32),} # torch.Size([16])

    # print(f" ========= timesteps {samples['timesteps'].shape}")
    # print(f" ========= latents {samples['latents'].shape}" )
    # import ipdb; ipdb.set_trace()
    
    # log reward & loss
    gathered_reward = gather_tensor(samples["rewards"]) # torch.Size([64]) gpu * num_generations
    gathered_loss = gather_tensor(samples["loss"]) # torch.Size([64]) gpu * num_generations
    print("*" * 50)
    # print("gather_loss",gathered_loss.shape)
    if dist.get_rank()==0:
        print("gathered_reward", gathered_reward)
        with open(os.path.join(args.log_dir,'reward.txt'), 'a') as f:  
            f.write(f"{gathered_reward.mean().item()}\n")
        print("gathered_loss", gathered_loss)
        with open(os.path.join(args.log_dir,'loss.txt'), 'a') as f:  
            f.write(f"{gathered_loss.mean().item()}\n")
    
    # test
    threshold = args.hybrid_train_threshold
    bad_mask = gathered_loss > threshold
    bad_mean = gathered_loss[bad_mask].mean().item()
    bad_count = torch.sum(bad_mask).item()
    good_mask = gathered_loss < threshold
    good_mean = gathered_loss[good_mask].mean().item()
    good_count = torch.sum(good_mask).item()
    if dist.get_rank()==0:
        with open(os.path.join(args.log_dir,'bad_good.txt'), 'a') as f:  
            f.write(f"{bad_count}\t{bad_mean}\t{good_count}\t{good_mean}\n")



    # calculate advantage
    n = len(samples["rewards"]) // (args.num_generations) # 1 表示1个gpu本次有几个prompt
    advantages = torch.zeros_like(samples["rewards"])
    
    for i in range(n):
        start_idx = i * args.num_generations
        end_idx = (i + 1) * args.num_generations
        group_rewards = samples["rewards"][start_idx:end_idx]
        group_mean = group_rewards.mean()
        group_std = group_rewards.std() + 1e-8
        advantages[start_idx:end_idx] = (group_rewards - group_mean) / group_std
    
    samples["advantages"] = advantages
    main_print(f"===== reward ====== \n {samples['rewards']}")
    main_print(f"===== advantages ====== \n {samples['advantages']}")

    # best-of-n strategy
    selected_idx_list = []
    for i in range(n):
        start_idx = i * args.num_generations
        end_idx = (i + 1) * args.num_generations        
        total_scores = samples["advantages"][start_idx:end_idx]
        sorted_indices = torch.argsort(total_scores) + start_idx # torch.Size([16])
        top_indices = sorted_indices[-args.bestofn//2:]     
        bottom_indices = sorted_indices[:args.bestofn//2]     
        selected_indices = torch.cat([top_indices, bottom_indices])
        selected_idx_list.append(selected_indices)
        
    assert len(selected_idx_list) == n , f"selected_idx_list's length{len(selected_idx_list)} != num of prompts {n}"
    selected_indices_all = torch.cat(selected_idx_list) if len(selected_idx_list) > 1 else selected_idx_list[0]

    shuffled_order = torch.randperm(len(selected_indices_all), device=selected_indices_all.device)
    selected_indices = selected_indices_all[shuffled_order]      # torch.Size([8])
    assert len(selected_indices) == args.bestofn * args.train_batch_size

    if args.train_batch_size * args.num_generations != args.bestofn:
        for key in samples:
            samples[key] = samples[key][selected_indices]
        batch_size = len(selected_indices)

    # random along timesteps
    perms = torch.stack(
        [
            torch.randperm(len(samples["timesteps"][0]))
            for _ in range(batch_size)
        ]
    ).to(device)  # torch.Size([8, 15])
    for key in ["timesteps", "latents", "next_latents", "log_probs"]:
        samples[key] = samples[key][
            torch.arange(batch_size).to(device) [:, None],
            perms,
        ]
    samples_batched = {
        k: v.unsqueeze(1)
        for k, v in samples.items()
    }
    # dict of lists -> list of dicts for easier iteration
    samples_batched_list = [
        dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
    ] # (8,) dict_keys(['timesteps', 'latents', 'next_latents', 'log_probs', 'vq_rewards', 'mq_rewards', 'encoder_hidden_states', 'encoder_attention_mask', 'vq_advantages', 'mq_advantages'])
    train_timesteps = int(len(samples["timesteps"][0])*args.timestep_fraction)


    # train
    sft_count = 0
    rl_count = 0
    for i,sample in list(enumerate(samples_batched_list)):

        # SFT RL count
        do_sft = False
        do_rl = True
        if args.hybrid_train and samples["loss"].mean().item() > args.hybrid_train_threshold:
            do_sft = True
            sft_count +=1
            rl_count +=1
        else:
            rl_count +=1
        print(f"* Loss {samples['loss'].mean().item()} , SFT{do_sft} RL{do_rl}")

        # SFT
        if args.hybrid_train:
            pipe.scheduler.set_timesteps(1000,device)
            timestep_id = torch.randint(0, 1000, (1,))
            timestep = pipe.scheduler.timesteps[timestep_id].to(device=device)

            loss = pipe.forward_loss(
                            video = videos[0],
                            timestep = timestep,
                            device = device,
                            prompt = texts[0],
                            height = args.height,
                            width = args.width,
                            num_frames = args.num_frames,
                        )
            final_loss = loss / args.gradient_accumulation_steps
            final_loss = args.sft_alpha * 1.0 * final_loss if do_sft else 0.0 * final_loss
            final_loss.backward()

        # RL
        pipe.scheduler.set_timesteps(args.sampling_steps, device=device)
        for _ in range(train_timesteps):
            clip_range = 1e-4
            adv_clip_max = 5.0

            new_log_probs = wanv2v_train_onestep_with_logprob(
                self = pipe,
                args = args,
                timestep = sample["timesteps"][0,_],
                latents = sample["latents"][:,_],
                prev_latents = sample["next_latents"][:,_],
                condition = sample["condition"],
                prompt_embeds = sample["prompt_embeds"],
                negative_prompt_embeds = sample["negative_prompt_embeds"],
                guidance_scale = args.guidance_scale,
                device = device,
                do_cfg = args.do_cfg,
            )

            # ratio
            ratio = torch.exp(new_log_probs - sample["log_probs"][:,_])
            if int(os.environ["LOCAL_RANK"]) <= 0:
                print(f"timestep:{_}")
                print(f"ratio:{ratio}")


            # RL loss
            advantages = torch.clamp(
                sample["advantages"],
                -adv_clip_max,
                adv_clip_max,
            )
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * torch.clamp(
                ratio,
                1.0 - clip_range,
                1.0 + clip_range,
            )
            loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss)) / (args.gradient_accumulation_steps * train_timesteps)
            
            final_loss = 1.0 * loss if do_rl else 0.0 * loss
            final_loss.backward()
            # avg_loss = final_loss.detach().clone()
            # dist.all_reduce(avg_loss, op=dist.ReduceOp.AVG)
            # total_loss += avg_loss.item()

        # step
        if (i+1)%args.gradient_accumulation_steps==0:
            grad_norm = transformer.clip_grad_norm_(max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # print
        if dist.get_rank()<=0:
            print(f"--------------------- TRAIN BATCH{i}  ---------------------")
            # print("* loss                        ", loss.item())
            print("* reward                      ", sample["rewards"].item())
            print("* ratio                       ", ratio)
            print("* advantage                   ", advantages.item())
            if (i+1)%args.gradient_accumulation_steps==0:
                print("* Model Update This Step!     ")

    # log count
    sft_tensor = torch.tensor(sft_count, dtype=torch.long, device=device)
    dist.all_reduce(sft_tensor, op=dist.ReduceOp.SUM)
    rl_tensor = torch.tensor(rl_count, dtype=torch.long, device=device)
    dist.all_reduce(rl_tensor, op=dist.ReduceOp.SUM)

    if dist.get_rank() == 0:
        with open(os.path.join(args.log_dir,'count.txt'), 'a') as f:  
            f.write(f"{sft_tensor.item()} {rl_tensor.item()}\n")
            
        current_lr = optimizer.param_groups[0]['lr']
        with open(os.path.join(args.log_dir,'lr.txt'), "a") as f:
            f.write(f"{step}\t{current_lr}\n")


    return None, grad_norm.item()

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
    print(f"== local_rank{local_rank} rank{rank} world_size{world_size} device{device} ==")

    # seed
    if args.seed is not None:
        set_seed(args.seed+rank)
    
    # repository 
    now = datetime.now()
    current_time = f'{now.year}-{now.month}-{now.day}-{now.hour}'
    exp_dir = os.path.join(args.output_dir,f"{args.exp_name}-{current_time}") # main exp root dir 

    ckpt_dir = os.path.join(exp_dir,"ckpt")
    log_dir = os.path.join(exp_dir,"log")
    video_cache_dir = os.path.join(exp_dir,"videos")
    args.video_cache_dir = video_cache_dir
    args.log_dir = log_dir
    if rank <= 0:
        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(video_cache_dir,exist_ok=True)
    
    # config
    if rank<=0:
        save_cfg(exp_dir,args)

    # reward model 
    reward_model = None
    reward_fn = None
    if args.reward_type == "vjepa2":
        from transformers import AutoVideoProcessor, AutoModel
        reward_model = AutoModel.from_pretrained(args.reward_model_path).to(device)
        transform = AutoVideoProcessor.from_pretrained(args.reward_model_path)
        reward_fn = reward_vjepa2
        modules_config = {"model":reward_model,
                          "transform":transform}
    elif args.reward_type == "pixel":
        reward_fn = reward_pixel
        modules_config = {"model":None,
                          "transform":None}   
    elif args.reward_type == "position":
        from transformers import Sam2VideoModel, Sam2VideoProcessor
        model = Sam2VideoModel.from_pretrained(args.reward_model_path).to(device, dtype=torch.bfloat16)
        processor = Sam2VideoProcessor.from_pretrained(args.reward_model_path)

        reward_fn = reward_position
        modules_config = {"model":model,
                          "processor":processor,
                          "mask_file_suffix":"_mask"}     

    # model
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(args.model_id,subfolder="transformer", torch_dtype=torch.float32)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_id, transformer= transformer, vae=vae, torch_dtype=torch.float32)
    pipe.vae.to(device)
    pipe.text_encoder.to(device)
    print("==== text_encoder.device== ",pipe.text_encoder.device)

    # gradient_checkpoint
    if args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    
    # lora or resume
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
        main_print(f"--> Resume LORA from {args.resume_from_lora_checkpoint}") 
    elif args.use_lora:
        pipe.transformer = get_peft_model(pipe.transformer, lora_config) 

    # info
    main_print(f"gradient_checkpointing {pipe.transformer.gradient_checkpointing}")
    if rank <= 0:
        print_model(pipe.transformer,"transformer")
    main_print(f"--> model loaded")
    main_print(f"  Total training parameters = {sum(p.numel() for p in pipe.transformer.parameters() if p.requires_grad) / 1e6} M")

    # fsdp 
    from peft.utils.other import fsdp_auto_wrap_policy
    from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "WanTransformerBlock"
    fsdp_kwargs={"auto_wrap_policy": fsdp_auto_wrap_policy(pipe.transformer),
                "sharding_strategy": ShardingStrategy.FULL_SHARD,
                "limit_all_gathers": True,
                "use_orig_params": False,  
                "sync_module_states": True,
                "device_id":device}
    pipe.transformer = FSDP(pipe.transformer, **fsdp_kwargs,)

    # set model as trainable.
    pipe.transformer.train()
    transformer = pipe.transformer

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
    init_steps = 0
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
        last_epoch=init_steps - 1,
    )

    # dataset
    train_dataset = WanV2V5BDataset(json_path = args.data_json_path, 
                                    width = args.width, 
                                    height = args.height,
                                    num_frames = args.num_frames,
                                    start_max = args.start_max,
                                    need_mask = True,
                                    data_repeat = args.data_repeat)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True, seed=args.sampler_seed)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=sampler,
                                  collate_fn=train_dataset.collate_fn,
                                  pin_memory=True,
                                  batch_size=args.train_batch_size, # 1
                                  num_workers=args.dataloader_num_workers,
                                  drop_last=True,)
    loader = iter(train_dataloader) # make it iterable

    # # tensorboard
    # if rank <= 0:
    #     writer = SummaryWriter(log_dir)

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
            sampler.set_epoch(epoch) # Crucial for distributed shuffling per epoch

        for step in range(init_steps, args.max_train_steps):
            start_time = time.time()

            # save weight
            if step % args.checkpointing_steps == 0 and step > 0:

                # save model
                if args.use_lora:
                    save_checkpoint_lora(transformer, rank, ckpt_dir,step, epoch)
                else:
                    save_checkpoint(transformer, rank, ckpt_dir, step, epoch)

                # clear
                if rank<=0:
                    clear_dir(ckpt_dir,args.checkpoints_total_limit)

                dist.barrier()


            
            # train
            loss, grad_norm = train_one_step(
                args = args,
                device = device,
                pipe = pipe,
                transformer = transformer,
                reward_fn = reward_fn,
                modules_config = modules_config,
                optimizer = optimizer,
                lr_scheduler = lr_scheduler,
                loader = loader,
                max_grad_norm = args.max_grad_norm,
                step = step,
            )

            # log
            step_time = time.time() - start_time
            step_times.append(step_time)
            avg_step_time = sum(step_times) / len(step_times)
    
            progress_bar.set_postfix(
                {
                    # "loss": f"{loss:.4f}",
                    "step_time": f"{step_time:.2f}s",
                    "grad_norm": grad_norm,
                }
            )
            progress_bar.update(1)
            # if rank <= 0:
            #     writer.add_scalar('Loss/train', loss, step)



    if get_sequence_parallel_state():
        destroy_sequence_parallel_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # train 
    parser.add_argument("--train_epoch", type=int, default=10)
    parser.add_argument("--is_rank",action="store_true")
    parser.add_argument("--exp_name", type=str, default="RL")
    parser.add_argument("--do_cfg",action="store_true")
    parser.add_argument("--full_traj_sde",action="store_true")
    parser.add_argument("--hybrid_train",action="store_true")
    parser.add_argument("--hybrid_train_threshold", type=float, default=8.0)
    parser.add_argument("--collision_loss_weight",action="store_true")
    parser.add_argument("--ignore_static",action="store_true")
    parser.add_argument("--sft_alpha", type=float, default=1.0)
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
    parser.add_argument(
        "--model_type", type=str, default="wan_hf", help="The type of model to train."
    )
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--dit_model_name_or_path", type=str, default=None)
    parser.add_argument("--vae_model_path", type=str, default=None, help="vae model.")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")

    # diffusion setting
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--precondition_outputs",
        action="store_true",
        help="Whether to precondition the outputs of the model.",
    )
    parser.add_argument("--train_cfg",action="store_true")
    parser.add_argument("--guidance_scale",type=float, default=5.0)
    parser.add_argument("--freeze_cfg_gradient",action="store_true")

    # reward 
    parser.add_argument("--reward_type", type=str, default="vjepa2",help="vjepa2,dinov3,clip,pixel")
    parser.add_argument("--reward_model_path", type=str, required = True,help="vjepa2,dinov3,clip,pixel")


    # validation & logs
    parser.add_argument("--validation_prompt_dir", type=str)
    parser.add_argument("--uncond_prompt_dir", type=str)
    parser.add_argument(
        "--validation_sampling_steps",
        type=str,
        default="64",
        help="use ',' to split multi sampling steps",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=str,
        default="4.5",
        help="use ',' to split multi scale",
    )
    parser.add_argument("--log_validation", action="store_true")
    parser.add_argument("--tracker_project_name", type=str, default=None)
    parser.add_argument(
        "--seed", type=int, default=42, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required = True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
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
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
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
        "--t",
        type=int,
        default=None,   
        help="video length",
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
        "--use_videoalign",
        action="store_true",
        default=False,
        help="whether to videoalign reward model",
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
    parser.add_argument(
        "--vq_coef",
        type=float,
        default=1.0,   
        help="vq coef",
    )
    parser.add_argument(
        "--mq_coef",
        type=float,
        default=0.0,   
        help="mq coef",
    )

    args = parser.parse_args()
    main(args)