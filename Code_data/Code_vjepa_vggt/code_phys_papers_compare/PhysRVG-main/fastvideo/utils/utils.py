import argparse
import binascii
import os
import os.path as osp
import json
from typing import Set
import imageio
import torch
import torchvision
from peft import LoraConfig, inject_adapter_in_model
import shutil

__all__ = ['cache_video', 'cache_image', 'str2bool']


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
        try:
            # preprocess
            tensor = tensor.clamp(min(value_range), max(value_range))
            tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
            tensor = (tensor * 255).type(torch.uint8).cpu()

            # write video
            writer = imageio.get_writer(
                cache_file, fps=fps, codec='libx264', quality=8)
            for frame in tensor.numpy():
                writer.append_data(frame)
            writer.close()
            return cache_file
        except Exception as e:
            error = e
            continue
    else:
        print(f'cache_video failed, error: {error}', flush=True)
        return None


def cache_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    # cache file
    suffix = osp.splitext(save_file)[1]
    if suffix.lower() not in [
            '.jpg', '.jpeg', '.png', '.tiff', '.gif', '.webp'
    ]:
        suffix = '.png'

    # save to cache
    error = None
    for _ in range(retry):
        try:
            tensor = tensor.clamp(min(value_range), max(value_range))
            torchvision.utils.save_image(
                tensor,
                save_file,
                nrow=nrow,
                normalize=normalize,
                value_range=value_range)
            return save_file
        except Exception as e:
            error = e
            continue


def str2bool(v):
    """
    Convert a string to a boolean.

    Supported true values: 'yes', 'true', 't', 'y', '1'
    Supported false values: 'no', 'false', 'f', 'n', '0'

    Args:
        v (str): String to convert.

    Returns:
        bool: Converted boolean value.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be converted to boolean.
    """
    if isinstance(v, bool):
        return v
    v_lower = v.lower()
    if v_lower in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v_lower in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected (True/False)')

def save_cfg(path, args):
    os.makedirs(path, exist_ok=True)
    with open(f'{path}/args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def print_model(model,text="Unknown Model"):
    """
    打印模型的参数数量
    """
    print(f"*----------- {text} ----------*")
    device = next(model.parameters()).device
    print(f"device:{device}")
    
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freeze_params = sum(p.numel() for p in model.parameters() if p.requires_grad is False)

    print(f'## 模型总参数数量:{total_params / 1_000_000:.2f}M')
    print(f'#  模型训练数量:{train_params / 1_000_000:.2f}M')
    print(f'# 模型冻结参数数量:{freeze_params / 1_000_000:.2f}M')

def print_param_num(model,text="Unknown Model"):
    """
    打印模型的参数数量
    """
    print(f"*----------- {text} ----------*")
    total_params = sum(p.numel() for p in model.parameters())

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    freeze_params = sum(p.numel() for p in model.parameters() if p.requires_grad is False)

    print(f'## 模型总参数数量:{total_params / 1_000_000:.2f}M')
    print(f'#  模型训练数量:{train_params / 1_000_000:.2f}M')
    print(f'# 模型冻结参数数量:{freeze_params / 1_000_000:.2f}M')

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True

def save_video(video:torch.Tensor,save_path="output.mp4",fps=16):
    # video : T C H W (-1,1)
    assert video.shape[1]==3 , f"Required Shape is T C H W, your shape is {video.shape}"
    video = (video + 1)/2*255
    video = video.clamp(0, 255).to(device="cpu", dtype=torch.uint8)
    video_numpy = video.permute(0, 2, 3, 1).numpy() 
    imageio.mimsave(save_path, video_numpy, fps=16)

def clear_ckpt_dir(directory,max_limit_ckpt=1):
    checkpoints = os.listdir(directory)
    checkpoints = [d for d in checkpoints if "safetensor" in d]
   
    # before we save the new checkpoint, we need to have at _most_ `checkpoint_total_limit ` checkpoints
    if len(checkpoints) > max_limit_ckpt:
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split(".")[0].split("-")[-1]))
        num_to_remove = len(checkpoints) - max_limit_ckpt 
        removing_checkpoints = checkpoints[0:num_to_remove]
        for removing_checkpoint in removing_checkpoints:
            removing_checkpoint = os.path.join(directory, removing_checkpoint)
            os.remove(removing_checkpoint)


class LoraManager():
    def __init__(self,model):
        self.model = model

    def trainable_parameters(self):
        # 只计算 model的参数
        trainable_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return trainable_parameters
    
    def trainable_param_names(self) -> Set[str]:
        trainable_param_names = self.trainable_named_params()
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        return trainable_param_names # set[str]
    
    def trainable_named_params(self):
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.model.named_parameters()))
        return trainable_param_names

    def add_lora_to_model(self, model, target_modules = ["q","k","v","o","ffn.0","ffn.2"], lora_rank=32, lora_alpha=None):
        if lora_alpha is None:
            lora_alpha = lora_rank
        lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=target_modules)
        model = inject_adapter_in_model(lora_config, model)
        return model
    
    def freeze_except(self,model,keywords):
        
        def is_exist_in_word(word,l):
            for w in l:
                if w in word:
                    return True
            return False
        
        if len(keywords)==0:
            for param in model.parameters():
                param.requires_grad = False 
        else:
            for k,param in model.named_parameters():
                if is_exist_in_word(k,keywords):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

def scale_with_linear_warmup(start_scale, final_scale, warmup_step=None, max_train_step=10000):
    if warmup_step is None:
      return [start_scale] * max_train_step

    scales = []
    # warmup阶段线性递减
    for i in range(warmup_step):
        interp = start_scale + (final_scale - start_scale) * i / (warmup_step - 1) if warmup_step > 1 else start_scale
        scales.append(interp)
    # warmup后恒定final_scale
    for i in range(warmup_step, max_train_step):
        scales.append(final_scale)
    return scales

def clear_dir(root,save_num = 2):
    pts = os.listdir(root)
    pts = sorted(pts, key=lambda x: int(x.split("-")[1]))
    move_pts = pts[:-save_num]
    print(f"--> ckpt removing : {move_pts}")
    for f in move_pts:
        shutil.rmtree(os.path.join(root,f))