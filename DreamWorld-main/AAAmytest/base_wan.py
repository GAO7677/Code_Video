
'''

CUDA_VISIBLE_DEVICES=1 python /home/gaoya/Code_Video/DreamWorld-main/AAAmytest/base_wan.py
'''
import os
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video

model_id = '/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B-Diffusers'
out_dir = '/home/gaoya/Code_Video/DreamWorld-main/outputs/demo_min'
os.makedirs(out_dir, exist_ok=True)

prompt = 'A small corgi runs across a sunny grass field, realistic motion, natural lighting.'
seed = 42
height = 256
width = 448
num_frames = 33
num_inference_steps = 20
guidance_scale = 5.0
fps = 15

save_file = os.path.join(
    out_dir,
    'wan21_diffusers_base_corgi_seed42_256x448_33f_20steps.mp4'
)

vae = AutoencoderKLWan.from_pretrained(
    model_id,
    subfolder='vae',
    torch_dtype=torch.float32,
)
pipe = WanPipeline.from_pretrained(
    model_id,
    vae=vae,
    torch_dtype=torch.bfloat16,
)
pipe.to('cuda')

generator = torch.Generator(device='cuda').manual_seed(seed)
frames = pipe(
    prompt=prompt,
    height=height,
    width=width,
    num_frames=num_frames,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
).frames[0]

export_to_video(frames, save_file, fps=fps)
print(save_file)

