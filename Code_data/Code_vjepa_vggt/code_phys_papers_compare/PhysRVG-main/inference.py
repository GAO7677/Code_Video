import os
import torch
import random
import argparse
from datetime import datetime
from accelerate.utils import set_seed
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from peft import  PeftModel
from safetensors.torch import load_file
from fastvideo.dataset.latent_rl_datasets import WanV2V5BInferDataset
from fastvideo.models.wan_v2v.pipeline_wan_v2v import WanImageToVideoPipeline
from fastvideo.models.wan_v2v.model_wan_v2v import WanTransformer3DModel

# inference func
@torch.no_grad()
def inference(args,text,video,pipe,device,save_name=""):
    save_dir = args.video_save_dir
    save_path = os.path.join(save_dir,f"{save_name}.mp4")
    # save_gt_path = os.path.join(save_dir,f"gt_{prefix}.mp4")
    sample = pipe(
        video = video,
        device = device,
        prompt = text,
        negative_prompt ="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        height = args.height,
        width = args.width, 
        num_frames = args.num_frames,
        num_inference_steps = args.num_inference_steps,
        guidance_scale = 5,
        do_cfg = False,
    )[0]
    # export_to_video(video, save_gt_path, fps=15)
    export_to_video(sample[0], save_path, fps=15)


def main(args):

    # device
    device = torch.device(f"cuda:{args.device}")

    # save dir
    os.makedirs(args.video_save_dir, exist_ok=True)

    # model
    print(args.model_id)
    vae = AutoencoderKLWan.from_pretrained(args.model_id, subfolder="vae", torch_dtype=torch.float32)
    transformer = WanTransformer3DModel.from_pretrained(args.model_id,subfolder="transformer", torch_dtype=torch.bfloat16)
    pipe = WanImageToVideoPipeline.from_pretrained(args.model_id, transformer= transformer, vae=vae, torch_dtype=torch.bfloat16)

    # load physrvg ckpt 
    if args.dit_checkpoint is not None:
        state_dict = load_file(args.dit_checkpoint)
        pipe.transformer.load_state_dict(state_dict)
    if args.lora_checkpoint is not None:
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, args.lora_checkpoint)
        pipe.transformer.set_adapter("default")  
    pipe.to(device) 

    # dataset
    train_dataset = WanV2V5BInferDataset(
        video_path = args.video_path,
        height = args.height,
        width = args.width, 
        num_frames = 5,
        start_max=32,)

    # inference
    for i in range(len(train_dataset)):
        # seed
        seed = random.randint(0, 10000) 
        set_seed(seed)

        # data
        text,video,path = train_dataset[i]

        # inference
        cur_name = os.path.basename(path)[:-4]
        now = datetime.now().strftime("%H:%M:%S")
        inference(args=args,text=text,video=video,pipe=pipe,device=device,save_name=f"{cur_name}_{now}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # args 
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--model_id",type=str, default="./models/Wan2.2-TI2V-5B-Diffusers")
    parser.add_argument("--video_save_dir", type=str, default="./output")
    parser.add_argument("--dit_checkpoint", type=str, default="./models/dit/diffusion_pytorch_model.safetensors")
    parser.add_argument("--lora_checkpoint", type=str, default="./models/lora/checkpoint")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=16)
    parser.add_argument("--video_path", type=str, default="./data/example_videos/2/video.mp4")

    # inference
    args = parser.parse_args()
    main(args)