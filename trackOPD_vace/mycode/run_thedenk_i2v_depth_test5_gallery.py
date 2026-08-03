from pathlib import Path
import argparse
import html
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

REPO = Path('/data/gaoya/agent-data/code/wan2.2-controlnet')
sys.path.insert(0, str(REPO))

from diffusers import AutoencoderKLWan, UniPCMultistepScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_video
from diffusers.pipelines.wan.pipeline_wan_i2v import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from transformers import T5TokenizerFast, UMT5EncoderModel, AutoImageProcessor, AutoModelForDepthEstimation

from wan_controlnet import WanControlnet
from wan_transformer import CustomWanTransformer3DModel
from wan_t2v_controlnet_pipeline import WanTextToVideoControlnetPipeline, prepare_controlnet_frames

DEFAULT_NEGATIVE_PROMPT = (
    'overexposed, static, blurry, subtitles, text, watermark, low quality, worst quality, '
    'jpeg artifacts, distorted geometry, inconsistent motion, wrong colors, cluttered background, '
    'deformed objects, melted objects, physically implausible motion'
)


def parse_args():
    p = argparse.ArgumentParser(description='Streaming batch Wan2.2 TI2V TheDenk depth ControlNet with first-frame conditioning and gallery.')
    p.add_argument('--json-list', default='/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt')
    p.add_argument('--out-root', default='/data/gaoya/agent-data/outputs/thedenk_i2v_depth_control_test5_30fps')
    p.add_argument('--base-model-path', default='/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers')
    p.add_argument('--controlnet-model-path', default='/data/gaoya/ckpt/TheDenk-wan2.2-ti2v-5b-controlnet-depth-v1')
    p.add_argument('--depth-model-path', default='/data/gaoya/agent-data/cache/huggingface/modelscope/fudanU123/depth_anything_small_hf')
    p.add_argument('--height', type=int, default=480)
    p.add_argument('--width', type=int, default=832)
    p.add_argument('--num-frames', type=int, default=81)
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--guidance-scale', type=float, default=5.0)
    p.add_argument('--controlnet-weight', type=float, default=0.8)
    p.add_argument('--controlnet-guidance-start', type=float, default=0.0)
    p.add_argument('--controlnet-guidance-end', type=float, default=0.8)
    p.add_argument('--controlnet-stride', type=int, default=3)
    p.add_argument('--negative-prompt', default=DEFAULT_NEGATIVE_PROMPT)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--force', action='store_true')
    return p.parse_args()


def safe_name(s):
    return ''.join(ch if (ch.isalnum() or ch in '._-') else '_' for ch in s)[:180]


def find_video_strings(obj, json_dir):
    out = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(find_video_strings(v, json_dir))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(find_video_strings(v, json_dir))
    elif isinstance(obj, str) and obj.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm')):
        p = Path(obj)
        if not p.is_absolute():
            p = (json_dir / p).resolve()
        if p.exists():
            out.append(str(p))
    return out


def find_source_video(data, json_path):
    if isinstance(data, dict) and isinstance(data.get('source_video'), str) and Path(data['source_video']).exists():
        return data['source_video']
    videos = find_video_strings(data, Path(json_path).parent)
    if not videos:
        raise RuntimeError(f'No existing video path found in {json_path}')
    ranked = []
    for v in videos:
        low = v.lower()
        score = 0
        if 'source' in low:
            score += 10
        if 'input_video' in low or 'context' in low:
            score += 3
        if 'target' in low or 'output' in low or 'mask' in low or 'depth' in low:
            score -= 8
        ranked.append((score, v))
    ranked.sort(reverse=True)
    return ranked[0][1]


def prompt_from_json(data):
    if isinstance(data, dict):
        for key in ('input_caption', 'caption', 'prompt'):
            if isinstance(data.get(key), str) and data[key].strip():
                return data[key].strip()
    return 'realistic physics simulation video, static camera, physically plausible motion, high quality'


def extend_frames(frames, n):
    if not frames:
        raise RuntimeError('No frames to extend')
    frames = list(frames)
    if len(frames) >= n:
        return frames[:n]
    cycle = frames + list(reversed(frames))
    out = []
    while len(out) < n:
        out.extend(cycle)
    return out[:n]


def load_json_items(args):
    raw_paths = [Path(x.strip()) for x in Path(args.json_list).read_text(encoding='utf-8').splitlines() if x.strip()]
    out_root = Path(args.out_root)
    seen = {}
    items = []
    duplicate_records = []
    for list_idx, json_path in enumerate(raw_paths):
        data = json.loads(json_path.read_text(encoding='utf-8'))
        source_video = find_source_video(data, json_path)
        if source_video in seen:
            duplicate_records.append({'json': str(json_path), 'same_source_as_case_dir': str(seen[source_video])})
            continue
        if args.max_cases is not None and len(items) >= args.max_cases:
            break
        case_dir = out_root / f'{len(items):02d}_{safe_name(json_path.stem)}'
        seen[source_video] = case_dir
        items.append({
            'list_index': list_idx,
            'json_path': json_path,
            'source_video': source_video,
            'prompt': prompt_from_json(data),
            'case_dir': case_dir,
        })
    return items, duplicate_records, raw_paths


def case_paths(item, args):
    case_dir = item['case_dir']
    return {
        'first_frame': case_dir / 'input_first_frame.png',
        'source_h264': case_dir / 'input_source_h264_30fps.mp4',
        'depth_h264': case_dir / 'input_depth_control_h264_30fps.mp4',
        'condition': case_dir / f'first_frame_i2v_condition_{args.num_frames}f_{args.width}x{args.height}_seed{args.seed}.pt',
        'output': case_dir / 'thedenk_i2v_depth_control_firstframe_depth_30fps.mp4',
        'manifest': case_dir / 'manifest.json',
    }


def prepare_case_assets(item, args, depth_processor, depth_model):
    paths = case_paths(item, args)
    item['case_dir'].mkdir(parents=True, exist_ok=True)
    need_source = args.force or not paths['source_h264'].exists() or not paths['first_frame'].exists()
    need_depth = args.force or not paths['depth_h264'].exists()
    if not need_source and not need_depth:
        print(f'Assets cached: {item["case_dir"].name}', flush=True)
        return

    print(f'Preparing assets: {item["case_dir"].name}', flush=True)
    source_frames = load_video(item['source_video'])
    if not source_frames:
        raise RuntimeError(f'No source frames: {item["source_video"]}')
    first_frame = source_frames[0].convert('RGB').resize((args.width, args.height))
    first_frame.save(paths['first_frame'])
    source_export_frames = [f.convert('RGB').resize((args.width, args.height)) for f in extend_frames(source_frames, args.num_frames)]
    if need_source:
        export_to_video(source_export_frames, str(paths['source_h264']), fps=args.fps)
    if need_depth:
        depth_frames = []
        with torch.no_grad():
            for frame in source_export_frames:
                inputs = depth_processor(images=frame, return_tensors='pt')
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
                pred = depth_model(**inputs).predicted_depth
                pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(args.height, args.width), mode='bicubic', align_corners=False).squeeze()
                pred = pred.detach().float().cpu()
                pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-6)
                arr = (pred.numpy() * 255.0).astype('uint8')
                rgb = np.stack([arr, arr, arr], axis=-1)
                depth_frames.append(Image.fromarray(rgb).convert('RGB'))
        export_to_video(depth_frames, str(paths['depth_h264']), fps=args.fps)
        torch.save({'path': str(paths['depth_h264'])}, item['case_dir'] / 'depth_frames_81.pt')
    del source_frames
    torch.cuda.empty_cache()


def prepare_first_frame_condition(vae, image, args, cache_path):
    if cache_path.exists() and not args.force:
        try:
            cached = torch.load(cache_path, map_location='cpu')
            return cached['latents'], cached['condition'], cached['first_frame_mask']
        except Exception as e:
            print(f'Bad condition cache, rebuilding {cache_path}: {e}', flush=True)
    print(f'Pre-encoding first-frame condition: {cache_path}', flush=True)
    device = torch.device('cpu')
    dtype = torch.float32
    vae_scale_factor_temporal = 4
    vae_scale_factor_spatial = 16
    num_latent_frames = (args.num_frames - 1) // vae_scale_factor_temporal + 1
    latent_height = args.height // vae_scale_factor_spatial
    latent_width = args.width // vae_scale_factor_spatial
    generator = torch.Generator(device='cpu').manual_seed(args.seed)
    latents = randn_tensor((1, vae.config.z_dim, num_latent_frames, latent_height, latent_width), generator=generator, device=device, dtype=dtype)
    arr = np.array(image).astype('float32') / 127.5 - 1.0
    image_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32).unsqueeze(2)
    video_condition = torch.cat([image_tensor, image_tensor.new_zeros(image_tensor.shape[0], image_tensor.shape[1], args.num_frames - 1, args.height, args.width)], dim=2).to(device=device, dtype=vae.dtype)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype)
    latent_condition = retrieve_latents(vae.encode(video_condition), sample_mode='argmax').to(dtype)
    latent_condition = (latent_condition - latents_mean) * latents_std
    first_frame_mask = torch.ones(1, 1, num_latent_frames, latent_height, latent_width, dtype=dtype, device=device)
    first_frame_mask[:, :, 0] = 0
    torch.save({'latents': latents, 'condition': latent_condition, 'first_frame_mask': first_frame_mask}, cache_path)
    return latents, latent_condition, first_frame_mask


def precompute_conditions(items, args):
    print('Precomputing first-frame conditions on CPU', flush=True)
    vae_cpu = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder='vae', torch_dtype=torch.float32)
    if hasattr(vae_cpu, 'enable_tiling'):
        vae_cpu.enable_tiling()
    if hasattr(vae_cpu, 'enable_slicing'):
        vae_cpu.enable_slicing()
    vae_cpu.to('cpu')
    for item in items:
        paths = case_paths(item, args)
        first_frame = Image.open(paths['first_frame']).convert('RGB').resize((args.width, args.height))
        prepare_first_frame_condition(vae_cpu, first_frame, args, paths['condition'])
    del vae_cpu
    torch.cuda.empty_cache()


def load_pipe(args):
    print('Loading Wan2.2 TI2V base + TheDenk depth ControlNet', flush=True)
    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder='vae', torch_dtype=torch.float32)
    text_encoder = UMT5EncoderModel.from_pretrained(args.base_model_path, subfolder='text_encoder', torch_dtype=torch.bfloat16)
    tokenizer = T5TokenizerFast.from_pretrained(args.base_model_path, subfolder='tokenizer')
    transformer = CustomWanTransformer3DModel.from_pretrained(args.base_model_path, subfolder='transformer', torch_dtype=torch.bfloat16)
    controlnet = WanControlnet.from_pretrained(args.controlnet_model_path, torch_dtype=torch.bfloat16)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.base_model_path, subfolder='scheduler')
    pipe = WanTextToVideoControlnetPipeline(tokenizer=tokenizer, text_encoder=text_encoder, transformer=transformer, vae=vae, controlnet=controlnet, scheduler=scheduler)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=5.0)
    pipe.register_to_config(expand_timesteps=True)
    return pipe


@torch.no_grad()
def run_case(pipe, item, args):
    paths = case_paths(item, args)
    if paths['output'].exists() and not args.force:
        print(f'SKIP existing output: {paths["output"]}', flush=True)
        return paths['output']
    print(f'Running inference: {item["case_dir"].name}', flush=True)
    cached = torch.load(paths['condition'], map_location='cpu')
    depth_frames = [f.convert('RGB').resize((args.width, args.height)) for f in load_video(str(paths['depth_h264']))[:args.num_frames]]
    device = torch.device('cuda')
    pipe.transformer.to(device)
    pipe.controlnet.to(device)
    pipe.text_encoder.to(device)
    pipe.vae.to('cpu')
    torch.cuda.empty_cache()
    pipe._guidance_scale = args.guidance_scale
    pipe._guidance_scale_2 = args.guidance_scale
    pipe._attention_kwargs = None
    pipe._current_timestep = None
    pipe._interrupt = False
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=item['prompt'], negative_prompt=args.negative_prompt, do_classifier_free_guidance=pipe.do_classifier_free_guidance,
        num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None, max_sequence_length=512, device=device,
    )
    transformer_dtype = pipe.transformer.dtype
    prompt_embeds = prompt_embeds.to(transformer_dtype)
    negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
    pipe.text_encoder.to('cpu')
    torch.cuda.empty_cache()
    pipe.scheduler.set_timesteps(args.steps, device=device)
    timesteps = pipe.scheduler.timesteps
    latents = cached['latents'].to(device=device, dtype=torch.float32)
    condition = cached['condition'].to(device=device, dtype=torch.float32)
    first_frame_mask = cached['first_frame_mask'].to(device=device, dtype=torch.float32)
    controlnet_latents = prepare_controlnet_frames(depth_frames, args.height, args.width, dtype=pipe.controlnet.dtype, device=pipe.controlnet.device)
    del depth_frames
    num_warmup_steps = len(timesteps) - args.steps * pipe.scheduler.order
    with pipe.progress_bar(total=args.steps) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (1 - first_frame_mask) * condition + first_frame_mask * latents
            latent_model_input = latent_model_input.to(transformer_dtype)
            timestep = (first_frame_mask[0][0][:, ::2, ::2] * t).flatten().unsqueeze(0).expand(latents.shape[0], -1)
            current_sampling_percent = i / len(timesteps)
            controlnet_states = None
            if args.controlnet_guidance_start <= current_sampling_percent < args.controlnet_guidance_end:
                controlnet_states = pipe.controlnet(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds, attention_kwargs=None, controlnet_states=controlnet_latents, return_dict=False)[0]
                if isinstance(controlnet_states, (tuple, list)):
                    controlnet_states = [x.to(dtype=transformer_dtype) for x in controlnet_states]
                else:
                    controlnet_states = controlnet_states.to(dtype=transformer_dtype)
            kwargs = dict(hidden_states=latent_model_input, timestep=timestep, encoder_hidden_states=prompt_embeds, controlnet_states=controlnet_states, controlnet_weight=args.controlnet_weight, controlnet_stride=args.controlnet_stride, attention_kwargs=None, return_dict=False)
            with pipe.transformer.cache_context('cond'):
                noise_pred = pipe.transformer(**kwargs)[0]
            if pipe.do_classifier_free_guidance:
                uncond_kwargs = dict(kwargs)
                uncond_kwargs['encoder_hidden_states'] = negative_prompt_embeds
                with pipe.transformer.cache_context('uncond'):
                    noise_uncond = pipe.transformer(**uncond_kwargs)[0]
                noise_pred = noise_uncond + args.guidance_scale * (noise_pred - noise_uncond)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0):
                progress_bar.update()
    latents = (1 - first_frame_mask) * condition + first_frame_mask * latents
    pipe.transformer.to('cpu')
    pipe.controlnet.to('cpu')
    torch.cuda.empty_cache()
    pipe.vae.to(device)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = torch.tensor(pipe.vae.config.latents_mean).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(1, pipe.vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
    video = pipe.vae.decode(latents / latents_std + latents_mean, return_dict=False)[0]
    frames = pipe.video_processor.postprocess_video(video, output_type='pil')[0]
    export_to_video(frames, str(paths['output']), fps=args.fps)
    pipe.vae.to('cpu')
    del cached, latents, condition, first_frame_mask, frames
    torch.cuda.empty_cache()
    print(f'OUTPUT={paths["output"]}', flush=True)
    return paths['output']


def record_for(item, args):
    paths = case_paths(item, args)
    return {
        'case_name': item['case_dir'].name,
        'json': str(item['json_path']),
        'source_video': item['source_video'],
        'prompt': item['prompt'],
        'negative_prompt': args.negative_prompt,
        'first_frame': str(paths['first_frame']),
        'source_h264_30fps': str(paths['source_h264']),
        'depth_h264_30fps': str(paths['depth_h264']),
        'output': str(paths['output']) if paths['output'].exists() else None,
        'fps': args.fps,
        'num_frames': args.num_frames,
        'steps': args.steps,
        'seed': args.seed,
        'base_model_path': args.base_model_path,
        'controlnet_model_path': args.controlnet_model_path,
        'depth_extractor': args.depth_model_path,
    }


def build_gallery(out_root, records, args, duplicate_records):
    def rel(p):
        return os.path.relpath(str(p), str(out_root))
    cards = []
    for r in records:
        if not r.get('output'):
            continue
        cards.append(f'''<section class="case"><h2>{html.escape(r['case_name'])}</h2><p><b>prompt</b>: {html.escape(r['prompt'])}</p><div class="grid"><div><h3>Source video 30fps</h3><video src="{html.escape(rel(r['source_h264_30fps']))}" controls muted loop></video></div><div><h3>Depth control 30fps</h3><video src="{html.escape(rel(r['depth_h264_30fps']))}" controls muted loop></video></div><div><h3>First frame condition</h3><img src="{html.escape(rel(r['first_frame']))}" /></div><div><h3>TheDenk output</h3><video src="{html.escape(rel(r['output']))}" controls muted loop></video></div></div><details><summary>metadata</summary><pre>{html.escape(json.dumps(r, ensure_ascii=False, indent=2))}</pre></details></section>''')
    dup_html = '<h2>Skipped duplicate source videos</h2><pre>' + html.escape(json.dumps(duplicate_records, ensure_ascii=False, indent=2)) + '</pre>' if duplicate_records else ''
    doc = f'''<!doctype html><html><head><meta charset="utf-8"><title>TheDenk I2V Depth ControlNet test_5</title><style>body{{margin:0;background:#10120f;color:#eee;font-family:ui-sans-serif,system-ui,sans-serif}}header{{padding:24px 32px;background:linear-gradient(135deg,#243122,#241f16);border-bottom:1px solid #424536;position:sticky;top:0;z-index:2}}h1{{margin:0 0 8px;color:#f2dc9b}}p{{color:#d7d7d0;line-height:1.45}}.case{{margin:24px 32px;padding:18px;background:#1b1e19;border:1px solid #3a4234;border-radius:18px}}.case h2{{margin-top:0;color:#9fd3b0}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(260px,1fr));gap:16px}}.grid div{{background:#0b0c0a;border:1px solid #32382e;border-radius:14px;padding:12px}}h3{{margin:0 0 10px;color:#e0bf73;font-size:15px}}video,img{{width:100%;max-height:430px;object-fit:contain;background:#000;border-radius:10px}}pre{{white-space:pre-wrap;overflow:auto;color:#ddd}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}header,.case{{margin-left:12px;margin-right:12px}}}}</style></head><body><header><h1>TheDenk Wan2.2 TI2V 5B Depth ControlNet: first frame + depth + prompt + negative</h1><p>Base: {html.escape(args.base_model_path)}<br>ControlNet: {html.escape(args.controlnet_model_path)}<br>Depth extractor: {html.escape(args.depth_model_path)}<br>{args.width}x{args.height}, {args.num_frames} frames, {args.fps} fps, {args.steps} steps, seed {args.seed}</p></header>{dup_html}{''.join(cards)}</body></html>'''
    (out_root / 'index.html').write_text(doc, encoding='utf-8')


def write_manifest(out_root, records, args, duplicate_records, raw_paths):
    (out_root / 'manifest.json').write_text(json.dumps({'script': __file__, 'json_list': args.json_list, 'raw_json_count': len(raw_paths), 'unique_source_count': len(records), 'duplicate_records': duplicate_records, 'params': vars(args), 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    items, duplicate_records, raw_paths = load_json_items(args)
    print(f'Loaded {len(raw_paths)} JSON rows; {len(items)} unique source videos; {len(duplicate_records)} duplicates skipped.', flush=True)

    print('Preparing cached inputs/depths streaming', flush=True)
    depth_processor = AutoImageProcessor.from_pretrained(args.depth_model_path)
    depth_model = AutoModelForDepthEstimation.from_pretrained(args.depth_model_path).to('cuda').eval()
    for item in items:
        prepare_case_assets(item, args, depth_processor, depth_model)
    depth_model.to('cpu')
    del depth_model, depth_processor
    torch.cuda.empty_cache()

    precompute_conditions(items, args)
    pipe = load_pipe(args)

    records = []
    for item in items:
        paths = case_paths(item, args)
        run_case(pipe, item, args)
        rec = record_for(item, args)
        paths['manifest'].write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding='utf-8')
        records.append(rec)
        write_manifest(out_root, records, args, duplicate_records, raw_paths)
        build_gallery(out_root, records, args, duplicate_records)
    write_manifest(out_root, records, args, duplicate_records, raw_paths)
    build_gallery(out_root, records, args, duplicate_records)
    print(f'GALLERY={out_root / "index.html"}', flush=True)
    print(f'MANIFEST={out_root / "manifest.json"}', flush=True)
    print('DONE_BATCH', flush=True)


if __name__ == '__main__':
    main()
