"""Replay, render and publish a representative 20-case full-video preview.

The script never changes the 4200-case bank.  It replays each selected case
from its original blueprint and seed, verifies the packed tensors against the
bank, renders all 49 RGB frames, and copies only RGB0-7 into a separate model
input bundle.  Rendering is resumable: an existing verified case is skipped,
while an incomplete case is rejected rather than silently overwritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from prepare_small_trial import BANK, ROOT, actor_metadata, dump, sha

BLENDER = Path('/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender')
RENDERER = Path('/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/render_physv_cycles.py')
SEED = 2026091905
FRAME_COUNT = 49
OBSERVED_COUNT = 8
WIDTH = 1280
HEIGHT = 720
SAMPLES = 64
VIDEO_FPS = 30
GPU_UUIDS = (
    'GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5',  # physical GPU0
    'GPU-74468333-11e0-dfa6-ef16-584a42fa5a02',  # physical GPU1
    'GPU-994fb224-27dc-b1e0-759d-0226b0c0d775',  # physical GPU2
    'GPU-05862376-967b-f129-f129-835daf8158cf',  # physical GPU3
)


def selected_records(manifest: dict) -> list[dict]:
    """Take 2 per split/family (18) plus two deterministic extras.

    The extra cases make the 20-case preview slightly more representative while
    retaining all three splits and all three physical families.
    """
    chosen: list[dict] = []
    pools: dict[tuple[str, str], list[tuple[int, dict]]] = {}
    for split in ('train', 'val', 'test'):
        for family in ('barrier', 'door', 'gap'):
            pool = [(i, row) for i, row in enumerate(manifest['records'])
                    if row['split'] == split and row['family'] == family]
            pool.sort(key=lambda pair: hashlib.sha256(
                f'{SEED}:{pair[1]["key"]}'.encode()).hexdigest())
            pools[(split, family)] = pool
            chosen.extend(dict(row, bank_index=i) for i, row in pool[:2])
    for split, family in (('train', 'barrier'), ('val', 'gap')):
        i, row = pools[(split, family)][2]
        chosen.append(dict(row, bank_index=i))
    if len(chosen) != 20 or len({row['key'] for row in chosen}) != 20:
        raise ValueError('Selection is not exactly 20 unique records')
    return chosen


def verify_source(manifest_path: Path, manifest: dict) -> None:
    for path, expected in manifest['source_sha256'].items():
        if sha(path) != expected:
            raise ValueError(f'Original generator source changed: {path}')
    if sha(manifest_path) != manifest['_manifest_sha256']:
        raise ValueError('Manifest changed while preparing the preview')
    if sha(BANK / 'data' / 'bank.safetensors') != manifest['_bank_sha256']:
        raise ValueError('Packed physical bank changed while preparing the preview')


def make_trajectory(pos: np.ndarray, quat: np.ndarray, names: list[str], count: int) -> dict:
    # Blender expects quaternion order [w, x, y, z]; the simulator stores xyzw.
    value = dict(object_names=names, frame_times_s=(np.arange(count) / 30.0).tolist())
    for j, name in enumerate(names):
        value[f'{name}_positions'] = pos[:count, j].tolist()
        value[f'{name}_rotations'] = quat[:count, j][:, [3, 0, 1, 2]].tolist()
    return value


def replay_case(output: Path, row: dict, bank) -> None:
    import sys
    sys.path.insert(0, str(BANK))
    import experiment as e

    key = row['key']
    sample = output / 'samples' / key
    receipt = sample / 'replay.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        for name, digest in saved['output_sha256'].items():
            if sha(sample / name) != digest:
                raise ValueError(f'Replay artifact changed: {key}/{name}')
        return
    if sample.exists():
        raise FileExistsError(f'Incomplete replay requires inspection: {sample}')

    blueprint_path = BANK / 'data' / 'blueprints' / f'{key}.json'
    original = json.loads(blueprint_path.read_text())
    case = e.make_case(row['family'], original['sample_key'], np.asarray(row['u']))
    if json.loads(json.dumps(asdict(case.blueprint))) != original:
        raise ValueError(f'Blueprint mismatch: {key}')
    objects, pos, vel, quat, corrections = e.audit.replay(case, row['simulation_seed'])
    packed = e.pack(case, objects, pos, vel, quat)
    context_hash = hashlib.sha256(packed['node'].numpy().tobytes()).hexdigest()
    motion_hash = hashlib.sha256(
        packed['target_position'].numpy().tobytes() + packed['node'].numpy().tobytes()).hexdigest()
    if context_hash != row['context_hash'] or motion_hash != row['motion_hash']:
        raise ValueError(f'Replay fingerprint mismatch: {key}')
    for name in ('node', 'target_position', 'last_position', 'history_dt', 'future_dt'):
        if not __import__('torch').equal(packed[name], bank.get_slice(name)[row['bank_index']]):
            raise ValueError(f'Bank mismatch: {key}/{name}')
    if corrections != row['source_rebound_corrections']:
        raise ValueError(f'Rebound correction count changed: {key}')

    camera = case.blueprint.camera
    metadata = dict(
        schema_version='visual_bank20_replay_v1', sample_id=key,
        family_key=case.family_key, seed=row['simulation_seed'], split=row['split'],
        simulation=dict(fps=30, sim_hz=e.g.SIM_HZ, frame_count=FRAME_COUNT,
                        pre_roll_s=case.blueprint.pre_roll_s),
        camera=dict(intrinsics=dict(yfov_deg=camera.yfov_deg),
                    extrinsics=dict(eye=camera.eye, target=camera.target, up=camera.up)),
        actors={o.name: actor_metadata(o) for o in objects},
        scenario_spec=case.blueprint.metadata, source=str(e.audit.SOURCE),
        rebound_corrections=corrections,
    )
    sample.mkdir(parents=True)
    (sample / 'raw').mkdir()
    dump(sample / 'metadata.json', metadata)
    np.savez_compressed(sample / 'raw' / 'states_xyzw.npz', positions=pos,
                        linear_velocities=vel, quats=quat,
                        object_names=np.array([o.name for o in objects]),
                        frame_times=np.arange(FRAME_COUNT) / 30.0)
    dump(sample / 'raw' / 'observed_trajectories.json',
         make_trajectory(pos, quat, [o.name for o in objects], OBSERVED_COUNT))
    dump(sample / 'raw' / 'full_trajectories.json',
         make_trajectory(pos, quat, [o.name for o in objects], FRAME_COUNT))
    names = ('metadata.json', 'raw/states_xyzw.npz', 'raw/observed_trajectories.json',
             'raw/full_trajectories.json')
    dump(receipt, dict(key=key, bank_index=row['bank_index'], context_hash=context_hash,
                      motion_hash=motion_hash, exact_replay=True,
                      output_sha256={name: sha(sample / name) for name in names}))


def render_case(output: Path, row: dict, gpu_uuid: str) -> dict:
    key = row['key']
    sample = output / 'samples' / key
    render_root = output / 'renders' / key
    frames = render_root / 'cycles_frames'
    receipt = render_root / 'execution.json'
    required = [frames / f'frame_{i:04d}.png' for i in range(1, FRAME_COUNT + 1)]
    metadata_path = frames / 'render_metadata.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['returncode'] != 0 or saved['renderer_sha256'] != sha(RENDERER):
            raise ValueError(f'Failed or changed render: {key}')
        if not all(path.is_file() for path in required) or not metadata_path.is_file():
            raise ValueError(f'Incomplete existing render: {key}')
    else:
        if render_root.exists():
            raise FileExistsError(f'Incomplete render requires inspection: {render_root}')
        render_root.mkdir(parents=True)
        argv = [str(BLENDER), '-b', '-t', '4', '--python', str(RENDERER), '--',
                '--sample-dir', str(sample), '--trajectory-json',
                str(sample / 'raw' / 'full_trajectories.json'), '--output-dir', str(frames),
                '--width', str(WIDTH), '--height', str(HEIGHT), '--samples', str(SAMPLES),
                '--engine', 'CYCLES', '--device', 'CUDA']
        dump(render_root / 'command.json', dict(argv=argv, gpu_uuid=gpu_uuid,
             timeout_seconds=900, frame_count=FRAME_COUNT,
             renderer_sha256=sha(RENDERER), trajectory_sha256=sha(sample / 'raw' / 'full_trajectories.json')))
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_uuid
        env.setdefault('TMPDIR', '/data/gaoya/agent-data/cache/sg-overlay/tmp')
        started = time.monotonic()
        with (render_root / 'render.log').open('w') as stream:
            proc = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT,
                                  check=False, timeout=900, env=env)
        saved = dict(returncode=proc.returncode, elapsed_seconds=time.monotonic() - started,
                     renderer_sha256=sha(RENDERER), gpu_uuid=gpu_uuid,
                     output_sha256={str(path.relative_to(render_root)): sha(path)
                                    for path in required + [metadata_path] if path.is_file()})
        dump(receipt, saved)
        if proc.returncode != 0 or not all(path.is_file() for path in required) or not metadata_path.is_file():
            raise RuntimeError(f'Render failed: {key}; see {render_root}/render.log')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('frame_count') != FRAME_COUNT:
        raise ValueError(f'Expected 49 rendered frames, got {metadata.get("frame_count")}: {key}')

    context = output / 'context_inputs' / key
    context.mkdir(parents=True, exist_ok=True)
    for i in range(OBSERVED_COUNT):
        source = frames / f'frame_{i + 1:04d}.png'
        target = context / f'rgb_{i:02d}.png'
        if target.exists():
            if sha(target) != sha(source):
                raise ValueError(f'Context frame changed: {key}/rgb_{i:02d}.png')
        else:
            shutil.copyfile(source, target)
    input_json = context / 'input.json'
    if not input_json.exists():
        from prepare_context import pixel_hash, read_rgb
        rgb = read_rgb([context / f'rgb_{i:02d}.png' for i in range(OBSERVED_COUNT)])
        dump(input_json, dict(schema='visual_context_rgb8_v1',
                              frames=[f'rgb_{i:02d}.png' for i in range(OBSERVED_COUNT)],
                              observed_indices=list(range(OBSERVED_COUNT)),
                              time_s=(np.arange(OBSERVED_COUNT) / 30.0).tolist(),
                              shape=list(rgb.shape), pixels_sha256=pixel_hash(rgb)))
    return dict(key=key, family=row['family'], split=row['split'],
                frames=[f'renders/{key}/cycles_frames/frame_{i:04d}.png'
                        for i in range(1, FRAME_COUNT + 1)],
                context_frames=[f'context_inputs/{key}/rgb_{i:02d}.png'
                                for i in range(OBSERVED_COUNT)],
                render_metadata=f'renders/{key}/cycles_frames/render_metadata.json',
                replay_exact=True, full_frame_count=FRAME_COUNT)


def encode_case(output: Path, row: dict) -> dict:
    """Encode the already-validated PNG sequence without touching rendering."""
    import imageio_ffmpeg

    key = row['key']
    source = output / 'renders' / key / 'cycles_frames'
    videos = output / 'videos'
    videos.mkdir(parents=True, exist_ok=True)
    target = videos / f'{key}.mp4'
    receipt = videos / f'{key}.json'
    frames = [source / f'frame_{i:04d}.png' for i in range(1, FRAME_COUNT + 1)]
    if not all(path.is_file() for path in frames):
        raise FileNotFoundError(f'Missing rendered frames for {key}')
    if target.is_file() and target.stat().st_size > 0 and receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get('frame_count') == FRAME_COUNT and saved.get('fps') == VIDEO_FPS:
            return dict(row, video=f'videos/{key}.mp4', video_frame_count=FRAME_COUNT,
                        video_fps=VIDEO_FPS)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    temporary = target.with_suffix('.tmp.mp4')
    if temporary.exists():
        temporary.unlink()
    argv = [ffmpeg, '-y', '-hide_banner', '-loglevel', 'error', '-framerate', str(VIDEO_FPS),
            '-start_number', '1', '-i', str(source / 'frame_%04d.png'),
            '-frames:v', str(FRAME_COUNT), '-c:v', 'libx264', '-preset', 'fast',
            '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary)]
    started = time.monotonic()
    proc = subprocess.run(argv, check=False, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0 or not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f'Video encoding failed for {key}: {proc.stderr[-500:]}')
    temporary.replace(target)
    dump(receipt, dict(schema='visual_bank20_video_v1', key=key, frame_count=FRAME_COUNT,
                       fps=VIDEO_FPS, resolution=[WIDTH, HEIGHT], codec='h264', crf=18,
                       elapsed_seconds=time.monotonic() - started, output_sha256=sha(target)))
    return dict(row, video=f'videos/{key}.mp4', video_frame_count=FRAME_COUNT,
                video_fps=VIDEO_FPS)


def build_page(output: Path, records: list[dict]) -> None:
    page = output / 'view'
    page.mkdir(exist_ok=True)
    for name in ('renders', 'context_inputs', 'videos'):
        link = page / name
        target = output / name
        if link.is_symlink():
            if link.resolve() != target.resolve():
                raise ValueError(f'Unexpected page asset link: {link.resolve()}')
        elif link.exists():
            raise FileExistsError(f'Unexpected page asset path: {link}')
        else:
            link.symlink_to(target, target_is_directory=True)
    payload = dict(schema='visual_bank20_full_render_v1', generated='2026-09-19',
                   total=len(records), frame_count=FRAME_COUNT, observed_frames=OBSERVED_COUNT,
                   resolution=[WIDTH, HEIGHT], samples=SAMPLES, records=records,
                   note='Full 49-frame Cycles/PBR render; only RGB0-7 are exported as model input.')
    dump(page / 'data.json', payload)
    html = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bank20 full renders</title><style>
:root{color-scheme:dark;background:#101417;color:#edf2f3;font:14px system-ui,sans-serif}body{margin:0;padding:22px;max-width:1500px;margin-inline:auto}h1{font-size:24px;margin:0 0 6px}p{color:#aab8bd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:18px}.card{background:#182126;border:1px solid #314149;padding:12px;border-radius:7px}.meta{display:flex;justify-content:space-between;gap:8px;font-size:13px;color:#c5d1d5;margin-bottom:8px}.frame,.video{width:100%;aspect-ratio:16/9;object-fit:contain;background:#050708;display:block}.video{margin-bottom:8px}.controls{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;margin-top:8px}.controls input{width:100%}.tag{color:#78c9b0}.links{font-size:12px;margin-top:7px}.links a{color:#8bc8ff;margin-right:10px}.caption{font-size:12px;color:#9daeb3;margin-top:5px}</style></head>
<body><h1>20-case full render preview</h1><p>49-frame Cycles/PBR render at 1280x720. The player shows the complete 30 FPS sequence. Slider frame 0-7 is the exact RGB context exported for the predictor; frames 8-48 are future ground-truth visualization only.</p><div id="grid" class="grid"></div>
<script>
fetch('data.json').then(r=>r.json()).then(data=>{const grid=document.querySelector('#grid');
for(const rec of data.records){const card=document.createElement('section');card.className='card';
const img=document.createElement('img');img.className='frame';img.loading='lazy';img.src=rec.frames[0];
const player=document.createElement('video');player.className='video';player.controls=true;player.preload='metadata';player.playsInline=true;player.src=rec.video||'';
const meta=document.createElement('div');meta.className='meta';meta.innerHTML=`<span>${rec.key}</span><span class="tag">${rec.family} · ${rec.split}</span>`;
const controls=document.createElement('div');controls.className='controls';const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=48;slider.value=0;slider.step=1;
const label=document.createElement('output');label.textContent='RGB0';slider.oninput=()=>{const i=Number(slider.value);img.src=rec.frames[i];label.textContent=`RGB${i}${i<8?' · predictor input':''}`};
const links=document.createElement('div');links.className='links';links.innerHTML=`<a href="${rec.context_frames[0]}" target="_blank">RGB0</a><a href="${rec.context_frames[7]}" target="_blank">RGB7</a><a href="${rec.frames[48]}" target="_blank">RGB48</a><a href="${rec.render_metadata}" target="_blank">metadata</a>`;
const caption=document.createElement('div');caption.className='caption';caption.textContent='Video: RGB0-48 at 30 FPS · Frame 0-7: model context · Frame 8-48: visualization-only future truth';controls.append(slider,label);card.append(meta,player,img,controls,links,caption);grid.append(card)}});
</script></body></html>'''
    (page / 'index.html').write_text(html)
    hub = Path('/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/generic-full-2175-lora1500-lineage/p4-v2-bank20-full-rendered')
    if hub.is_symlink():
        if hub.resolve() != page.resolve():
            raise ValueError(f'Unexpected existing hub target: {hub.resolve()}')
    elif hub.exists():
        raise FileExistsError(f'Unexpected existing hub path: {hub}')
    else:
        hub.symlink_to(page, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'render', 'video', 'all'))
    parser.add_argument('--output', type=Path, default=ROOT / 'bank20_full_render_20260919')
    parser.add_argument('--limit', type=int, default=0, help='Render only first N selected cases')
    parser.add_argument('--gpus', nargs='*', default=list(GPU_UUIDS))
    args = parser.parse_args()
    if any(uuid == 'GPU-4a8abb69-6a43-4b79-5713-31979b8d6d75' for uuid in args.gpus):
        raise ValueError('GPU4 is forbidden')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = BANK / 'data' / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['_manifest_sha256'] = sha(manifest_path)
    manifest['_bank_sha256'] = sha(BANK / 'data' / 'bank.safetensors')
    verify_source(manifest_path, manifest)
    records = selected_records(manifest)
    dump(output / 'selection.json', dict(seed=SEED, source_manifest_sha256=manifest['_manifest_sha256'],
                                        source_bank_sha256=manifest['_bank_sha256'], records=records,
                                        selection='2 per split x family plus train/barrier and val/gap extras'))
    if args.stage in ('prepare', 'all'):
        import sys
        sys.path.insert(0, str(BANK))
        import experiment as e
        e.g.SIM_DURATION_S = FRAME_COUNT / 30
        import torch
        torch.set_num_threads(2)
        from safetensors import safe_open
        with safe_open(str(BANK / 'data' / 'bank.safetensors'), framework='pt', device='cpu') as bank:
            for row in records:
                replay_case(output, row, bank)
                print('REPLAY_OK', row['key'], flush=True)
    if args.stage in ('render', 'all'):
        to_render = records[:args.limit or None]
        rendered = []
        with ThreadPoolExecutor(max_workers=min(len(args.gpus), len(to_render))) as pool:
            futures = {pool.submit(render_case, output, row, args.gpus[i % len(args.gpus)]): row
                       for i, row in enumerate(to_render)}
            for future in as_completed(futures):
                row = futures[future]
                rendered.append(future.result())
                print('RENDER_OK', row['key'], flush=True)
        rendered.sort(key=lambda row: row['key'])
        dump(output / 'render_manifest.json', dict(schema='visual_bank20_full_render_v1',
             source_selection_sha256=sha(output / 'selection.json'), total=len(rendered), records=rendered))
        build_page(output, rendered)
        dump(output / 'README.md', dict(url='http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-bank20-full-rendered/',
             total=len(rendered), full_frames=FRAME_COUNT, context_frames=OBSERVED_COUNT,
             resolution=[WIDTH, HEIGHT], samples=SAMPLES, gpu_uuids=list(args.gpus),
             note='All 49 frames rendered from exact replay; RGB0-7 copied to context_inputs and only those are model input.'))
        print('PAGE http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-bank20-full-rendered/', flush=True)
    if args.stage in ('video', 'all'):
        manifest = json.loads((output / 'render_manifest.json').read_text())
        rows = manifest['records']
        encoded = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(encode_case, output, row): row for row in rows}
            for future in as_completed(futures):
                row = futures[future]
                encoded.append(future.result())
                print('VIDEO_OK', row['key'], flush=True)
        encoded.sort(key=lambda row: row['key'])
        dump(output / 'render_manifest.json', dict(manifest, records=encoded,
             videos=True, video_codec='h264', video_fps=VIDEO_FPS))
        build_page(output, encoded)
        print('VIDEO_PAGE http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-bank20-full-rendered/', flush=True)


if __name__ == '__main__':
    main()
