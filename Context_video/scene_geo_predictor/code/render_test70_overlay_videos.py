"""Post-freeze display only: six H.264 stage videos per test70 case."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from context_rgb_pybullet_common import dump_json, sha256_file
from run_grounded_generic_pilot36 import verify


def line(im, pts, color, width=2):
    for a, b in zip(pts[:-1], pts[1:]):
        if np.isfinite([a, b]).all() and np.max(np.abs([a, b])) < 1e7:
            cv2.line(im, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, width, cv2.LINE_AA)


def main(root):
    cv2.setNumThreads(2)
    verify(root, 'rollout_freeze.json')
    data = json.loads((root / 'viewer_data.json').read_text())
    mapping = {r['id']: r for r in json.loads((root / 'evaluation_mapping.json').read_text())}
    out = root / 'overlay_videos_v2'
    out.mkdir(exist_ok=False)
    records = []
    for r in data['records']:
        cid = r['id']
        source = Path(mapping[cid]['source_sample']) / 'videos/rgb_cycles.mp4'
        cap = cv2.VideoCapture(str(source))
        frames = []
        for _ in range(49):
            ok, im = cap.read()
            if not ok:
                raise ValueError(f'{cid}: fewer than 49 source RGB frames')
            frames.append(im)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if abs(fps-30) > .01:
            raise ValueError(f'{cid}: source FPS mismatch')
        # Verify display video belongs to the exact inference context.
        errors = []
        psnr = []
        context = cv2.VideoCapture(mapping[cid]['source_context'])
        for t in range(8):
            rgb = cv2.imread(str(root / 'inputs' / cid / f'rgb_{t:02d}.png'))
            if rgb.shape != frames[t].shape:
                raise ValueError(f'{cid}: source resolution mismatch')
            ok, original_context = context.read()
            if not ok or not np.array_equal(original_context, rgb):
                raise ValueError(f'{cid}: frozen context/video identity mismatch')
            errors.append(float(np.mean(np.abs(rgb.astype(float)-frames[t]))))
            psnr.append(float(cv2.PSNR(rgb, frames[t])))
        context.release()
        # Context is independently re-encoded; exact equality is checked above
        # against its own video. Full-video cross-check allows compression noise.
        if min(psnr) < 30:
            raise ValueError(f'{cid}: source context mismatch {errors}')
        frames = [cv2.resize(im, (640, 360)) for im in frames]
        folder = out / cid
        folder.mkdir()
        for stage in ['context', 'mask', 'state', 'depth', 'geometry', 'rollout']:
            count = 49 if stage in ['geometry', 'rollout'] else 8
            path = folder / f'{stage}.mp4'
            cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '640x360', '-r', '30', '-i', '-',
                   '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '19',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for t in range(count):
                im = frames[t if stage != 'geometry' else 7].copy()
                if stage == 'depth':
                    im = cv2.resize(cv2.imread(str(root / 'diagnostics' / cid / f'depth_{t}.png')), (640, 360))
                if stage == 'geometry':
                    im = cv2.resize(cv2.imread(str(root / 'diagnostics' / cid / 'mesh.png')), (640, 360))
                if stage == 'mask' or (stage == 'rollout' and t < 8):
                    for c in r['mask_contours'][t]:
                        line(im, c+[c[0]], (255, 190, 50))
                if stage == 'state' and r.get('center_uv'):
                    line(im, r['center_uv'][:t+1], (50, 220, 255))
                if stage == 'rollout' and t >= 8:
                    for key, color in [('gt_future_uv', (100, 230, 70)), ('cv_eval_uv', (250, 180, 50)), ('pred_eval_uv', (60, 70, 250))]:
                        pts = r['evaluation'].get(key, [])[:t-7]
                        line(im, pts, (15, 15, 15), 4)
                        line(im, pts, color)
                        if pts and np.isfinite(pts[-1]).all() and np.max(np.abs(pts[-1])) < 1e7:
                            cv2.circle(im, tuple(np.rint(pts[-1]).astype(int)), 4, color, -1, cv2.LINE_AA)
                cv2.rectangle(im, (0, 0), (640, 40), (25, 32, 30), -1)
                label = f'{cid} | {stage} | RGB{7 if stage == "geometry" else t} | {r["rollout_status"]}'
                cv2.putText(im, label, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1, cv2.LINE_AA)
                note = 'Future RGB = evaluation only | GT green / CV blue / D red' if stage == 'rollout' else 'Context-only estimate | failed fit = diagnostic only'
                if stage == 'geometry':
                    note = 'RGB7 HOLD | visible mesh + yellow inferred plane | unvalidated'
                cv2.putText(im, note, (8, 32), cv2.FONT_HERSHEY_SIMPLEX, .37, (200, 230, 230), 1, cv2.LINE_AA)
                proc.stdin.write(im.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f'Video encoding failed: {path}')
        records.append({'id': cid, 'source': str(source), 'source_sha256': sha256_file(source),
                        'context_max_MAE': max(errors), 'full_video_min_PSNR': min(psnr),
                        'context_exact_match': True, 'fps': 30, 'videos': 6})
        print('VIDEO', cid, flush=True)
    dump_json(out / 'manifest.json', {'status': 'EXECUTED', 'cases': records,
        'future_rgb': 'evaluation background only; never fed into estimation',
        'projection': 'validated GT renderer camera for GT/CV/D evaluation overlays',
        'rollout_frames': 49, 'context_stage_frames': 8, 'geometry': 'RGB7 hold',
        'codec': 'H264 yuv420p faststart', 'cpu_threads': 2})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    main(p.parse_args().root)
