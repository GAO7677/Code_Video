"""Prepare anonymous old-pilot observations; never export GT calibration/state."""
import json,shutil
from pathlib import Path
from context_rgb_pybullet_common import dump_json,sha256_file
BASE=Path('/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1')
OUT=Path('/data/gaoya/agent-data/outputs/context_generic_pilot36_20260922_v1')
if __name__=='__main__':
    OUT.mkdir(exist_ok=False);(OUT/'vggt_cache').mkdir()
    records=json.loads((BASE/'vision_inputs/manifest.json').read_text())['records']
    provenance={r['case_id']:r for r in json.loads((BASE/'vision_v1/inference_report.json').read_text())['vggt']['records']}
    mapping=[]
    for i,r in enumerate(records):
        key=f'clip_{i:02d}';folder=BASE/'vision_inputs'/r['input_dir'];dest=OUT/'inputs'/key;dest.mkdir(parents=True)
        for t in range(8):shutil.copyfile(folder/f'rgb_{t:02d}.png',dest/f'rgb_{t:02d}.png')
        payload=json.loads((folder/'input.json').read_text());dump_json(dest/'timestamps.json',{'time_s':payload['time_s']})
        raw=BASE/'vision_v1/raw_vggt'/f"{r['case_id']}.npz"
        if sha256_file(raw)!=provenance[r['case_id']]['sha256']:raise ValueError('raw cache hash mismatch')
        shutil.copyfile(raw,OUT/'vggt_cache'/f'{key}.npz');mapping.append({'id':key,'case_id':r['case_id'],'family':r['family'],'group_id':r['group_id']})
    dump_json(OUT/'evaluation_mapping.json',mapping)
    protocol=json.loads(Path('/data/gaoya/agent-data/outputs/context_generic_six_20260922_v1/protocol.json').read_text())
    protocol.update(selection='all 36 existing v3 cases, fixed manifest order, no replacement',reuse_frozen_anonymous_vggt=True,
                    gravity='explicit lower_image_plane_prior, observation admission only; no GT gate at inference')
    protocol.pop('selected_ordinals',None);protocol.pop('excluded_visual_ordinals',None);protocol.pop('screened_only',None)
    dump_json(OUT/'protocol.json',protocol)
    paths=[OUT/'protocol.json',*sorted((OUT/'inputs').glob('*/*')),*sorted((OUT/'vggt_cache').glob('*.npz'))]
    dump_json(OUT/'input_freeze.json',{str(p.relative_to(OUT)):sha256_file(p) for p in paths})
