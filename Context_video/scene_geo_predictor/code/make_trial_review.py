"""Contact sheets for inspecting all eight observed masks, not future labels."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from prepare_small_trial import ROOT, dump, sha
from prepare_context import load_model_input


def sheets(root, output, limit):
    if output.exists():
        raise FileExistsError(output)
    reviewed = set()
    for path in (root/'reviews').glob('*/approval.json'):
        reviewed.update(json.loads(path.read_text())['approved_cases'])
    records = json.loads((root/'manifest.json').read_text())['records']
    ready = [r for r in records if r['key'] not in reviewed and
             (root/'observed_masks'/r['key']/'report.json').exists() and
             (root/'raw_probe'/r['key']/'report.json').exists()][:limit]
    if not ready:
        raise ValueError('No unreviewed ready cases')
    output.mkdir(parents=True)
    pages, proofs = [], []
    for offset in range(0,len(ready),10):
        rows = ready[offset:offset+10]
        canvas = Image.new('RGB',(1280,180*len(rows)),'white')
        draw = ImageDraw.Draw(canvas)
        for i,row in enumerate(rows):
            key = row['key']
            path = root/'observed_masks'/key/'observed_masks.npz'
            with np.load(path,allow_pickle=False) as data:
                masks, accepted = data['per_object'][:,0], data['accepted'][:,0]
            with np.load(root/'observed_context'/key/'context_geometry.npz',allow_pickle=False) as data:
                boxes = data['mask_boxes_xyxy'][:,0]
            rgb,_ = load_model_input(root/'context_inputs'/key/'input.json')
            draw.text((4,i*180+3),key,fill='black')
            canvas.paste(Image.fromarray(rgb[0]).resize((256,144)),(0,i*180+24))
            for t in range(8):
                pixels = rgb[t].copy()
                pixels[masks[t]] = (pixels[masks[t]]*.65+np.array([55,220,120])*.35).astype(np.uint8)
                x0,y0,x1,y1 = boxes[t]
                cx,cy = (x0+x1)/2,(y0+y1)/2
                side = max(x1-x0,y1-y0)*1.5
                crop = Image.fromarray(pixels).crop((int(cx-side/2),int(cy-side/2),int(cx+side/2),int(cy+side/2))).resize((128,128))
                x,y = 256+128*t,i*180+24
                draw.text((x+2,i*180+6),f'RGB{t} '+('gate OK' if accepted[t] else 'REJECT'),fill='black' if accepted[t] else 'red')
                canvas.paste(crop,(x,y))
            proofs.append(dict(key=key,mask_sha256=sha(path),all_prompt_gates_passed=bool(accepted.all())))
        path = output/f'page_{offset//10:02d}.png'
        canvas.save(path)
        pages.append(str(path))
    dump(output/'manifest.json',dict(cases=proofs,pages=pages,scope='All RGB0-7 dynamic-mask crops; RGB0 overview; not GT accuracy'))
    print(json.dumps(dict(cases=len(proofs),pages=pages),indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'small_trial_120')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=20)
    args = p.parse_args()
    sheets(args.root,args.output,args.limit)
