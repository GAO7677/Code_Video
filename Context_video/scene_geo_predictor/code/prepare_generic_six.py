"""Dataset curator only: context-only anonymous screening, then frozen six selection."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw
from context_rgb_pybullet_common import dump_json, sha256_file

DATA=Path('/data/gaoya/AAA_test_video/physv_v2v_0819/samples')
def screen(out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]; sheet=Image.new('RGB',(7*256,10*165),'#182030'); draw=ImageDraw.Draw(sheet)
    # Paths are enumerated only by curator; names never enter the estimator.
    for i,folder in enumerate(sorted(p for p in DATA.iterdir() if p.is_dir())):
        path=folder/'context/context8_cycles.mp4'; cap=cv2.VideoCapture(str(path))
        fps=cap.get(cv2.CAP_PROP_FPS); frames=[]
        for t in range(8):
            ok,bgr=cap.read()
            if not ok: raise ValueError(f'context read failed ordinal {i}')
            frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
        cap.release()
        anon=f'candidate_{i:03d}'; dest=out/'screening'/anon; dest.mkdir(parents=True)
        for t,rgb in enumerate(frames): Image.fromarray(rgb).save(dest/f'rgb_{t:02d}.png')
        dump_json(dest/'timestamps.json',{'time_s':(np.arange(8)/fps).tolist()})
        thumb=Image.fromarray(frames[0]);thumb.thumbnail((256,140))
        x=i%7*256;y=i//7*165;sheet.paste(thumb,(x,y+23));draw.text((x+4,y+4),anon,fill='white')
        rows.append({'anonymous_id':anon,'source_context':str(path),'source_sample':str(folder),'fps':fps})
    sheet.save(out/'context_screening.jpg'); dump_json(out/'curator_mapping.json',rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();cv2.setNumThreads(2);screen(a.output)
