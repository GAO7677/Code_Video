"""Assemble frozen text-detection outputs; no model, no GT, no frame selection."""
import argparse,json,collections
from pathlib import Path
from PIL import Image,ImageDraw
from context_rgb_pybullet_common import dump_json,sha256_file

def run(a):
    root=a.root;c=json.loads(a.config.read_text());cases=[];groups=[];attempts=[]
    review=root/'review_crops';review.mkdir(exist_ok=False)
    for folder in sorted(root.iterdir()):
        if not (folder/'results.json').exists():continue
        by=collections.defaultdict(list)
        for r in json.loads((folder/'results.json').read_text()):by[r['group']].append(r)
        for key,rows in by.items():attempts.append({'run':folder.name,'group':key,'prompt':rows[0]['prompt'],'total':len(rows),'unique':sum(r['count']==1 for r in rows),'zero':sum(r['count']==0 for r in rows),'multiple':sum(r['count']>1 for r in rows)})
    for g in c['groups']:
        source='r7_length_all8' if g['id']=='incline_length_release' else 'r6_all8'
        folder=root/source
        for name,h in json.loads((folder/'freeze.json').read_text()).items():
            assert sha256_file(folder/name)==h
        rows=[r for r in json.loads((folder/'results.json').read_text()) if r['group']==g['id']]
        assert len(rows)==40 and all(r['prompt']==g['grounding_text'] for r in rows)
        montage=Image.new('RGB',(8*200,5*160),'#182132')
        for j,clip in enumerate(g['anonymous_clip_ids']):
            rr=sorted([r for r in rows if r['id']==clip],key=lambda x:x['frame']);assert [r['frame'] for r in rr]==list(range(8))
            for t,r in enumerate(rr):
                r['run']=source;r['image']=f'{source}/{g["id"]}/{clip}/rgb_{t:02d}.jpg'
                im=Image.open(root/r['image']).convert('RGB');w,h=im.size
                if r['count']==1:
                    x,y,z,v=r['boxes'][0];pad=max(z-x,v-y)*.6;crop=im.crop((max(0,int(x-pad)),max(0,int(y-pad)),min(w,int(z+pad)),min(h,int(v+pad))))
                else:crop=im
                crop.thumbnail((200,132));tile=Image.new('RGB',(200,160),'#182132');tile.paste(crop,((200-crop.width)//2,28));ImageDraw.Draw(tile).text((3,3),f'{clip} RGB{t} n={r["count"]}',fill='white');montage.paste(tile,(t*200,j*160))
            cases.append({'id':clip,'label':g['label'],'prompt':g['grounding_text'],'frames':rr,'identity_review':'PENDING'})
        montage.save(review/f'{g["id"]}.jpg',quality=94)
        groups.append({'id':g['id'],'label':g['label'],'prompt':g['grounding_text'],'total':40,'unique':sum(r['count']==1 for r in rows),'unique_rgb7':sum(r['count']==1 and r['frame']==7 for r in rows)})
    summary=f"70 cases / 560 frames; unique raw box: {sum(g['unique'] for g in groups)}/560. Identity review pending."
    dump_json(root/'viewer_data.json',{'cases':cases,'groups':groups,'attempts':attempts,'summary':summary});dump_json(root/'attempts.json',attempts);dump_json(root/'final_vocab.json',c)
    print(summary)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--apply-review',type=Path);a=p.parse_args()
    if a.apply_review:
        review=json.loads(a.apply_review.read_text());v=json.loads((a.root/'viewer_data.json').read_text())
        assert set(review['groups'])=={g['id'] for g in v['groups']}
        assert all(r['count']==1 for c in v['cases'] for r in c['frames'])
        for c in v['cases']:c['identity_review']='PASS: assistant visual inspection of full RGB7 and all 8 context target crops; not independent human annotation or GT IoU'
        v['summary']='70例 / 560帧：560/560恰好一个原始检测框；全部context目标图经助手目视检查通过。SAM2未重跑，GT框IoU未评测。'
        dump_json(a.root/'viewer_data.json',v);dump_json(a.root/'final_vocab.json',json.loads(a.config.read_text()))
        dump_json(a.root/'final_freeze.json',{name:sha256_file(a.root/name) for name in ['viewer_data.json','final_vocab.json','attempts.json','visual_review.json']})
    else:run(a)
