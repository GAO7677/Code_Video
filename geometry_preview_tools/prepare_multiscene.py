from pathlib import Path
import json,requests,concurrent.futures,hashlib,uuid
from PIL import Image,ImageDraw
R=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');R.mkdir(exist_ok=True)
C=Path('/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes')
names=['Cafe interior loft bar','Interior Office','LUXURY interior house','Coffee Cafe interior','Luxury Dining Room Interior','Hallway indoor interior','Indoor luxury interior minimal style','Home minimal interior','kitchen_Scene_06',"Kyra's Living Room",'Studio Apartment Furnished','Coffee Office bar','Butterfly Mural Room']
rows=json.load(open('/tmp/test70_scene_catalog.json'))['results'];assets=[]
for name in names:
 a=next(x for x in rows if x['displayName']==name);assert a['isFree'];assets.append(a)
(R/'assets_catalog.json').write_text(json.dumps(assets,indent=2))
thumbs=R/'asset_thumbnails';thumbs.mkdir(exist_ok=True)
def thumbnail(a):
 p=thumbs/(a['assetBaseId']+'.jpg')
 if not p.exists():
  r=requests.get(a['thumbnailMiddleUrl'],timeout=60);r.raise_for_status();p.write_bytes(r.content)
 return p
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:paths=list(pool.map(thumbnail,assets))
canvas=Image.new('RGB',(1200,4*260),'#eaeaea');draw=ImageDraw.Draw(canvas)
for i,(a,p) in enumerate(zip(assets,paths)):
 im=Image.open(p).convert('RGB');im.thumbnail((300,230));x=i%4*300;y=i//4*260;canvas.paste(im,(x,y));draw.text((x+5,y+232),f'{i+1}: '+a['displayName'][:37],fill='black')
canvas.save(R/'asset_candidates.jpg',quality=85)
print('THUMBNAILS_READY',flush=True)
def download(a):
 d=C/a['assetBaseId'];d.mkdir(exist_ok=True);p=d/'scene.blend'
 if p.exists() and (d/'asset_source.json').exists():return dict(name=a['displayName'],path=str(p),cached=True)
 f=next(f for f in a['files'] if f['fileType']=='blend')
 r=requests.get(f['downloadUrl'],params={'scene_uuid':str(uuid.uuid4())},timeout=60);r.raise_for_status();url=r.json()['filePath']
 with requests.get(url,stream=True,timeout=(30,120)) as resp:
  resp.raise_for_status();h=hashlib.sha256();n=0
  with p.with_suffix('.partial').open('wb') as out:
   for chunk in resp.iter_content(1024*1024):out.write(chunk);h.update(chunk);n+=len(chunk)
  assert n==f['fileUploadSize'],(n,f['fileUploadSize']);p.with_suffix('.partial').replace(p)
 (d/'asset_source.json').write_text(json.dumps(dict(name=a['displayName'],asset_id=a['assetBaseId'],isFree=True,license=a['license'],source_blender=a['sourceAppVersion'],bytes=n,sha256=h.hexdigest()),indent=2))
 return dict(name=a['displayName'],path=str(p),bytes=n)
print('DOWNLOAD_TOTAL_BYTES',sum(next(f['fileUploadSize'] for f in a['files'] if f['fileType']=='blend') for a in assets),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 futures={pool.submit(download,a):a for a in assets}
 for future in concurrent.futures.as_completed(futures):
  try:print(json.dumps(future.result()),flush=True)
  except Exception as e:print('ASSET_FAILED',futures[future]['displayName'],repr(e),flush=True)
