import requests,json,uuid,hashlib,concurrent.futures
from pathlib import Path
xs=json.load(open('/tmp/test70_scene_catalog.json'))['results'];names=['Bare Room 2','Minimal Dark Interior Wall Art Mockup'];r=Path('/data/gaoya/agent-data/outputs_v1/geometry_transfer/20261003_test70_multiscene_preview');assets=[next(x for x in xs if x['displayName']==name) for name in names]
def get(a):
 assert a['isFree'];d=Path('/data/gaoya/agent-data/cache/rigidbench/blenderkit/scenes')/a['assetBaseId'];d.mkdir(exist_ok=True);f=next(f for f in a['files'] if f['fileType']=='blend');res=requests.get(f['downloadUrl'],params={'scene_uuid':str(uuid.uuid4())},timeout=60);res.raise_for_status();resp=requests.get(res.json()['filePath'],stream=True,timeout=(30,120));resp.raise_for_status();h=hashlib.sha256();n=0
 with (d/'scene.partial').open('wb') as out:
  for b in resp.iter_content(1024*1024):out.write(b);h.update(b);n+=len(b)
 assert n==f['fileUploadSize'];(d/'scene.partial').replace(d/'scene.blend');(d/'asset_source.json').write_text(json.dumps(dict(name=a['displayName'],asset_id=a['assetBaseId'],isFree=True,license=a['license'],source_blender=a['sourceAppVersion'],bytes=n,sha256=h.hexdigest()),indent=2));print('READY',a['displayName'],n,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(get,assets))
catalog=json.loads((r/'assets_catalog.json').read_text());catalog.extend(assets);(r/'assets_catalog.json').write_text(json.dumps(catalog,indent=2))
