"""Add receipt-verified Nano P0 results without modifying historical artifacts."""
import hashlib
import json
from pathlib import Path
from statistics import median

REGISTRY=Path('/home/gaoya/code_Video_CONTEXT/nano_physicsiq_p0_controls/registry.json')
SCORE_KEYS=['verified','original','spatial','spatiotemporal','weighted_spatial','mse']

def read(path,default):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return default

def add_nano_methods(payload, output, ensure_link, case_loader, score_loader):
    runs=[];records={}
    for item in read(REGISTRY,[]):
        root=Path(item['out']);key=item['key'];label=item['label']
        plan=read(root/'protocol.json',{});status=read(root/'status.json',{})
        digest=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest()
        receipts={}
        for case in plan['cases']:
            name=case['generated_video_name']
            record=read(root/'full/cases'/Path(name).stem/'result.json',{})
            if (record.get('status')=='COMPLETE' and record.get('protocol_sha256')==digest
                    and (root/'full/submission'/name).is_file() and (root/'full/raw'/name).is_file()):
                receipts[name]=record
        records[key]=receipts
        count=len(receipts)
        complete=(count==198 and status.get('metrics')=='COMPLETE' and
                  read(root/'full_verified.json',{}).get('passed',False))
        csv=root/'metrics/physics-IQ-benchmark-verified/results/submission.csv'
        metrics=csv.with_name('submission_metrics.json')
        per_case=case_loader(csv) if complete else {}
        scores=score_loader(metrics) if complete else dict.fromkeys(SCORE_KEYS)
        for suffix,folder in [('', 'submission'),('_raw','raw')]:
            target=root/'full'/folder
            if target.is_dir():ensure_link(output/'assets'/(key+suffix),target)
        for case in payload['cases']:
            for view in case['views']:
                filename=view['benchmark_scenario'].replace('_take-1_','_')
                for suffix in ('','_raw'):
                    view['videos'][key+suffix]=f'assets/{key+suffix}/{filename}' if filename in receipts else None
                if complete:view['metrics'][key]=per_case[case['event']][view['view']]
        detail='clean context · no time embedding · guidance 6 · shift 10 · 40 steps · seed42'
        payload['methods'].append(dict(key=key,label=label,kind='model',color=item['color'],detail=detail))
        payload['raw_methods'].append(dict(key=key+'_raw',label=label+' · raw',color=item['color'],detail='189 frames · same Nano protocol'))
        payload['scoreboard'].append(dict(key=key,label=label,short_label=label,family='Cosmos3-Nano',
            run=root.name,scores=scores,video_status=f'{count}/198 generated',color=item['color'],
            case_metric_status=('Official Verified complete' if complete else 'Official Verified 正在计算' if status.get('metrics')=='RUNNING' and status.get('state')=='SCORING' else 'Official Verified 评分失败，待恢复' if status.get('state')=='FAILED' and count==198 else 'Official Verified 待评分' if count==198 else 'Official Verified 待完整生成后计算'),
            sampling=detail,source=str(csv),auxiliary_metric_status=False))
        payload['availability']['assets'][key]=count
        if complete:payload['availability']['per_view_metric_methods'].append(key)
        payload['data_sources'][key]=dict(checkpoint=plan['checkpoint'],csv=str(csv),protocol=str(root/'protocol.json'))
        seconds=[r['seconds'] for r in receipts.values()]
        smoke=read(root/'smoke/cases'/Path(plan['smoke_case']).stem/'result.json',{})
        if not seconds and smoke:seconds=[smoke['seconds']]
        runs.append(dict(key=key,label=label,generation=count,total=198,
            state=status.get('state','QUEUED'),smoke=status.get('smoke','PENDING'),
            workers=status.get('workers',[]),metrics=status.get('metrics','PENDING'),
            updated_utc=status.get('updated_utc'),error=status.get('error'),
            median_video_seconds=median(seconds) if seconds else None,
            remaining_gpu_hours=(198-count)*median(seconds)/3600 if seconds else None))
    # Compare only actual paired receipts. Never imply matching seeds prove matching noise.
    pair_checks=[]
    for left,right in [('nano_baseline','nano_n75_s1000'),('nano_clean_s1000','nano_n75_s1000')]:
        a=records.get(left,{});b=records.get(right,{});names=sorted(a.keys()&b.keys());bad=[]
        for name in names:
            x=a[name]['role_audit'];y=b[name]['role_audit']
            if (x['input']['prefix_sha256']!=y['input']['prefix_sha256'] or
                x['condition_latent_sha256']!=y['condition_latent_sha256'] or
                x['initial_future_sha256']!=y['initial_future_sha256']):bad.append(name)
        pair_checks.append(dict(left=left,right=right,paired_count=len(names),mismatches=bad,
            status='MISMATCH' if bad else 'VERIFIED' if names else 'WAITING_FOR_PAIRS'))
    payload['nano_runs']=runs
    payload['nano_pair_checks']=pair_checks
    payload['protocol'].update(
        comparison_scope='Physics-IQ Verified · Nano matched-inference controls + historical references',
        inference='Nano: guidance 6 / shift 10; Wan initial0907: CFG off / shift 5. Both 40 steps, seed42; samplers differ.',
        comparison_note='Nano 三组共享 198 个 BPP 输入、seed42、896×512、24fps、40步、guidance6、shift10、clean context、无 context 时间 embedding、原生 Nano UniPC 和同一导出代码。输入72帧，编码前69帧；生成189帧，评分 raw[69:189] 共120帧。历史 clean 为普通微调参照，不能据此声称训练仅有单因素差异。Wan initial0907 六组为 CFG关闭、shift5；其余历史方法按原记录保留，不能统称相同采样配置。')
    payload['availability']['note']='Nano progress updates every 30s; official scores appear only after all 198 videos and official scoring finish. Historical results retain their own sampling protocols.'
    payload['scoreboard'].sort(key=lambda r:r['scores']['verified'] if r['scores']['verified'] is not None else -1,reverse=True)
    for rank,row in enumerate(payload['scoreboard'],1):row['rank']=rank if row['scores']['verified'] is not None else '-'
