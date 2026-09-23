"""Build a small viewer index without changing frozen experiment results."""
import json
from pathlib import Path
import shutil
import sys


def write_index(root, payload):
    rows = []
    for record in payload['records']:
        rows.append({k: record[k] for k in ('id', 'case_id', 'rollout_status')})
        rows[-1]['evaluation'] = {'trajectory_metrics': {
            'ADE_m': record.get('evaluation', {}).get('trajectory_metrics', {}).get('ADE_m')
        }}
        rows[-1]['detail_url'] = 'cases/' + record['id'] + '.json'
    (root / 'viewer_index.json').write_text(json.dumps(
        {'summary': payload['summary'], 'records': rows}, ensure_ascii=False, separators=(',', ':')))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    write_index(root, json.loads((root / 'viewer_data.json').read_text()))
    web = Path(__file__).resolve().parents[1] / 'web'
    for source, target in [('test70_context_viewer.js', 'test70_viewer.js'),
                           ('test70_video_mode.js', 'test70_video_mode.js')]:
        shutil.copyfile(web / source, root / target)
    html = (root / 'index.html').read_text()
    import re
    html = re.sub(r'(test70_(?:viewer|video_mode)\.js)\?[^"\s]+',
                  r'\1?v=lazy-video-20260923', html)
    (root / 'index.html').write_text(html)
    print('index bytes:', (root / 'viewer_index.json').stat().st_size)
