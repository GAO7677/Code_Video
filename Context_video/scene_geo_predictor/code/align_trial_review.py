"""Align only a contact sheet explicitly inspected by the operator."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from prepare_small_trial import ROOT, dump, sha


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT/'small_trial_120')
    p.add_argument('--review',type=Path,required=True)
    p.add_argument('--confirm-all-overlays-inspected',action='store_true')
    args = p.parse_args()
    if not args.confirm_all_overlays_inspected:
        raise ValueError('Inspect every contact-sheet overlay before proceeding')
    manifest = json.loads((args.review/'manifest.json').read_text())
    for row in manifest['cases']:
        if sha(args.root/'observed_masks'/row['key']/'observed_masks.npz') != row['mask_sha256']:
            raise ValueError('Reviewed masks changed')
        if not row['all_prompt_gates_passed']:
            raise ValueError('Rejected masks need investigation, not automatic override')
    proof = dict(approved_cases=[r['key'] for r in manifest['cases']],
        review_manifest_sha256=sha(args.review/'manifest.json'),
        reviewed_indices=list(range(8)), scope='coarse visual inspection; not GT accuracy',
        static_geometry_accuracy_certified=False)
    dump(args.review/'approval.json',proof)
    results = []
    for row in manifest['cases']:
        key = row['key']
        out = args.root/'aligned_scene'/key
        if not (out/'report.json').exists():
            argv = [sys.executable,'-B',str(ROOT/'align_observed_depth.py'),
                '--input',str(args.root/'context_inputs'/key/'input.json'),
                '--geometry',str(args.root/'observed_context'/key/'context_geometry.npz'),
                '--masks',str(args.root/'observed_masks'/key/'observed_masks.npz'),
                '--raw',str(args.root/'raw_probe'/key/'raw_vggt.npz'),
                '--reviewed-masks','--output',str(out)]
            with (args.review/f'{key}_alignment.log').open('w') as log:
                result = subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=120)
        report = json.loads((out/'report.json').read_text())
        results.append(dict(key=key,status=report['status'],
                            accepted=report.get('alignment',{}).get('accepted',False),
                            failure_reasons=report.get('failure_reasons',report.get('alignment',{}).get('failure_reasons',[]))))
        print('ALIGN',key,report['status'],flush=True)
    dump(args.review/'alignment_summary.json',dict(records=results,all_passed=all(r['accepted'] for r in results)))
