"""Read-only grouped split and appearance-leakage audit."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


GROUP_KEYS = ('key', 'lineage_cell', 'context_hash', 'motion_hash', 'episode_id')
APPEARANCE_KEYS = ('appearance_id', 'appearance_hash', 'render_style_id',
                   'texture_id', 'hdri_id', 'background_id')


def audit(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    records = manifest.get('records')
    if not isinstance(records, list) or not records:
        raise ValueError('manifest.records must be a nonempty list')
    rows = [r for r in records if isinstance(r, dict)]
    if len(rows) != len(records):
        raise ValueError('manifest contains a non-object record')
    splits = sorted({r.get('split') for r in rows})
    duplicate_keys = [key for key, count in Counter(r.get('key') for r in rows).items() if count > 1]
    overlaps = {}
    for field in GROUP_KEYS + APPEARANCE_KEYS:
        by_split = {}
        for split in splits:
            values = {r[field] for r in rows if r.get('split') == split and field in r}
            by_split[split] = values
        hit = {}
        for i, left in enumerate(splits):
            for right in splits[i + 1:]:
                common = sorted(by_split[left] & by_split[right], key=str)
                if common:
                    hit[f'{left}__{right}'] = common[:20]
        if hit:
            overlaps[field] = hit
    missing_group_fields = {field: sum(field not in r for r in rows) for field in GROUP_KEYS
                            if any(field in r for r in rows)}
    appearance_present = {field: sum(field in r for r in rows) for field in APPEARANCE_KEYS
                          if any(field in r for r in rows)}
    return dict(schema='predictor_split_audit_v1', manifest=str(path.resolve()),
                record_count=len(rows), splits=splits,
                counts=dict(Counter(r.get('split') for r in rows)),
                duplicate_keys=duplicate_keys, group_overlap=overlaps,
                missing_group_fields=missing_group_fields,
                appearance_fields_present=appearance_present,
                appearance_split_check=('NOT_APPLICABLE_NO_APPEARANCE_IDS'
                                        if not appearance_present else 'CHECKED'),
                pass_no_group_overlap=not overlaps and not duplicate_keys,
                note='Appearance disjointness cannot be established when the manifest has no appearance IDs.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
