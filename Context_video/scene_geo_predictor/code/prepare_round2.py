"""Round-2 isolated data links; change only final scene-token selection."""
import json

from prepare_small_trial import ROOT, dump


def main():
    for split in ('train', 'control'):
        source = ROOT/'two_round_release/round1'/split
        dest = ROOT/'two_round_release/round2'/split
        dest.mkdir(parents=True, exist_ok=False)
        manifest = json.loads((source/'manifest.json').read_text())
        manifest['token_selection'] = 'motion_corridor'
        dump(dest/'manifest.json', manifest)
        for name in ('samples', 'observed_context', 'context_inputs', 'observed_masks',
                     'raw_probe', 'render_context8', 'reviews', 'aligned_scene', 'simulation_audit.json'):
            if (source/name).exists():
                (dest/name).symlink_to((source/name).resolve(), target_is_directory=(source/name).is_dir())


if __name__ == '__main__':
    main()
