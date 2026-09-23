"""Check that a saved baseline/caption pair differs only in Experiment 1A input."""
import argparse
import json
from pathlib import Path

from mre_captions import digest, load_caption_texts


def compare_runs(baseline, caption_run):
    runs = [Path(baseline), Path(caption_run)]
    configs = [json.loads((run / 'config.json').read_text(encoding='utf-8')) for run in runs]
    if configs[0].get('use_image_caption', False) or not configs[1].get('use_image_caption', False):
        raise ValueError('Expected caption-disabled baseline and caption-enabled experiment')
    ignored = {'use_image_caption', 'experiment_variant', 'runtime'}
    def settings(config):
        return {k: v for k, v in config.items() if k not in ignored and not k.startswith('caption_')}
    a, b = map(settings, configs)
    differences = {k: [a.get(k), b.get(k)] for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
    if differences:
        raise ValueError('Non-caption experiment settings differ: {}'.format(differences))
    if configs[0]['runtime']['source_sha256'] != configs[1]['runtime']['source_sha256']:
        raise ValueError('Code fingerprints differ; use the same current code for both runs')
    if configs[0]['runtime']['packages'] != configs[1]['runtime']['packages']:
        raise ValueError('Training package versions differ')
    manifests = []
    for run, cfg in zip(runs, configs):
        path = run / 'data' / 'manifest.json'
        if digest(path) != cfg['manifest_sha256']:
            raise ValueError('Manifest hash mismatch: {}'.format(run))
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifests.append(manifest)
        for name, expected in manifest['file_sha256'].items():
            if digest(run / 'data' / name) != expected:
                raise ValueError('Prepared split hash mismatch: {}'.format(name))
    if manifests[0] != manifests[1]:
        raise ValueError('Source data, split, class order or IDs differ')
    load_caption_texts(configs[1], runs[1])
    return {'paired_settings_match': True, 'split_files_identical': True,
            'seed': configs[0]['seed'], 'llm_enabled': configs[0]['llm_enabled'],
            'note': 'Checks recorded settings; does not prove GPU determinism or remote GPT reproducibility.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--caption-run', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.baseline, args.caption_run), indent=2))
