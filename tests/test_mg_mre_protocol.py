"""New three-source MRE adapter; no images, model downloads or API calls."""
import json

import pytest

import run_mre
from mre_protocol import prepare_dataset


def make_sources(root, array=False):
    root.mkdir()
    for name, count in [('train', 3), ('val', 2), ('test', 2)]:
        rows = [dict(token=['Head', 'entity', 'meets', 'Tail', str(i), name],
                     h={'pos': [0, 2]}, t={'pos': [3, 4]},
                     relation='/relation/{}'.format(c))
                for c in range(22) for i in range(count)]
        rows.append(dict(rows[0], relation='None'))
        content = json.dumps(rows) if array else '\n'.join(map(json.dumps, rows))
        (root / (name + '.txt')).write_text(content, encoding='utf-8')


@pytest.mark.parametrize('array', [False, True])
@pytest.mark.parametrize('seed', [0, 2, 3])
def test_three_sources_keep_all_positive_rows_and_hide_test_labels(tmp_path, array, seed):
    source = tmp_path / 'source'
    make_sources(source, array)
    output = tmp_path / 'data'
    manifest = prepare_dataset(source, output, seed, 'half_open', 22,
                               ('train.txt', 'val.txt', 'test.txt'))
    assert (manifest['n_base'], manifest['n_novel']) == (11, 11)
    assert {k: v['count'] for k, v in manifest['splits'].items()} == {
        'train_labeled': 22, 'validation': 11, 'test': 121, 'train_unlabeled': 121}
    assert sum(x['filtered_records'] for x in manifest['sources'].values()) == 3
    assert sum(x['retained_records'] for x in manifest['sources'].values()) == 154
    assert 'None' not in manifest['base_classes'] + manifest['novel_classes']
    rows = [json.loads(x) for x in (output / 'train_unlabeled.jsonl').read_text().splitlines()]
    assert all(set(x) == {'id', 'text'} for x in rows)
    assert any(x['id'].startswith('val:') for x in rows)
    assert all('Head: Head entity.' in x['text'] for x in rows)
    assert manifest['class_split']['source_order'] == ['train.txt', 'val.txt', 'test.txt']


def test_missing_val_does_not_silently_drop_it(tmp_path):
    source = tmp_path / 'source'
    make_sources(source)
    (source / 'val.txt').unlink()
    with pytest.raises(FileNotFoundError):
        prepare_dataset(source, tmp_path / 'data', position_format='half_open',
                        expected_total=22, source_files=('train.txt', 'val.txt', 'test.txt'))
    assert not (tmp_path / 'data').exists()


@pytest.mark.parametrize('names', [(), ('train.txt', 'train.json'), ('../train.txt',)])
def test_invalid_source_names(tmp_path, names):
    with pytest.raises(ValueError, match='source_files'):
        prepare_dataset(tmp_path, tmp_path / 'out', source_files=names)


def test_prepare_runner_uses_python_configuration(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    make_sources(source)
    monkeypatch.setattr(run_mre.defaults, 'SOURCE_FILES', ('train.txt', 'val.txt', 'test.txt'), raising=False)
    monkeypatch.setattr(run_mre.defaults, 'EXPECTED_CLASSES', 22, raising=False)
    monkeypatch.setattr(run_mre.defaults, 'POSITION_FORMAT', 'half_open')
    def forbidden(*args, **kwargs):
        raise AssertionError('No model or API during prepare-only')
    monkeypatch.setattr(run_mre, 'load_data', forbidden)
    monkeypatch.setattr(run_mre, 'create_client', forbidden)
    monkeypatch.setattr(run_mre, 'runtime_info', lambda: {'test': True})
    run_mre.main(['--prepare-only', '--source-dir', str(source),
                  '--output-root', str(tmp_path / 'outputs'), '--experiment-variant', 'mg_mre'])
    run = next((tmp_path / 'outputs').iterdir())
    config = json.loads((run / 'config.json').read_text())
    assert config['expected_classes'] == 22
    assert config['source_files'] == ['train.txt', 'val.txt', 'test.txt']
    assert config['experiment_variant'] == 'mg_mre'
    assert config['topk'] == run_mre.defaults.TOPK

