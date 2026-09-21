import json
from types import SimpleNamespace

import numpy as np
import pytest

from met_adapter import read_met_source
from mre_protocol import FILTERED_RELATIONS, prepare_dataset
from mre_prompts import MET_PROMPT_VERSION
from llm_client import LLMClient
import run_mre


LABELS = ['APP', 'Book', 'Building', 'Country', 'Currency', 'Event',
          'Location', 'Movie', 'Music', 'Organization', 'People', 'Site']
FILES = ('train.json', 'valid.json', 'test.json')


def sources(root):
    root.mkdir()
    # Every sentence has two targets; image/topic/wiki strings must never be inputs.
    for filename in FILES:
        records = [['Alice met Bob.', 'SECRET_IMAGE', 'SECRET_TOPIC',
                    [['Alice', label, 0, 5, 'SECRET_WIKI'], ['Bob', label, 10, 13, 'SECRET_WIKI']]]
                   for label in LABELS]
        records.append(['Alice', None, 'SECRET_TOPIC', [['Alice', 'Other', 0, 5, None]]])
        (root / filename).write_text(json.dumps(records), encoding='utf-8')


@pytest.mark.parametrize('seed', [0, 2, 3])
def test_met_expansion_split_rng_and_existing_loader(tmp_path, seed):
    source = tmp_path / 'source'
    sources(source)
    out = tmp_path / 'data'
    m = prepare_dataset(source, out, seed, 'half_open', 12, FILES, 'entity_type')
    assert (m['n_base'], m['n_novel']) == (6, 6)
    shuffled = LABELS.copy()
    np.random.RandomState(seed).shuffle(shuffled)
    assert m['classes'] == shuffled
    assert sum(x['retained_records'] for x in m['sources'].values()) == 72
    assert sum(x['filtered_records'] for x in m['sources'].values()) == 3
    assert {k: v['count'] for k, v in m['splits'].items()} == {
        'train_labeled': 12, 'validation': 6, 'test': 54, 'train_unlabeled': 54}
    assert m['position_unit'] == 'character'
    assert m['audit']['sentence_overlap']['sentence_identity']['labeled_test'] == 1
    unlabeled = [json.loads(s) for s in (out / 'train_unlabeled.jsonl').read_text().splitlines()]
    assert all(set(r) == {'id', 'text'} and 'SECRET' not in r['text'] for r in unlabeled)
    assert len({r['id'] for r in unlabeled}) == 54
    from mre_data import MREData
    from test_mre_data import FakeTokenizer
    data = MREData(out, FakeTokenizer(), labeled_batch_size=4, train_batch_size=4)
    assert data.n_base == 6 and data.n_total == 12


def test_character_offsets_before_whitespace_normalization(tmp_path):
    path = tmp_path / 'train.json'
    path.write_text(json.dumps([['  UK  UK', None, 'topic',
                                [['UK', 'Country', 6, 8, None]]]]))
    rows, _, _ = read_met_source(path, FILTERED_RELATIONS)
    assert rows[0]['text'] == 'Entity: UK. Sentence: UK [ENTITY] UK [/ENTITY]'


@pytest.mark.parametrize('span', [(0, 4), (-1, 5), (0, 99), (True, 5)])
def test_invalid_spans_stop_before_output(tmp_path, span):
    path = tmp_path / 'train.json'
    path.write_text(json.dumps([['Alice', None, None, [['Alice', 'People', *span, None]]]]))
    with pytest.raises(ValueError, match='record 1:entity 1'):
        prepare_dataset(tmp_path, tmp_path / 'out', position_format='half_open',
                        expected_total=12, source_files=('train.json',), task_type='entity_type')
    assert not (tmp_path / 'out').exists()


def test_met_prompts_and_cache_are_separate_from_relation(monkeypatch, tmp_path):
    requests = []
    def post(url, **kwargs):
        requests.append(kwargs['json'])
        answer = 'Choice 1' if len(kwargs['json']['messages']) == 1 else 'person'
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            'model': 'gpt-3.5-turbo', 'choices': [{'message': {'content': answer}, 'finish_reason': 'stop'}]})
    monkeypatch.setattr('llm_client.requests.post', post)
    cache = tmp_path / 'cache.jsonl'
    query = 'Entity: Alice. Sentence: [ENTITY] Alice [/ENTITY] arrived.'
    choices = [query, 'Entity: Rome. Sentence: [ENTITY] Rome [/ENTITY] is a city.']
    relation = LLMClient(api_key='test', cache_path=cache)
    relation.choose_neighbor(query, choices)
    met = LLMClient(api_key='test', cache_path=cache, task_type='entity_type')
    met.choose_neighbor(query, choices)
    assert met.requests_made == 1 and met.cache_hits == 0
    assert 'Compare the semantic types' in requests[-1]['messages'][0]['content']
    met.name_cluster([query] * 3)
    assert 'shared semantic type' in requests[-1]['messages'][1]['content']
    reloaded = LLMClient(cache_path=cache, task_type='entity_type')
    reloaded.choose_neighbor(query, choices)
    assert reloaded.cache_hits == 1 and reloaded.requests_made == 0
    assert reloaded.prompt_version == MET_PROMPT_VERSION


def test_runner_met_config_prepare_and_api_check(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source'
    sources(source)
    for name, value in [('TASK_TYPE', 'entity_type'), ('SOURCE_FILES', FILES),
                        ('EXPECTED_CLASSES', 12), ('POSITION_FORMAT', 'half_open')]:
        monkeypatch.setattr(run_mre.defaults, name, value, raising=False)
    monkeypatch.setattr(run_mre, 'runtime_info', lambda: {})
    def forbidden(*args, **kwargs):
        raise AssertionError('No model/network permitted')
    monkeypatch.setattr(run_mre, 'load_data', forbidden)
    monkeypatch.setattr('llm_client.requests.post', forbidden)
    run_mre.main(['--prepare-only', '--source-dir', str(source), '--output-root', str(tmp_path / 'outputs')])
    cfg = json.loads(next((tmp_path / 'outputs').glob('*/config.json')).read_text())
    assert cfg['task_type'] == 'entity_type' and cfg['prompt_version'] == MET_PROMPT_VERSION
    client = run_mre.create_client(run_mre.build_parser().parse_args([]))
    assert client.task_type == 'entity_type'
    client.last_query_fallback = False
    def choose(q, choices):
        assert '[ENTITY]' in q and all('Head:' not in c for c in choices)
        return 0
    monkeypatch.setattr(client, 'choose_neighbor', choose)
    monkeypatch.setattr(run_mre, 'create_client', lambda *args: client)
    run_mre.main(['--check-llm'])
    assert MET_PROMPT_VERSION in capsys.readouterr().out
    monkeypatch.setattr(run_mre, 'load_data', lambda *args: (None, None))
    class Trainer:
        def __init__(self, config, data, tokenizer, run_dir, llm):
            assert config['task_type'] == 'entity_type'
        def train(self):
            pass
    monkeypatch.setattr('mre_trainer.MRETrainer', Trainer)
    run_mre.main(['--source-dir', str(source), '--bert-model', str(source),
                  '--output-root', str(tmp_path / 'training')])
    summary = json.loads(next((tmp_path / 'training').glob('*/llm_summary.json')).read_text())
    assert summary['prompt_version'] == MET_PROMPT_VERSION
