"""Offline Experiment 1A regressions; no downloads, image model or paid API."""
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import BertTokenizer, BertConfig, BertForMaskedLM

from mre_captions import (DEFAULT_MODEL, SUFFIX, source_images, image_paths, generate_cache,
    read_cache, snapshot_inputs, load_caption_texts, truncation_stats, digest, source_binding)
from mre_data import MREData
from mre_protocol import prepare_dataset
from mre_trainer import NeighborPairs, MRETrainer
from run_mre import build_parser, experiment_config, load_data, evaluate_saved_run


@pytest.fixture
def experiment(tmp_path):
    source, images = tmp_path / 'source', tmp_path / 'images'
    source.mkdir()
    images.mkdir()
    for i in range(3):
        Image.new('RGB', (8, 8), (i * 80, 10, 20)).save(images / '{}.jpg'.format(i))
    all_rows = [{'token': ['Alice', 'visits', 'Paris', str(i)],
        'h': {'pos': [0, 1]}, 't': {'pos': [2, 3]},
        'relation': 'r{}'.format(i // 10), 'img_id': '{}.jpg'.format(i % 3)} for i in range(40)]
    (source / 'train.txt').write_text(json.dumps(all_rows[:20]), encoding='utf-8')
    (source / 'val.txt').write_text('\n' + '\n'.join(repr(r) for r in all_rows[20:30]), encoding='utf-8')
    (source / 'test.txt').write_text('\n'.join(json.dumps(r) for r in all_rows[30:]), encoding='utf-8')
    files = ['train.txt', 'val.txt', 'test.txt']
    rows, mapping, sources = source_images(source, files, 'half_open')
    binding = source_binding(sources, mapping, 'img_id')
    paths = image_paths(mapping, images)
    cache = tmp_path / 'captions.json'
    class Stub:
        def __init__(self, *args):
            pass
        def __call__(self, path):
            return 'people playing football on field ' + path.name
    generate_cache(paths, cache, DEFAULT_MODEL, captioner_factory=Stub, binding=binding)
    args = build_parser().parse_args(['--source-dir', str(source), '--source-files', *files,
        '--expected-classes', '4', '--position-format', 'half_open', '--no-llm',
        '--use-image-caption', '--caption-path', str(cache)])
    config = experiment_config(args)
    run = tmp_path / 'run'
    manifest = prepare_dataset(source, run / 'data', expected_total=4,
                               source_files=files, position_format='half_open')
    snapshot_inputs(config, run, manifest)
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'alice', 'paris', 'visits',
             'head', 'tail', 'sentence', ':', '.', '[', ']', '/', 'the', 'image', 'caption', 'is',
             'people', 'playing', 'football', 'on', 'field', 'jpg'] + [str(i) for i in range(40)]
    bert = tmp_path / 'bert'
    bert.mkdir()
    (bert / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = BertTokenizer(str(bert / 'vocab.txt'))
    tokenizer.save_pretrained(bert)
    config.update(bert_model=str(bert), tokenizer=str(bert), manifest_sha256=digest(run / 'data/manifest.json'))
    return SimpleNamespace(source=source, images=images, files=files, mapping=mapping, paths=paths,
        cache=cache, rows=rows, config=config, run=run, manifest=manifest, tokenizer=tokenizer, bert=bert,
        binding=binding)


def test_image_alignment_reuses_array_line_ids_and_paths(experiment):
    e = experiment
    assert e.mapping['train:00000001'] == '0.jpg'
    assert e.mapping['val:00000002'] == '2.jpg'  # blank physical line preserved
    assert e.mapping['test:00000001'] == '0.jpg'
    assert len(e.rows) == 40 and len(e.paths) == 3
    with pytest.raises(FileNotFoundError):
        image_paths({'sample': 'missing.jpg'}, e.images)
    with pytest.raises(ValueError, match='escapes'):
        image_paths({'sample': '../outside.jpg'}, e.images)


def test_missing_field_fails_only_caption_path(experiment):
    e = experiment
    source_images(e.source, e.files, 'half_open')
    with pytest.raises(ValueError, match='image field'):
        source_images(e.source, e.files, 'half_open', 'unobserved_field')


def test_cache_hits_never_load_model_and_changed_image_rejected(experiment):
    e = experiment
    def forbidden(*args):
        raise AssertionError('Must not load caption model for cached images')
    result = generate_cache(e.paths, e.cache, DEFAULT_MODEL, captioner_factory=forbidden, binding=e.binding)
    assert result == {'unique_images': 3, 'generated': 0, 'cache_hits': 3}
    with pytest.raises(ValueError, match='NEW cache'):
        generate_cache(e.paths, e.cache, 'different-model', captioner_factory=forbidden)
    Image.new('RGB', (8, 8), 'green').save(e.paths['0.jpg'])
    with pytest.raises(ValueError, match='Image changed'):
        generate_cache(e.paths, e.cache, DEFAULT_MODEL, captioner_factory=forbidden, binding=e.binding)


def test_interrupted_generation_resumes_completed_images(experiment, tmp_path):
    cache = tmp_path / 'resume.json'
    seen = []
    class Interrupted:
        def __init__(self, *args):
            pass
        def __call__(self, path):
            seen.append(path.name)
            if len(seen) == 2:
                raise RuntimeError('interrupted')
            return 'first saved caption'
    with pytest.raises(RuntimeError, match='interrupted'):
        generate_cache(experiment.paths, cache, DEFAULT_MODEL, captioner_factory=Interrupted)
    assert len(read_cache(cache)['entries']) == 1
    class Resume:
        def __init__(self, *args):
            pass
        def __call__(self, path):
            assert path.name != '0.jpg'
            return 'remaining caption'
    assert generate_cache(experiment.paths, cache, DEFAULT_MODEL, captioner_factory=Resume)['generated'] == 2


def test_baseline_splits_and_rng_unchanged(experiment, tmp_path):
    e = experiment
    second = tmp_path / 'baseline'
    prepare_dataset(e.source, second / 'data', expected_total=4,
                    source_files=e.files, position_format='half_open')
    for file in (e.run / 'data').iterdir():
        assert file.read_bytes() == (second / 'data' / file.name).read_bytes()
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    snapshot_inputs(e.config, e.run, e.manifest)
    assert random.getstate() == py_state
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    disabled = dict(e.config, use_image_caption=False, caption_path='does-not-exist')
    snapshot_inputs(disabled, second, e.manifest)
    assert load_caption_texts(disabled, second) is None
    assert not (second / 'caption_inputs.json').exists()
    before = MREData(second / 'data', e.tokenizer, max_length=64)
    after = MREData(second / 'data', e.tokenizer, max_length=64, caption_records=None)
    assert not hasattr(after, 'llm_input_ids')
    for a, b in zip(before.semi_dataset.tensors, after.semi_dataset.tensors):
        assert torch.equal(a, b)


def test_caption_only_changes_encoder_text_not_labels_or_llm(experiment, tmp_path):
    e = experiment
    records = load_caption_texts(e.config, e.run)
    baseline = MREData(e.run / 'data', e.tokenizer, max_length=64)
    augmented = MREData(e.run / 'data', e.tokenizer, max_length=64, caption_records=records)
    assert augmented.semi_records == baseline.semi_records
    for name in ('labeled_dataset', 'unlabeled_dataset', 'validation_dataset', 'test_dataset'):
        assert torch.equal(getattr(baseline, name).tensors[3], getattr(augmented, name).tensors[3])
    assert torch.equal(augmented.llm_input_ids, baseline.semi_dataset.tensors[0])
    assert not torch.equal(augmented.semi_dataset.tensors[0], baseline.semi_dataset.tensors[0])
    assert augmented.test_dataset.tensors[0] is augmented.unlabeled_dataset.tensors[0]
    class Client:
        def __init__(self):
            self.calls = []
        def choose_neighbor(self, query, choices):
            self.calls.append((query, choices))
            return 1
    clients = []
    for index, data in enumerate((baseline, augmented)):
        client = Client()
        clients.append(client)
        n = len(data.semi_dataset)
        pairs = NeighborPairs(data, np.tile(np.arange(n), (n, 1)), [0],
                              np.arange(n) % 2, client, e.tokenizer, tmp_path / '{}.jsonl'.format(index))
        with patch('numpy.random.choice', side_effect=lambda values, size: values[:size]):
            pairs[0]
    assert clients[0].calls == clients[1].calls
    assert 'caption' not in clients[1].calls[0][0]


@pytest.mark.parametrize('problem', ['missing', 'empty', 'model', 'source_binding'])
def test_bad_cache_fails_without_dropping_records(experiment, problem):
    e = experiment
    cache = read_cache(e.cache)
    if problem == 'missing':
        del cache['entries']['0.jpg']
    elif problem == 'empty':
        cache['entries']['0.jpg']['caption'] = ''
    elif problem == 'model':
        cache['generator']['model'] = 'different-model'
    else:
        cache['generator']['source_binding']['image_field'] = 'wrong_field'
    e.cache.write_text(json.dumps(cache), encoding='utf-8')
    with pytest.raises(ValueError):
        snapshot_inputs(e.config, e.run, e.manifest)


def test_saved_snapshot_independent_of_cache_and_hash_checked(experiment):
    e = experiment
    e.cache.unlink()
    assert len(load_caption_texts(e.config, e.run)) == 40
    with (e.run / 'caption_inputs.json').open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='hash'):
        load_caption_texts(e.config, e.run)


def test_truncation_newly_truncated_not_already_long(experiment):
    e = experiment
    rows = [{'id': 'a', 'text': 'Alice'}, {'id': 'b', 'text': 'Alice ' * 20}]
    captions = {r['id']: {'text': r['text'] + SUFFIX + 'football ' * 20} for r in rows}
    audit = truncation_stats(e.tokenizer, rows, 12, captions)
    assert audit['original_truncated'] == 1
    assert audit['truncated'] == 2
    assert audit['newly_truncated_by_caption'] == 1


def test_flags_and_task_guard(experiment):
    assert not build_parser().parse_args([]).use_image_caption
    assert build_parser().parse_args(['--use_image_caption']).use_image_caption
    assert not build_parser().parse_args(['--no-image-caption']).use_image_caption
    with pytest.raises(ValueError, match='MRE relation'):
        snapshot_inputs(dict(experiment.config, task_type='entity_type'), experiment.run, experiment.manifest)


def test_paired_run_checker_rejects_other_ablation_changes(experiment, tmp_path):
    from compare_caption_runs import compare_runs
    e = experiment
    baseline = tmp_path / 'paired_baseline'
    prepare_dataset(e.source, baseline / 'data', expected_total=4,
                    source_files=e.files, position_format='half_open')
    e.config['runtime'] = {'source_sha256': {'model.py': 'same-code'}, 'packages': {'torch': 'same'}}
    base_cfg = dict(e.config, use_image_caption=False, experiment_variant='baseline')
    for run, cfg in ((baseline, base_cfg), (e.run, e.config)):
        (run / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    assert compare_runs(baseline, e.run)['split_files_identical']
    base_cfg['train_batch_size'] += 1
    (baseline / 'config.json').write_text(json.dumps(base_cfg), encoding='utf-8')
    with pytest.raises(ValueError, match='settings differ'):
        compare_runs(baseline, e.run)


def test_caption_real_tiny_bert_training_and_saved_reevaluation(experiment):
    e = experiment
    torch.set_num_threads(1)
    torch.manual_seed(0)
    BertForMaskedLM(BertConfig(vocab_size=len(e.tokenizer), hidden_size=768,
        num_hidden_layers=1, num_attention_heads=12, intermediate_size=32,
        max_position_embeddings=64)).save_pretrained(e.bert)
    cfg = e.config
    cfg.update(pretrain_epochs=1, train_epochs=1, max_length=64, train_batch_size=12,
               labeled_batch_size=4, eval_batch_size=12, kmeans_n_init=2,
               query_pool_size=3, name_clusters=False)
    (e.run / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    with patch('requests.post', side_effect=AssertionError('Network forbidden')):
        tokenizer, data = load_data(cfg, e.run)
        result = MRETrainer(cfg, data, tokenizer, e.run).train()
        assert all(name in result for name in ('Base', 'Novel', 'Overall'))
        e.cache.unlink()
        evaluate_saved_run(e.run)
    assert json.loads((e.run / 'reevaluation.json').read_text())['predictions_identical']
    audit = json.loads((e.run / 'tokenization_audit.json').read_text())
    assert audit['unique_samples']['total'] == 40
