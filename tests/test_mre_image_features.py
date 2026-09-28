"""Experiment 1B offline alignment/cache/ablation regressions; no API/downloads."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from mre_captions import digest, image_paths, source_binding, source_images
from mre_image_features import (DEFAULT_IMAGE_MODEL, FrozenViTEncoder, cache_manifest,
    fuse_features, generate_feature_cache, load_image_features, snapshot_image_features)
from mre_protocol import prepare_dataset
from run_mre import build_parser, experiment_config


@pytest.fixture
def experiment(tmp_path):
    source, images, model = (tmp_path / name for name in ('source', 'images', 'vit'))
    for path in (source, images, model):
        path.mkdir()
    (model / 'config.json').write_text(json.dumps({'hidden_size': 768}))
    for index in range(3):
        Image.new('RGB', (8, 8), (index * 60, 10, 20)).save(images / '{}.jpg'.format(index))
    rows = [{'token': ['Alice', 'met', 'Bob', str(i)], 'h': {'pos': [0, 1]},
             't': {'pos': [2, 3]}, 'relation': 'r{}'.format(i // 10),
             'img_id': '{}.jpg'.format(i % 3)} for i in range(40)]
    files = ['train.txt', 'val.txt', 'test.txt']
    (source / files[0]).write_text(json.dumps(rows[:20]))
    (source / files[1]).write_text('\n' + '\n'.join(repr(r) for r in rows[20:30]))
    (source / files[2]).write_text('\n'.join(json.dumps(r) for r in rows[30:]))
    retained, mapping, sources = source_images(source, files, 'half_open')
    paths = image_paths(mapping, images)
    binding = source_binding(sources, mapping, 'img_id')
    cache = tmp_path / 'features'
    class Encoder:
        def __init__(self, *args):
            pass
        def __call__(self, paths):
            return np.array([[int(path.stem) + 1, 2., 3.] for path in paths], dtype=np.float32)
    generate_feature_cache(paths, cache, str(model), encoder_factory=Encoder,
                           binding=binding, image_root=images, batch_size=2)
    args = build_parser().parse_args(['--source-dir', str(source), '--source-files', *files,
        '--position-format', 'half_open', '--expected-classes', '4', '--bert-model', str(model),
        '--use-image-feature', '--image-feature-cache', str(cache), '--image-model-path', str(model)])
    config = experiment_config(args)
    run = tmp_path / 'run'
    manifest = prepare_dataset(source, run / 'data', expected_total=4,
        source_files=files, position_format='half_open')
    records = [json.loads(line) for line in (run / 'data' / manifest['files']['test']).read_text().splitlines()]
    return SimpleNamespace(source=source, images=images, model=model, files=files, mapping=mapping,
        paths=paths, binding=binding, cache=cache, Encoder=Encoder, config=config,
        run=run, manifest=manifest, records=records, retained=retained)


def test_fusion_normalizes_modalities_separately_without_projection():
    text = np.array([[3., 4.], [0., 2.]], dtype=np.float32)
    image = np.array([[6., 8., 0.], [0., 0., 7.]], dtype=np.float32)
    fused = fuse_features(text, image)
    assert fused.shape == (2, 5)
    np.testing.assert_allclose(fused, [[.6, .8, .6, .8, 0.], [0., 1., 0., 0., 1.]], atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(fused, axis=1), np.sqrt(2), atol=1e-6)


@pytest.mark.parametrize('bad', [np.zeros((2, 3)), np.ones((1, 3)), np.full((2, 3), np.nan)])
def test_invalid_or_misaligned_image_features_are_rejected(bad):
    with pytest.raises(ValueError):
        fuse_features(np.ones((2, 4)), bad)


def test_cache_alignment_reuse_and_changed_images_fail(experiment):
    e = experiment
    assert e.mapping['val:00000002'] == '2.jpg'
    def forbidden(*args):
        raise AssertionError('Cache hit must not load any ViT')
    result = generate_feature_cache(e.paths, e.cache, str(e.model), encoder_factory=forbidden,
                                   binding=e.binding, image_root=e.images)
    assert result == {'unique_images': 3, 'generated': 0, 'cache_hits': 3, 'feature_dim': 3}
    Image.new('RGB', (8, 8), 'green').save(e.paths['0.jpg'])
    with pytest.raises(ValueError, match='Image changed'):
        generate_feature_cache(e.paths, e.cache, str(e.model), encoder_factory=forbidden, binding=e.binding)


def test_missing_images_and_cache_model_mismatch(experiment):
    e = experiment
    with pytest.raises(FileNotFoundError):
        image_paths({'x': 'missing.jpg'}, e.images)
    with pytest.raises(ValueError, match='escapes'):
        image_paths({'x': '../outside.jpg'}, e.images)
    with pytest.raises(ValueError, match='NEW'):
        generate_feature_cache(e.paths, e.cache, DEFAULT_IMAGE_MODEL, binding=e.binding)
    e.config['image_model_path'] = DEFAULT_IMAGE_MODEL
    with pytest.raises(ValueError, match='differs'):
        snapshot_image_features(e.config, e.run, e.manifest)


def test_interrupted_generation_resumes_without_recomputing(experiment, tmp_path):
    e, cache = experiment, tmp_path / 'resume'
    class Interrupted(e.Encoder):
        def __call__(self, paths):
            if paths[0].stem == '1':
                raise RuntimeError('interrupted')
            return super().__call__(paths)
    with pytest.raises(RuntimeError, match='interrupted'):
        generate_feature_cache(e.paths, cache, str(e.model), batch_size=1,
                               encoder_factory=Interrupted, binding=e.binding)
    assert len(cache_manifest(cache)['entries']) == 1
    class Resume(e.Encoder):
        def __call__(self, paths):
            assert all(path.stem != '0' for path in paths)
            return super().__call__(paths)
    result = generate_feature_cache(e.paths, cache, str(e.model), batch_size=1,
                                   encoder_factory=Resume, binding=e.binding)
    assert result['generated'] == 2 and result['cache_hits'] == 1


def test_snapshot_preserves_splits_rng_and_sample_order(experiment, capsys):
    e = experiment
    before = {p.name: p.read_bytes() for p in (e.run / 'data').iterdir()}
    py, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    snapshot_image_features(e.config, e.run, e.manifest)
    assert before == {p.name: p.read_bytes() for p in (e.run / 'data').iterdir()}
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    assert np.random.get_state()[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    features = load_image_features(e.config, e.run, e.records)
    expected = np.array([[int(e.mapping[row['id']].split('.')[0]) + 1, 2., 3.] for row in e.records])
    np.testing.assert_array_equal(features, expected)
    assert not (e.run / 'caption_inputs.json').exists()
    output = capsys.readouterr().out
    assert 'Frozen: True' in output and 'Missing images: 0' in output
    assert output.count('Text feature shape (expected): (768,)') == 3
    with pytest.raises(ValueError, match='order mismatch'):
        load_image_features(e.config, e.run, e.records[::-1])
    with np.load(e.run / 'image_features.npz', allow_pickle=False) as saved:
        assert set(saved.files) == {'features', 'sample_ids', 'image_ids'}  # No labels.


def test_snapshot_replay_does_not_require_original_images_or_cache(experiment):
    e = experiment
    snapshot_image_features(e.config, e.run, e.manifest)
    for path in e.images.iterdir():
        path.unlink()
    assert len(load_image_features(e.config, e.run, e.records)) == len(e.records)
    e.config['image_features_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash mismatch'):
        load_image_features(e.config, e.run, e.records)


def test_baseline_never_reads_images_or_cache(experiment, monkeypatch):
    e = experiment
    cfg = dict(e.config, use_image_feature=False, image_feature_cache='missing')
    monkeypatch.setattr('mre_image_features.source_images', lambda *a: pytest.fail('No image parsing allowed'))
    snapshot_image_features(cfg, e.run, e.manifest)
    assert load_image_features(cfg, e.run, e.records) is None
    assert not (e.run / 'image_features.npz').exists()


@pytest.mark.parametrize('options', [
    ['--use-image-feature'],
    ['--use-image-feature', '--image-feature-cache', 'x', '--use-image-caption'],
    ['--use-image-feature', '--image-feature-cache', 'x', '--no-freeze-image-encoder'],
    ['--use-image-feature', '--image-feature-cache', 'x', '--task-type', 'entity_type'],
])
def test_invalid_experiment_combinations_fail_before_training(options):
    with pytest.raises(ValueError):
        experiment_config(build_parser().parse_args(options))


def test_image_switch_does_not_change_training_or_llm_settings():
    parser = build_parser()
    baseline = experiment_config(parser.parse_args([]))
    fusion = experiment_config(parser.parse_args(['--use_image_feature', '--image_feature_cache', 'cache',
                                                 '--freeze_image_encoder']))
    assert fusion['freeze_image_encoder'] is True and fusion['use_image_caption'] is False
    for key in baseline:
        if key not in ('use_image_feature', 'image_feature_cache'):
            assert baseline[key] == fusion[key], key


def test_python_config_accepts_local_model_path(experiment, monkeypatch):
    import run_mre
    monkeypatch.setattr(run_mre.defaults, 'IMAGE_MODEL_PATH', experiment.model, raising=False)
    cfg = experiment_config(build_parser().parse_args([]))
    assert cfg['image_model_path'] == str(experiment.model)
    json.dumps(cfg)


def test_real_tiny_vit_is_frozen_and_returns_cls(tmp_path):
    from transformers import ViTConfig, ViTModel, ViTImageProcessor
    local = tmp_path / 'tiny-vit'
    torch.manual_seed(17)
    model = ViTModel(ViTConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=24, image_size=16, patch_size=4))
    model.save_pretrained(local)
    ViTImageProcessor(size={'height': 16, 'width': 16}).save_pretrained(local)
    path = tmp_path / 'image.png'
    Image.new('RGB', (20, 20), 'red').save(path)
    encoder = FrozenViTEncoder(str(local))
    first = encoder([path])
    assert first.shape == (1, 16)
    assert not encoder.model.training
    assert all(not parameter.requires_grad for parameter in encoder.model.parameters())
    np.testing.assert_array_equal(first, encoder([path]))
    with Image.open(path) as image:
        inputs = encoder.processor(images=image.convert('RGB'), return_tensors='pt')
    with torch.no_grad():
        expected = encoder.model(**inputs).last_hidden_state[:, 0, :].numpy()
    np.testing.assert_array_equal(first, expected)


def test_full_training_is_identical_and_only_final_kmeans_is_fused(experiment, monkeypatch):
    from transformers import BertConfig, BertForMaskedLM, BertTokenizer
    import mre_trainer
    from mre_data import MREData
    from run_mre import evaluate_saved_run
    e = experiment
    torch.set_num_threads(1)
    bert = e.run.parent / 'bert'
    bert.mkdir()
    vocab = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'alice', 'bob', 'met',
             'head', 'tail', 'sentence', ':', '.', '[', ']', '/'] + [str(i) for i in range(40)]
    (bert / 'vocab.txt').write_text('\n'.join(vocab), encoding='utf-8')
    tokenizer = BertTokenizer(str(bert / 'vocab.txt'))
    tokenizer.save_pretrained(bert)
    torch.manual_seed(10)
    BertForMaskedLM(BertConfig(vocab_size=len(vocab), hidden_size=768, num_hidden_layers=1,
        num_attention_heads=12, intermediate_size=24, max_position_embeddings=64)).save_pretrained(bert)
    def forbidden(*args, **kwargs):
        raise AssertionError('No real API allowed')
    monkeypatch.setattr('requests.post', forbidden)
    observed_neighbors = []
    def mine(features, pseudo, centers, *args):
        observed_neighbors.append(features.copy())
        assert features.shape[1] == centers.shape[1] == 768
        all_rows = np.arange(len(features))
        return np.tile(all_rows, (len(features), 1)), list(range(3))
    monkeypatch.setattr(mre_trainer, 'mine_neighbors', mine)
    class Client:
        model = 'gpt-3.5-turbo'
        requests_made = cache_hits = fallback_count = 0
        last_query_fallback = False
        def __init__(self):
            self.calls = []
        def choose_neighbor(self, query, choices):
            self.calls.append((query, choices))
            self.requests_made += 1
            return 1
    original_score = mre_trainer.cluster_score_loader
    observed_dimensions = []
    def score(model, loader, device, n_base, n_total, seed, n_init, **kwargs):
        image = kwargs.get('image_features')
        result = original_score(model, loader, device, n_base, n_total, seed, n_init, **kwargs)
        observed_dimensions.append((image is not None, result[2].shape[1]))
        return result
    monkeypatch.setattr(mre_trainer, 'cluster_score_loader', score)
    states, histories, clients = [], [], []
    for enabled in (False, True):
        run = e.run.parent / ('fusion' if enabled else 'baseline')
        manifest = prepare_dataset(e.source, run / 'data', expected_total=4,
            source_files=e.files, position_format='half_open')
        cfg = dict(e.config, use_image_feature=enabled, bert_model=str(bert), tokenizer=str(bert),
            pretrain_epochs=1, train_epochs=2, max_length=32, labeled_batch_size=4,
            train_batch_size=8, eval_batch_size=8, kmeans_n_init=2, update_every=1, name_clusters=False)
        cfg['manifest_sha256'] = digest(run / 'data' / 'manifest.json')
        data = MREData(run / 'data', tokenizer, max_length=32,
            labeled_batch_size=4, train_batch_size=8, eval_batch_size=8)
        if enabled:
            snapshot_image_features(cfg, run, manifest)
            data.final_image_features = load_image_features(cfg, run, data.test_records)
        (run / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
        client = Client()
        trainer = mre_trainer.MRETrainer(cfg, data, tokenizer, run, client)
        trainer.train()
        states.append(mre_trainer.cpu_state(trainer.model))
        histories.append(trainer.history)
        clients.append(client.calls)
        if enabled:
            evaluate_saved_run(run)
            assert json.loads((run / 'reevaluation.json').read_text())['predictions_identical']
            assert json.loads((run / 'final_feature_shapes.json').read_text())['fused'][1] == 771
    assert clients[0] == clients[1] and clients[0]
    assert histories[0] == histories[1]  # Same losses, LLM decisions and intermediate scores.
    for key in states[0]:
        assert torch.equal(states[0][key], states[1][key]), key
    for old, new in zip(observed_neighbors[:2], observed_neighbors[2:]):
        np.testing.assert_array_equal(old, new)
    assert observed_dimensions == [(False, 768), (False, 768), (False, 768), (True, 771), (True, 771)]
