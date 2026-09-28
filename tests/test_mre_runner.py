"""Test the API-check entry point without loading BERT or making paid calls."""
import json
import sys
from types import SimpleNamespace

import pytest

import run_mre


def test_default_runner_constructs_gpt35_client_without_key_or_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No key prompt or network allowed')

    monkeypatch.setattr('llm_client.requests.post', forbidden)
    monkeypatch.setattr('llm_client.getpass', forbidden)
    args = run_mre.build_parser().parse_args([])
    assert args.model == 'gpt-3.5-turbo'
    client = run_mre.create_client(args)
    assert client.model == client.naming_model == 'gpt-3.5-turbo'
    assert client.reasoning_effort is None
    assert client.endpoint == run_mre.defaults.API_BASE.rstrip('/') + '/chat/completions'
    assert client.max_output_tokens is None
    assert client.max_retries == 0
    assert client.requests_made == 0
    assert client.temperature == 0.0
    assert client.relation_prompt == 'current'


@pytest.mark.parametrize('options', [
    ['--model', 'gpt-5.6-sol'], ['--model', 'gpt-6-astra'],
    ['--reasoning-effort', 'low'],
])
def test_runner_rejects_incompatible_options(options):
    with pytest.raises(SystemExit) as error:
        run_mre.build_parser().parse_args(options)
    assert error.value.code == 2


def test_check_llm_reports_actual_version_without_loading_training(monkeypatch, capsys):
    class Client:
        model = 'gpt-3.5-turbo'
        last_response_model = 'gpt-3.5-turbo-0125'
        last_query_fallback = False
        calls = 0

        def choose_neighbor(self, query, choices):
            assert 'Head:' in query and len(choices) == 2
            self.calls += 1
            return 0

    client = Client()
    monkeypatch.setattr(run_mre, 'create_client', lambda args: client)
    run_mre.main(['--check-llm'])
    assert client.calls == 1
    output = capsys.readouterr().out
    assert 'response model: gpt-3.5-turbo-0125' in output
    assert 'Choice 1' in output
    assert 'UNVERIFIED' in output
    assert 'Prompt version: fewrel-directed-relation-choice-v4' in output
    assert 'LLM temperature: 0.0' in output


def test_check_llm_rejects_fallback_instead_of_claiming_connection_passed(monkeypatch):
    class Client:
        last_query_fallback = True
        def choose_neighbor(self, query, choices):
            return 0
    monkeypatch.setattr(run_mre, 'create_client', lambda args: Client())
    with pytest.raises(RuntimeError, match='only the upstream error fallback'):
        run_mre.main(['--check-llm'])


def test_original_defaults_and_mre_exceptions_are_explicit():
    cfg = run_mre.experiment_config(run_mre.build_parser().parse_args([]))
    assert (cfg['labeled_batch_size'], cfg['train_batch_size'], cfg['eval_batch_size']) == (64, 128, 64)
    assert (cfg['pretrain_epochs'], cfg['train_epochs'], cfg['patience']) == (100, 50, 20)
    assert cfg['checkpoint_selection'] == 'last_epoch' and cfg['evaluation'] == 'test_kmeans'
    assert cfg['topk'] == 50 and cfg['query_pool_size'] == 500 and cfg['update_every'] == 5
    assert cfg['view_strategy'] == 'rtr' and cfg['rtr_prob'] == .25
    assert cfg['name_clusters'] is True
    assert cfg['max_queries_per_refresh'] is None
    assert cfg['model'] == cfg['naming_model'] == 'gpt-3.5-turbo'
    assert cfg['llm_temperature'] == 0.0 and cfg['temperature'] == 0.07
    assert cfg['relation_prompt'] == 'current'
    assert cfg['prompt_version'] == 'fewrel-directed-relation-choice-v4'
    assert cfg['experiment_variant'] == 'loop_mre_random40_gpt35_v3'
    assert 'known_cls_ratio' not in cfg and 'labeled_ratio' not in cfg
    for args in (['--no-llm'], ['--smoke'], ['--no-name-clusters']):
        assert not run_mre.experiment_config(run_mre.build_parser().parse_args(args))['name_clusters']


def test_prepare_only_records_new_protocol_and_prints_class_lists(tmp_path, monkeypatch, capsys):
    source = tmp_path / 'source'
    source.mkdir()
    for name, labels in (('train', range(64)), ('test', range(64, 80))):
        records = [
            {'tokens': ['Head', 'met', 'Tail', str(label), str(i)],
             'h': {'pos': [0]}, 't': {'pos': [2]}, 'relation': 'relation-{}'.format(label)}
            for label in labels for i in range(5)]
        (source / (name + '.txt')).write_text(
            '\n'.join(json.dumps(row) for row in records), encoding='utf-8')
    def forbidden(*args, **kwargs):
        raise AssertionError('Prepare-only must not load BERT or create an API client')
    monkeypatch.setattr(run_mre, 'load_data', forbidden)
    monkeypatch.setattr(run_mre, 'create_client', forbidden)
    monkeypatch.setattr(run_mre, 'runtime_info', lambda: {'test': True})
    outputs = tmp_path / 'outputs'
    run_mre.main(['--prepare-only', '--seed', '2', '--source-dir', str(source),
                  '--output-root', str(outputs)])
    directories = list(outputs.iterdir())
    assert len(directories) == 1
    run_dir = directories[0]
    assert run_dir.name.startswith('loop_mre_random40_gpt35_v3_seed2_')
    config = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
    manifest = json.loads((run_dir / 'data' / 'manifest.json').read_text(encoding='utf-8'))
    assert config['data_protocol'] == manifest['protocol'] == 'mre_transductive_random_half'
    assert config['seed'] == manifest['seed'] == 2
    assert config['llm_temperature'] == 0.0 and config['relation_prompt'] == 'current'
    assert (manifest['n_base'], manifest['n_novel']) == (40, 40)
    output = capsys.readouterr().out
    assert 'MRE random-half classes: 40 base / 40 novel (seed=2)' in output
    assert 'Base: ' + str(manifest['base_classes']) in output
    assert 'Novel: ' + str(manifest['novel_classes']) in output


@pytest.mark.parametrize('option,expected', [('0', 0.0), ('0.5', 0.5), ('2', 2.0), ('provider', None)])
def test_runner_temperature_reaches_config_and_client(option, expected):
    args = run_mre.build_parser().parse_args(['--llm-temperature', option])
    config = run_mre.experiment_config(args)
    client = run_mre.create_client(args)
    assert config['llm_temperature'] == client.temperature == expected
    assert config['relation_prompt'] == client.relation_prompt == 'current'
    assert config['prompt_version'] == client.prompt_version
    assert config['temperature'] == 0.07  # RNCL still uses its independent temperature.
    assert config['llm_enabled'] is True


def test_runner_can_set_temperature_in_python_config(monkeypatch):
    monkeypatch.setattr(run_mre.defaults, 'LLM_TEMPERATURE', None, raising=False)
    args = run_mre.build_parser().parse_args([])
    config, client = run_mre.experiment_config(args), run_mre.create_client(args)
    assert config['llm_temperature'] is client.temperature is None
    assert config['relation_prompt'] == client.relation_prompt == 'current'
    assert config['temperature'] == 0.07


def test_server_current_prompt_constants_reach_client_and_metadata(monkeypatch):
    """Servers may keep a different current prompt without changing the runner."""
    import mre_prompts
    monkeypatch.setattr(mre_prompts, 'PROMPT_VERSION', 'server-current-version')
    monkeypatch.setattr(mre_prompts, 'NEIGHBOR_INSTRUCTIONS', 'Server neighbor instructions')
    monkeypatch.setattr(mre_prompts, 'NAMING_INSTRUCTIONS', 'Server naming instructions')
    args = run_mre.build_parser().parse_args([])
    config, client = run_mre.experiment_config(args), run_mre.create_client(args)
    assert config['prompt_version'] == client.prompt_version == 'server-current-version'
    assert client.neighbor_instructions == 'Server neighbor instructions'
    assert client.naming_instructions == 'Server naming instructions'


@pytest.mark.parametrize('option', ['-1', '2.01', 'nan', 'inf', 'True', 'bad'])
def test_invalid_cli_temperature_is_rejected(option):
    with pytest.raises(SystemExit) as error:
        run_mre.build_parser().parse_args(['--llm-temperature', option])
    assert error.value.code == 2


def test_no_llm_switch_retains_baseline_training_settings():
    parser = run_mre.build_parser()
    baseline = run_mre.experiment_config(parser.parse_args(['--no-llm']))
    changed = run_mre.experiment_config(parser.parse_args([
        '--no-llm', '--llm-temperature', 'provider']))
    assert changed['llm_enabled'] is changed['name_clusters'] is False
    changed.pop('llm_temperature')
    changed.pop('relation_prompt')
    changed.pop('prompt_version')
    for key in ('llm_temperature', 'relation_prompt', 'prompt_version'):
        baseline.pop(key)
    assert baseline == changed


@pytest.mark.parametrize('no_llm', [False, True])
def test_runner_records_settings_and_skips_client_when_disabled(tmp_path, monkeypatch, no_llm):
    """Exercise runner orchestration with stand-ins; never load BERT or train."""
    def forbidden(*args, **kwargs):
        raise AssertionError('No network or credentials allowed')
    monkeypatch.setattr('llm_client.requests.post', forbidden)
    monkeypatch.setattr('llm_client.getpass', forbidden)

    def prepare(source, destination, seed, position_format, **kwargs):
        destination.mkdir(parents=True)
        manifest = {'n_base': 11, 'n_novel': 11, 'base_classes': [], 'novel_classes': [],
                    'splits': {}, 'sources': {}, 'audit': {}, 'protocol': 'test_fixture'}
        (destination / 'manifest.json').write_text(json.dumps(manifest))
        return manifest
    monkeypatch.setattr('mre_protocol.prepare_dataset', prepare)
    monkeypatch.setattr('mre_captions.snapshot_inputs', lambda *args: None)
    monkeypatch.setattr(run_mre, 'runtime_info', lambda: {})
    monkeypatch.setattr(run_mre, 'load_data', lambda *args: (object(), object()))
    received = []

    class Trainer:
        def __init__(self, config, data, tokenizer, run_dir, client):
            received.append(client)
        def train(self):
            pass
    monkeypatch.setitem(sys.modules, 'mre_trainer', SimpleNamespace(MRETrainer=Trainer))
    if no_llm:
        monkeypatch.setattr(run_mre, 'create_client', forbidden)
    outputs = tmp_path / 'outputs'
    arguments = ['--output-root', str(outputs), '--bert-model', str(tmp_path),
                 '--llm-temperature', '0']
    if no_llm:
        arguments.append('--no-llm')
    run_mre.main(arguments)
    run = next(outputs.iterdir())
    config = json.loads((run / 'config.json').read_text())
    assert config['llm_temperature'] == 0.0 and config['temperature'] == 0.07
    assert config['relation_prompt'] == 'current'
    assert config['llm_enabled'] is (not no_llm)
    if no_llm:
        assert received == [None]
        assert not (run / 'llm_summary.json').exists()
    else:
        summary = json.loads((run / 'llm_summary.json').read_text())
        assert summary['llm_temperature'] == 0.0
        assert summary['relation_prompt'] == 'current'
        assert summary['prompt_version'] == config['prompt_version'] == received[0].prompt_version
        assert summary['http_attempts'] == 0 and summary['fallback_count'] == 0
