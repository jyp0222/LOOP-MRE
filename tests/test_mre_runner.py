"""Test the API-check entry point without loading BERT or making paid calls."""
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
    assert client.model == 'gpt-3.5-turbo'
    assert client.reasoning_effort is None
    assert client.endpoint == 'https://api.openai.com/v1/chat/completions'
    assert client.requests_made == 0


@pytest.mark.parametrize('options', [
    ['--model', 'gpt-5.6-sol'], ['--model', 'gpt-6-astra'],
    ['--model', 'gpt-3.5-turbo-0301'], ['--reasoning-effort', 'low'],
])
def test_runner_rejects_incompatible_options(options):
    with pytest.raises(SystemExit) as error:
        run_mre.build_parser().parse_args(options)
    assert error.value.code == 2


def test_check_llm_reports_actual_version_without_loading_training(monkeypatch, capsys):
    class Client:
        model = 'gpt-3.5-turbo'
        last_response_model = 'gpt-3.5-turbo-0125'
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
