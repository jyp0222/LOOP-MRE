"""Offline compatibility tests: original prompts/parser and auditable transport."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

import llm_client
from llm_client import LLMClient, LLMError, LLMConfigurationError, LLMBudgetExceeded

ROOT = Path(__file__).resolve().parents[1]
KEY = "private-test-key-never-log"
QUERY = "tokenizer decoded query"
CHOICES = ["tokenizer decoded first", "tokenizer decoded second"]


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No live API or real credential may be used in tests")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(llm_client.requests, "post", forbidden)
    monkeypatch.setattr(llm_client, "getpass", forbidden)
    monkeypatch.setattr(llm_client.time, "sleep", lambda seconds: None)


def completion(text="Choice 2", model="gpt-3.5-turbo-0301", finish="stop"):
    return {"id": "test-id", "model": model, "choices": [
        {"message": {"content": text}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 30, "completion_tokens": 3, "total_tokens": 33,
                  "ignored_secret": KEY}}


def stub(monkeypatch, response=None, status=200):
    calls = []
    def post(url, **kwargs):
        calls.append((url, kwargs))
        return SimpleNamespace(status_code=status, headers={},
                               json=lambda: response if response is not None else completion())
    monkeypatch.setattr(llm_client.requests, "post", post)
    return calls


def original_prompt(relative_path, locals_):
    """Read/evaluate only the upstream prompt assignment, without importing legacy code."""
    tree = ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "prompt" for target in node.targets))
    return eval(compile(ast.Expression(node.value), "<upstream prompt>", "eval"),
                {"__builtins__": {}}, locals_)


def test_neighbor_request_matches_upstream_prompt_and_api_defaults(monkeypatch):
    calls = stub(monkeypatch)
    client = LLMClient(api_key=KEY)
    assert client.choose_neighbor(QUERY, CHOICES) == 1
    url, request = calls[0]
    assert url == "https://api.openai.com/v1/chat/completions"
    assert request["json"] == {
        "model": "gpt-3.5-turbo-0301",
        "messages": [{"role": "user", "content": original_prompt(
            "utils/neighbor_dataset.py", dict(s=QUERY, s1=CHOICES[0], s2=CHOICES[1]))}]}
    assert request["headers"]["Authorization"] == "Bearer " + KEY
    assert request["allow_redirects"] is False
    assert client.requests_made == 1 and client.fallback_count == 0
    assert client.snapshot_verified is False


def test_naming_uses_original_alias_system_and_three_decoded_examples(monkeypatch):
    calls = stub(monkeypatch, completion(" common intent ", model="gpt-3.5-turbo"))
    client = LLMClient(api_key=KEY)
    samples = ["one", "two", "three"]
    assert client.name_cluster(samples) == " common intent "
    assert calls[0][1]["json"] == {
        "model": "gpt-3.5-turbo", "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": original_prompt(
                "loop.py", dict(s1="one", s2="two", s3="three"))}]}


@pytest.mark.parametrize("text,expected", [
    ("Choice 1", 0), ("I select Choice 2.", 1),
    ("Choice 2, but Choice 1 is also mentioned.", 0)])
def test_original_case_sensitive_substring_and_choice1_precedence(monkeypatch, text, expected):
    # Upstream checks content and does not use finish_reason as an extra filter.
    stub(monkeypatch, completion(text, finish="length"))
    assert LLMClient(api_key=KEY).choose_neighbor(QUERY, CHOICES) == expected


@pytest.mark.parametrize("response", [
    completion("choice 2"), completion("neither"), completion(None),
    {"model": "gpt-3.5-turbo-0301", "choices": []},
    completion("Choice 2", model="unrecognized-backend"),
])
def test_bad_neighbor_answers_use_observable_uncached_q1_fallback(monkeypatch, tmp_path, response):
    stub(monkeypatch, response)
    client = LLMClient(api_key=KEY, cache_path=tmp_path / "cache.jsonl",
                       log_path=tmp_path / "calls.jsonl")
    with pytest.warns(RuntimeWarning, match="q1 fallback"):
        assert client.choose_neighbor(QUERY, CHOICES) == 0
    assert client.last_query_fallback and client.fallback_count == 1
    assert not (tmp_path / "cache.jsonl").exists()
    log = [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert log[-1]["status"] == "neighbor_fallback"
    assert log[-1]["fallback"] == "original_q1"


@pytest.mark.parametrize("status", [401, 403, 404, 429, 500])
def test_http_failure_never_switches_model_and_is_not_silent(monkeypatch, tmp_path, status):
    calls = stub(monkeypatch, {"error": {"message": KEY}}, status)
    client = LLMClient(api_key=KEY, log_path=tmp_path / "calls.jsonl")
    with pytest.warns(RuntimeWarning, match="q1 fallback"):
        assert client.choose_neighbor(QUERY, CHOICES) == 0
    assert len(calls) == client.requests_made == 1
    assert all(call[1]["json"]["model"] == "gpt-3.5-turbo-0301" for call in calls)
    assert KEY not in (tmp_path / "calls.jsonl").read_text()


def test_network_failure_fallback_and_naming_failure_are_distinct(monkeypatch):
    def fail(*args, **kwargs):
        raise requests.exceptions.Timeout(KEY)
    monkeypatch.setattr(llm_client.requests, "post", fail)
    client = LLMClient(api_key=KEY)
    with pytest.warns(RuntimeWarning, match="q1 fallback"):
        assert client.choose_neighbor(QUERY, CHOICES) == 0
    with pytest.raises(LLMError, match="network/timeout") as error:
        client.name_cluster(["one", "two", "three"])
    assert KEY not in str(error.value)
    assert client.fallback_count == 1 and client.requests_made == 2


def test_budget_and_empty_key_never_become_fake_supervision(monkeypatch):
    budget = LLMClient(max_requests=0)
    with pytest.raises(LLMBudgetExceeded):
        budget.choose_neighbor(QUERY, CHOICES)
    assert budget.requests_made == budget.fallback_count == 0
    client = LLMClient(api_key="  ")
    with pytest.raises(LLMConfigurationError, match="empty"):
        client.choose_neighbor(QUERY, CHOICES)
    assert client.requests_made == client.fallback_count == 0


def test_explicit_retries_count_toward_attempt_budget(monkeypatch):
    calls = stub(monkeypatch, status=500)
    client = LLMClient(api_key=KEY, max_retries=3, max_requests=2)
    with pytest.raises(LLMBudgetExceeded):
        client.choose_neighbor(QUERY, CHOICES)
    assert len(calls) == client.requests_made == 2
    assert client.fallback_count == 0


def test_persistent_cache_is_private_and_needs_no_credentials(monkeypatch, tmp_path):
    calls = stub(monkeypatch)
    cache, log = tmp_path / "cache.jsonl", tmp_path / "log.jsonl"
    client = LLMClient(api_key=KEY, cache_path=cache, log_path=log)
    assert client.choose_neighbor(QUERY, CHOICES) == 1
    assert client.choose_neighbor(QUERY, CHOICES) == 1
    offline_client = LLMClient(cache_path=cache, log_path=log, max_requests=0)
    assert offline_client.choose_neighbor(QUERY, CHOICES) == 1
    assert len(calls) == 1
    assert offline_client.cache_hits == 1 and offline_client.requests_made == 0
    for file in (cache, log):
        text = file.read_text()
        for secret in [KEY, QUERY] + CHOICES:
            assert secret not in text
    assert json.loads(cache.read_text())["metadata"]["usage"]["total_tokens"] == 33


@pytest.mark.parametrize("changed", [
    {"base_url": "https://proxy.example/v1"}, {"model": "gpt-3.5-turbo"},
    {"max_output_tokens": 64}])
def test_cache_does_not_cross_endpoint_model_or_generation_settings(monkeypatch, tmp_path, changed):
    stub(monkeypatch)
    cache = tmp_path / "cache.jsonl"
    LLMClient(api_key=KEY, cache_path=cache).choose_neighbor(QUERY, CHOICES)
    client = LLMClient(cache_path=cache, max_requests=0, **changed)
    with pytest.raises(LLMBudgetExceeded):
        client.choose_neighbor(QUERY, CHOICES)


def test_old_prompt_version_cache_is_not_reused(monkeypatch, tmp_path):
    stub(monkeypatch)
    cache = tmp_path / "cache.jsonl"
    LLMClient(api_key=KEY, cache_path=cache).choose_neighbor(QUERY, CHOICES)
    record = json.loads(cache.read_text())
    record["version"] = "previous-json-relation-prompt"
    cache.write_text(json.dumps(record))
    with pytest.raises(LLMBudgetExceeded):
        LLMClient(cache_path=cache, max_requests=0).choose_neighbor(QUERY, CHOICES)


def test_corrupt_cache_fails_instead_of_silent_fallback(monkeypatch, tmp_path):
    stub(monkeypatch)
    cache = tmp_path / "cache.jsonl"
    LLMClient(api_key=KEY, cache_path=cache).choose_neighbor(QUERY, CHOICES)
    record = json.loads(cache.read_text())
    record["answer"]["choice"] = True
    cache.write_text(json.dumps(record))
    client = LLMClient(cache_path=cache, max_requests=0)
    with pytest.raises(LLMError, match="Cached answer"):
        client.choose_neighbor(QUERY, CHOICES)
    assert client.fallback_count == 0


def test_key_file_precedence_and_missing_file(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    key_file = tmp_path / "private.txt"
    key_file.write_text("file-key\n")
    calls = stub(monkeypatch)
    LLMClient(api_key_file=key_file).choose_neighbor(QUERY, CHOICES)
    assert calls[0][1]["headers"]["Authorization"] == "Bearer file-key"
    LLMClient(api_key="explicit", api_key_file=key_file).choose_neighbor(QUERY, CHOICES)
    assert calls[1][1]["headers"]["Authorization"] == "Bearer explicit"
    with pytest.raises(LLMConfigurationError, match="Cannot read"):
        LLMClient(api_key_file=tmp_path / "missing").choose_neighbor(QUERY, CHOICES)


def test_response_alias_mismatch_is_recorded_without_claiming_snapshot(monkeypatch):
    stub(monkeypatch, completion(model="gpt-3.5-turbo"))
    client = LLMClient(api_key=KEY)
    with pytest.warns(RuntimeWarning, match="snapshot is unverified"):
        assert client.choose_neighbor(QUERY, CHOICES) == 1
    assert client.model == "gpt-3.5-turbo-0301"
    assert client.last_response_model == "gpt-3.5-turbo"
    assert client.last_response_model_matches_request is False
    assert client.model_mismatch_count == 1 and client.snapshot_verified is False


@pytest.mark.parametrize("options", [
    {"base_url": "http://api.example/v1"}, {"base_url": "https://secret:password@host"},
    {"base_url": "https://host/v1?api_key=secret"}, {"timeout": 0},
    {"max_requests": -1}, {"max_requests": True}, {"max_retries": -1},
    {"max_output_tokens": 0}, {"reasoning_effort": "low"}, {"model": "gpt-6-astra"},
])
def test_invalid_local_configuration_is_rejected(options):
    with pytest.raises(ValueError):
        LLMClient(**options)


def test_output_files_cannot_overwrite_key_file(tmp_path):
    path = tmp_path / "private"
    with pytest.raises(ValueError):
        LLMClient(api_key_file=path, cache_path=path)
    with pytest.raises(ValueError):
        LLMClient(cache_path=path, log_path=path)
