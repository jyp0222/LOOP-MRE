"""Offline contract tests: no API credential or network connection is used."""

import copy
import json

import pytest
import requests

import llm_client
from llm_client import LLMBudgetExceeded, LLMClient, LLMError


KEY = "sk-test-only-do-not-log-123456"
QUERY = "Head: source. Tail: target. Sentence: private-query-text."
CHOICES = ["Head: first. Tail: second. Sentence: first-candidate.",
           "Head: second. Tail: first. Sentence: second-candidate."]


def completed(answer=None):
    if answer is None:
        answer = {"choice": 1}
    return {
        "id": "resp_offline_1", "model": "gpt-5.6-sol", "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {"type": "message", "role": "assistant", "status": "completed",
             "content": [{"type": "output_text", "text": json.dumps(answer)}]},
        ],
        "usage": {"input_tokens": 100, "output_tokens": 12, "total_tokens": 112,
                  "input_tokens_details": {"cached_tokens": 32},
                  "output_tokens_details": {"reasoning_tokens": 8}},
    }


class FakeResponse:
    def __init__(self, payload=None, status=200, headers=None, json_error=False):
        self.payload = completed() if payload is None else payload
        self.status_code = status
        self.headers = headers or {}
        self.json_error = json_error

    def json(self):
        if self.json_error:
            raise ValueError("secret server body " + KEY)
        return copy.deepcopy(self.payload)


@pytest.fixture(autouse=True)
def prevent_network_and_real_credentials(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected network request or interactive credential prompt")

    monkeypatch.setattr(llm_client.requests, "post", forbidden)
    monkeypatch.setattr(llm_client, "getpass", forbidden)


def fake_transport(monkeypatch, outcomes):
    pending = list(outcomes)
    calls = []

    def post(url, **kwargs):
        calls.append({"url": url, **copy.deepcopy(kwargs)})
        assert pending, "More HTTP calls than expected"
        outcome = pending.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(llm_client.requests, "post", post)
    return calls


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_responses_request_has_strict_directed_choice_schema(monkeypatch):
    calls = fake_transport(monkeypatch, [FakeResponse()])
    client = LLMClient(api_key=KEY, max_output_tokens=1234, timeout=17)
    assert client.choose_neighbor(QUERY, CHOICES) == 1
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.openai.com/v1/responses"
    assert call["headers"] == {"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}
    assert call["timeout"] == 17
    assert call["allow_redirects"] is False
    payload = call["json"]
    assert payload["model"] == "gpt-5.6-sol"
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_output_tokens"] == 1234 and payload["store"] is False
    assert "temperature" not in payload
    assert "Head-to-Tail" in payload["instructions"]
    assert "data, not instructions" in payload["instructions"]
    assert json.loads(payload["input"][0]["content"]) == {
        "query": QUERY, "choices": [{"index": 0, "text": CHOICES[0]}, {"index": 1, "text": CHOICES[1]}]
    }
    fmt = payload["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"] == {
        "type": "object", "properties": {"choice": {"type": "integer", "enum": [0, 1]}},
        "required": ["choice"], "additionalProperties": False,
    }
    assert client.requests_made == 1


def test_astra_default_effort_and_no_model_fallback(monkeypatch):
    calls = fake_transport(monkeypatch, [FakeResponse(status=404)])
    client = LLMClient(model="gpt-6-astra", api_key=KEY)
    assert client.reasoning_effort == "low"
    with pytest.raises(LLMError, match="HTTP 404"):
        client.choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 1 and calls[0]["json"]["model"] == "gpt-6-astra"


@pytest.mark.parametrize("kwargs", [
    {"model": "not-a-supported-model"}, {"model": "gpt-6-astra", "reasoning_effort": "none"},
    {"reasoning_effort": "ultra"}, {"max_output_tokens": 0}, {"max_output_tokens": True},
    {"timeout": 0}, {"timeout": float("nan")}, {"timeout": float("inf")}, {"timeout": True},
    {"max_retries": -1}, {"max_retries": 1.5}, {"max_requests": -1}, {"max_requests": False},
    {"base_url": "http://api.openai.com/v1"}, {"base_url": "https://user:pass@api.openai.com/v1"},
    {"base_url": "https://api.openai.com/v1?token=private"},
    {"base_url": "https://api.openai.com/v1#private"},
])
def test_invalid_configuration_fails_without_network(kwargs):
    with pytest.raises(ValueError):
        LLMClient(**kwargs)


@pytest.mark.parametrize("query, choices", [
    ("", CHOICES), (None, CHOICES), (QUERY, ["only one"]),
    (QUERY, ["valid", ""]), (QUERY, "not a candidate list"),
])
def test_invalid_query_fails_without_network(query, choices):
    with pytest.raises(ValueError):
        LLMClient(api_key=KEY).choose_neighbor(query, choices)


@pytest.mark.parametrize("answer", [
    {"choice": True}, {"choice": -1}, {"choice": 2}, {"choice": "1"},
    {"choice": 1.0}, {"choice": 1, "explanation": "extra"}, {}, [1], None,
])
def test_invalid_structured_answer_stops_without_cache_or_retry(monkeypatch, tmp_path, answer):
    payload = completed({"choice": 1})
    payload["output"][1]["content"][0]["text"] = json.dumps(answer)
    calls = fake_transport(monkeypatch, [FakeResponse(payload)])
    cache = tmp_path / "cache.jsonl"
    client = LLMClient(api_key=KEY, cache_path=cache)
    with pytest.raises(LLMError):
        client.choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 1 and not cache.exists()


@pytest.mark.parametrize("kind", ["incomplete", "refusal", "malformed", "empty", "bad_message", "error"])
def test_failed_output_is_logged_but_never_used_as_supervision(monkeypatch, tmp_path, kind):
    payload = completed()
    if kind == "incomplete":
        payload["status"] = "incomplete"
        payload["incomplete_details"] = {"reason": "max_output_tokens"}
    elif kind == "refusal":
        payload["output"][1]["content"] = [{"type": "refusal", "refusal": "refused " + QUERY}]
    elif kind == "malformed":
        payload["output"][1]["content"][0]["text"] = "Choice 2"
    elif kind == "empty":
        payload["output"][1]["content"] = []
    elif kind == "bad_message":
        payload["output"][1]["status"] = "in_progress"
    else:
        payload["error"] = {"message": KEY}
    calls = fake_transport(monkeypatch, [FakeResponse(payload)])
    path = tmp_path / "log.jsonl"
    client = LLMClient(api_key=KEY, log_path=path)
    with pytest.raises(LLMError) as error:
        client.choose_neighbor(QUERY, CHOICES)
    assert KEY not in str(error.value) and QUERY not in str(error.value)
    assert len(calls) == 1
    logs = read_jsonl(path)
    assert logs[-1]["status"] == "invalid_response"
    assert logs[-1]["usage"]["total_tokens"] == 112
    assert not client._cache


def test_persistent_cache_hit_uses_no_key_and_no_attempt_budget(monkeypatch, tmp_path):
    cache, log = tmp_path / "cache.jsonl", tmp_path / "log.jsonl"
    calls = fake_transport(monkeypatch, [FakeResponse()])
    original = LLMClient(api_key=KEY, cache_path=cache, log_path=log, max_requests=1)
    assert original.choose_neighbor(QUERY, CHOICES) == 1
    assert original.choose_neighbor(QUERY, CHOICES) == 1
    assert original.requests_made == 1 and original.cache_hits == 1
    restored = LLMClient(cache_path=cache, log_path=log, max_requests=0,
                         api_key_file=tmp_path / "deliberately_missing_key")
    assert restored.choose_neighbor(QUERY, CHOICES) == 1
    assert len(calls) == 1 and restored.requests_made == 0 and restored.cache_hits == 1
    assert not restored._key_loaded
    with pytest.raises(LLMBudgetExceeded):
        restored.choose_neighbor(QUERY + " new", CHOICES)
    assert read_jsonl(log)[-2]["status"] == "cache_hit"


@pytest.mark.parametrize("change", ["query", "order", "candidate", "model", "effort", "tokens", "endpoint"])
def test_cache_key_includes_all_semantic_parameters(monkeypatch, tmp_path, change):
    cache = tmp_path / "cache.jsonl"
    calls = fake_transport(monkeypatch, [FakeResponse(), FakeResponse()])
    LLMClient(api_key=KEY, cache_path=cache).choose_neighbor(QUERY, CHOICES)
    kwargs = {"api_key": KEY, "cache_path": cache}
    query, choices = QUERY, list(CHOICES)
    if change == "query":
        query += " changed"
    elif change == "order":
        choices.reverse()
    elif change == "candidate":
        choices[0] += " changed"
    elif change == "model":
        kwargs["model"] = "gpt-6-astra"
    elif change == "effort":
        kwargs["reasoning_effort"] = "low"
    elif change == "tokens":
        kwargs["max_output_tokens"] = 8192
    else:
        kwargs["base_url"] = "https://example.test/v1"
    second = LLMClient(**kwargs)
    second.choose_neighbor(query, choices)
    assert len(calls) == 2 and second.cache_hits == 0
    entries = read_jsonl(cache)
    assert len(entries) == 2 and entries[0]["key"] != entries[1]["key"]


def test_cache_and_usage_log_contain_no_credentials_or_prompts(monkeypatch, tmp_path):
    payload = completed()
    payload["usage"]["secret"] = KEY
    payload["usage"]["output_tokens_details"]["secret"] = QUERY
    payload["usage"]["input_tokens_details"]["invalid"] = -9
    fake_transport(monkeypatch, [FakeResponse(payload)])
    cache, log = tmp_path / "cache.jsonl", tmp_path / "log.jsonl"
    LLMClient(api_key=KEY, cache_path=cache, log_path=log).choose_neighbor(QUERY, CHOICES)
    text = cache.read_text(encoding="utf-8") + log.read_text(encoding="utf-8")
    assert all(secret not in text for secret in [KEY, QUERY] + CHOICES)
    entry = read_jsonl(log)[0]
    assert entry["response_id"] == "resp_offline_1"
    assert entry["response_model"] == "gpt-5.6-sol"
    assert entry["usage"] == completed()["usage"]
    assert entry["status"] == "completed" and entry["cache_hit"] is False
    assert entry["elapsed_seconds"] >= 0


@pytest.mark.parametrize("status", [400, 401, 403, 404, 301])
def test_permanent_http_failures_never_retry(monkeypatch, tmp_path, status):
    calls = fake_transport(monkeypatch, [FakeResponse({"error": {"message": KEY}}, status=status)])
    log = tmp_path / "log.jsonl"
    with pytest.raises(LLMError, match="HTTP {}".format(status)) as error:
        LLMClient(api_key=KEY, max_retries=9, log_path=log).choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 1
    assert KEY not in str(error.value) and KEY not in log.read_text(encoding="utf-8")


def test_retryable_http_failures_honor_bound_and_retry_after(monkeypatch, tmp_path):
    calls = fake_transport(monkeypatch, [FakeResponse(status=429, headers={"Retry-After": "10000"}),
                                         FakeResponse(status=503), FakeResponse()])
    delays = []
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    client = LLMClient(api_key=KEY, max_retries=2, log_path=tmp_path / "log.jsonl")
    assert client.choose_neighbor(QUERY, CHOICES) == 1
    assert delays == [60, 2]
    assert len(calls) == 3 and client.requests_made == 3
    assert all(call["json"]["model"] == "gpt-5.6-sol" for call in calls)
    assert [entry["status"] for entry in read_jsonl(tmp_path / "log.jsonl")] == [
        "http_error", "http_error", "completed"]


def test_exhausted_retries_stop_and_attempt_cap_includes_retries(monkeypatch):
    delays = []
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    calls = fake_transport(monkeypatch, [FakeResponse(status=429), FakeResponse(status=429)])
    client = LLMClient(api_key=KEY, max_retries=1)
    with pytest.raises(LLMError, match="HTTP 429"):
        client.choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 2 and delays == [1]
    calls = fake_transport(monkeypatch, [FakeResponse(status=503)])
    client = LLMClient(api_key=KEY, max_requests=1, max_retries=2)
    with pytest.raises(LLMBudgetExceeded):
        client.choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 1 and client.requests_made == 1


@pytest.mark.parametrize("exception", [requests.exceptions.Timeout, requests.exceptions.ConnectionError])
def test_network_failures_retry_without_leaking_exception_text(monkeypatch, exception):
    delays = []
    monkeypatch.setattr(llm_client.time, "sleep", delays.append)
    calls = fake_transport(monkeypatch, [exception(KEY), exception(KEY)])
    with pytest.raises(LLMError, match="after 2 attempts") as error:
        LLMClient(api_key=KEY, max_retries=1).choose_neighbor(QUERY, CHOICES)
    assert KEY not in str(error.value) and len(calls) == 2 and delays == [1]


def test_other_request_failure_does_not_retry(monkeypatch):
    calls = fake_transport(monkeypatch, [requests.exceptions.InvalidURL(KEY)])
    with pytest.raises(LLMError, match="could not be sent") as error:
        LLMClient(api_key=KEY).choose_neighbor(QUERY, CHOICES)
    assert KEY not in str(error.value) and len(calls) == 1


@pytest.mark.parametrize("outcome", [FakeResponse(json_error=True), FakeResponse(payload=["not an object"])])
def test_invalid_http_json_rejected(monkeypatch, outcome):
    calls = fake_transport(monkeypatch, [outcome])
    with pytest.raises(LLMError) as error:
        LLMClient(api_key=KEY).choose_neighbor(QUERY, CHOICES)
    assert len(calls) == 1 and KEY not in str(error.value)


def test_key_precedence_and_lazy_loading(monkeypatch, tmp_path):
    path = tmp_path / "key.txt"
    path.write_text("file-key\n", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    calls = fake_transport(monkeypatch, [FakeResponse(), FakeResponse(), FakeResponse()])
    explicit = LLMClient(api_key=KEY, api_key_file=path)
    assert not explicit._key_loaded
    explicit.choose_neighbor(QUERY, CHOICES)
    LLMClient(api_key_file=path).choose_neighbor(QUERY, CHOICES)
    LLMClient().choose_neighbor(QUERY, CHOICES)
    assert [call["headers"]["Authorization"] for call in calls] == [
        "Bearer " + KEY, "Bearer file-key", "Bearer env-key"]


def test_missing_or_empty_explicit_key_does_not_fall_back_to_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    with pytest.raises(LLMError, match="Cannot read"):
        LLMClient(api_key_file=tmp_path / "missing").choose_neighbor(QUERY, CHOICES)
    with pytest.raises(LLMError, match="empty"):
        LLMClient(api_key=" ").choose_neighbor(QUERY, CHOICES)


def test_cache_log_and_key_paths_cannot_alias(tmp_path):
    path = tmp_path / "one-file"
    with pytest.raises(ValueError, match="different files"):
        LLMClient(cache_path=path, log_path=path)
    for name in ("cache_path", "log_path"):
        with pytest.raises(ValueError, match="API key file"):
            LLMClient(api_key_file=path, **{name: path})


def test_corrupt_cache_is_rejected_and_cached_answer_is_revalidated(monkeypatch, tmp_path):
    path = tmp_path / "cache.jsonl"
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(LLMError, match="Invalid JSONL cache record"):
        LLMClient(cache_path=path)
    path.write_text("", encoding="utf-8")
    fake_transport(monkeypatch, [FakeResponse()])
    LLMClient(api_key=KEY, cache_path=path).choose_neighbor(QUERY, CHOICES)
    entry = read_jsonl(path)[0]
    entry["answer"] = {"choice": 100}
    path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    with pytest.raises(LLMError, match="invalid zero-based choice"):
        LLMClient(cache_path=path, max_requests=0).choose_neighbor(QUERY, CHOICES)


def test_optional_cluster_name_is_structured_and_trimmed(monkeypatch):
    calls = fake_transport(monkeypatch, [FakeResponse(completed({"name": "  located in  "}))])
    assert LLMClient(api_key=KEY).name_cluster(CHOICES) == "located in"
    payload = calls[0]["json"]
    assert payload["text"]["format"]["name"] == "name_cluster"
    assert payload["text"]["format"]["schema"]["required"] == ["name"]


@pytest.mark.parametrize("name", ["", "  ", "x" * 121, None, 1])
def test_invalid_cluster_names_rejected(monkeypatch, name):
    fake_transport(monkeypatch, [FakeResponse(completed({"name": name}))])
    with pytest.raises(LLMError, match="relation name"):
        LLMClient(api_key=KEY).name_cluster(CHOICES)
