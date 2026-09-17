"""Audited Chat Completions transport for the original LOOP prompts.

Uses requests rather than the OpenAI SDK so the original Python 3.8 / openai
0.28 training environment can be retained. No network call or key prompt occurs
on import, construction, or a cache hit. This client is intended for the main
training process (DataLoader num_workers=0), not concurrent cache writers.

Cache keys include the complete request, endpoint, and prompt version. Cache
records contain the selected index/name and response metadata, not input texts or
API keys. Neighbor API/parse failures use LOOP's original first-candidate fallback,
which is counted, logged, and never cached. No automatic model switch occurs.
Local configuration errors and request-budget exhaustion always stop the run.
max_requests caps HTTP attempts per client instance, including failed attempts
and retries; it is not a monetary limit. Restarting creates a new attempt budget.

The requested 0301 model identifier does not verify the proxy's backend snapshot.
The returned model identifier and any mismatch are recorded separately.
"""

import copy
from datetime import datetime, timezone
from getpass import getpass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import warnings
from urllib.parse import urlparse

import requests


PROMPT_VERSION = "loop-original-intent-choice-chat-v3"
SUPPORTED_MODELS = ("gpt-3.5-turbo-0301", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-3.5-turbo-1106")


class LLMError(RuntimeError):
    """A request or response failed."""


class LLMConfigurationError(LLMError):
    """A local configuration/key problem that must not become supervision."""


class LLMBudgetExceeded(LLMError):
    """The configured HTTP-attempt cap has been reached."""


class LLMClient:
    """Original LOOP text prompts/parsing with JSONL caching and audit logs.

    Key precedence: explicit api_key, explicit api_key_file, OPENAI_API_KEY,
    then an interactive hidden prompt. An explicitly configured empty/missing
    key file is an error, not permission to silently use another credential.
    The key is read only on the first uncached request.
    """

    def __init__(
        self,
        model="gpt-3.5-turbo-0301",
        reasoning_effort=None,
        api_key=None,
        api_key_file=None,
        cache_path=None,
        log_path=None,
        max_output_tokens=None,
        timeout=120,
        max_retries=0,
        max_requests=None,
        base_url="https://api.openai.com/v1",
        naming_model="gpt-3.5-turbo",
    ):
        if model not in SUPPORTED_MODELS or naming_model not in SUPPORTED_MODELS:
            raise ValueError("model must be a supported GPT-3.5 Turbo model: {}".format(
                ", ".join(SUPPORTED_MODELS)))
        if reasoning_effort not in (None, "none"):
            raise ValueError("GPT-3.5 Turbo does not support reasoning_effort; use None")
        for name, value, minimum in (
            ("max_output_tokens", max_output_tokens, 1),
            ("max_retries", max_retries, 0),
            ("max_requests", max_requests, 0),
        ):
            if value is None and name in ("max_requests", "max_output_tokens"):
                continue
            if type(value) is not int or value < minimum:
                raise ValueError("{} must be an integer >= {}".format(name, minimum))
        if max_output_tokens is not None and max_output_tokens > 4096:
            raise ValueError("GPT-3.5 Turbo max_output_tokens must be <= 4096")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive number")
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("timeout must be a positive finite number")
        endpoint = base_url.rstrip("/") + "/chat/completions"
        url = urlparse(endpoint)
        if url.scheme != "https" or not url.netloc or url.username or url.password:
            raise ValueError("base_url must be an HTTPS API base without embedded credentials")
        if url.query or url.fragment:
            raise ValueError("base_url must not contain a query string or fragment")

        self.model = model
        self.naming_model = naming_model
        # Kept as null in run metadata for compatibility; never sent to the API.
        self.reasoning_effort = None
        self.last_response_model = None
        self.last_response_model_matches_request = None
        self.snapshot_verified = False
        self.response_models = set()
        self.model_mismatch_count = 0
        self._warned_model_mismatches = set()
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_requests = max_requests
        self.endpoint = endpoint
        self.cache_path = Path(cache_path) if cache_path else None
        self.log_path = Path(log_path) if log_path else None
        self._api_key = api_key
        self._key_loaded = False
        self._api_key_file = Path(api_key_file).expanduser() if api_key_file else None
        if self.cache_path and self.log_path:
            if self.cache_path.resolve() == self.log_path.resolve():
                raise ValueError("cache_path and log_path must be different files")
        for output_path in (self.cache_path, self.log_path):
            if output_path and self._api_key_file:
                if output_path.resolve() == self._api_key_file.resolve():
                    raise ValueError("Cache/log file cannot be the API key file")
        self.requests_made = 0
        self.cache_hits = 0
        self.fallback_count = 0
        self.last_query_fallback = False
        self._cache = {}
        self._load_cache()

    def choose_neighbor(self, query, choices):
        """Return index 0/1 using the original prompt and Choice 1 precedence.

        The caller supplies the same tokenizer-decoded strings used upstream.
        API/malformed-answer failures select index 0, matching upstream q1.
        """
        self._check_text(query, "query")
        self._check_text_list(choices, "choices", count=2)
        prompt = (
            "Select the customer utterance that better corresponds with the Query in terms of intent. "
            "Please respond with 'Choice 1' or 'Choice 2' without explanation. \n Query: "
            + query + "\n Choice 1: " + choices[0] + "\n Choice 2: " + choices[1]
        )
        return self._query("choose_neighbor", [{"role": "user", "content": prompt}])["choice"]

    def name_cluster(self, samples):
        """Return the raw naming text for the original three-utterance prompt."""
        self._check_text_list(samples, "samples", count=3)
        prompt = (
            "Given the following customer utterances, return a word or a phrase to summarize "
            "the common intent of these utterances without explanation. \n Utterance 1: "
            + samples[0] + "\n Utterance 2: " + samples[1] + "\n Utterance 3: " + samples[2]
        )
        return self._query("name_cluster", [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ])["name"]

    @staticmethod
    def _check_text(value, name):
        if not isinstance(value, str) or not value.strip():
            raise ValueError("{} must be a nonempty string".format(name))

    @classmethod
    def _check_text_list(cls, values, name, count):
        if not isinstance(values, (list, tuple)) or len(values) != count:
            raise ValueError("{} must contain exactly {} texts".format(name, count))
        for value in values:
            cls._check_text(value, name)

    def _load_key(self):
        if self._key_loaded:
            return self._api_key
        key = self._api_key
        if key is None and self._api_key_file is not None:
            try:
                key = self._api_key_file.read_text(encoding="utf-8-sig")
            except OSError:
                raise LLMConfigurationError("Cannot read configured OpenAI API key file") from None
        if key is None:
            key = os.environ.get("OPENAI_API_KEY")
        if key is None:
            try:
                key = getpass("OpenAI API key (input hidden): ")
            except (EOFError, OSError):
                raise LLMConfigurationError("Set api_key_file or OPENAI_API_KEY for a noninteractive run") from None
        if not isinstance(key, str) or not key.strip():
            raise LLMConfigurationError("OpenAI API key is empty")
        self._api_key = key.strip()
        self._key_loaded = True
        return self._api_key

    def _load_cache(self):
        if self.cache_path is None or not self.cache_path.exists():
            return
        with self.cache_path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    key = entry["key"]
                    if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
                        raise ValueError("bad key")
                    if entry.get("version") != PROMPT_VERSION:
                        continue
                    if not isinstance(entry["answer"], dict):
                        raise ValueError("bad answer")
                    self._cache[key] = entry
                except (ValueError, KeyError, TypeError):
                    raise LLMError("Invalid JSONL cache record at line {}; repair the cache before retrying".format(lineno)) from None

    @staticmethod
    def _append_jsonl(path, record):
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()

    @staticmethod
    def _safe_identifier(value):
        if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,160}", value):
            return value
        return None

    @classmethod
    def _safe_usage(cls, usage):
        if not isinstance(usage, dict):
            return {}
        allowed = {"input_tokens", "output_tokens", "total_tokens", "cached_tokens",
                   "reasoning_tokens", "input_tokens_details", "output_tokens_details",
                   "prompt_tokens", "completion_tokens", "prompt_tokens_details",
                   "completion_tokens_details"}
        return {k: (cls._safe_usage(v) if isinstance(v, dict) else v)
                for k, v in usage.items()
                if k in allowed and (isinstance(v, dict) or (type(v) is int and v >= 0))}

    def _log(self, key, task, status, elapsed=0.0, response=None, **fields):
        response = response if isinstance(response, dict) else {}
        requested_model = self.naming_model if task == "name_cluster" else self.model
        response_model = self._safe_identifier(response.get("model"))
        record = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "request_key": key,
            "task": task,
            "status": status,
            "requested_model": requested_model,
            "reasoning_effort": self.reasoning_effort,
            "response_model": response_model,
            "model_matches_request": response_model == requested_model if response_model else None,
            "snapshot_verified": False,
            "response_id": self._safe_identifier(response.get("id")),
            "usage": self._safe_usage(response.get("usage")),
            "elapsed_seconds": round(elapsed, 6),
            "requests_made": self.requests_made,
            "fallback_count": self.fallback_count,
        }
        record.update(fields)
        self._append_jsonl(self.log_path, record)

    def _query(self, task, messages):
        self.last_query_fallback = False
        self.last_response_model = None
        self.last_response_model_matches_request = None
        payload = {
            "model": self.naming_model if task == "name_cluster" else self.model,
            "messages": messages,
        }
        # Upstream supplied only model/messages. An explicit override is recorded
        # in the cache key; the default request retains upstream API defaults.
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens
        canonical = json.dumps({"version": PROMPT_VERSION, "endpoint": self.endpoint,
                                "payload": payload}, sort_keys=True, ensure_ascii=False,
                               separators=(",", ":"))
        key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        if key in self._cache:
            entry = self._cache[key]
            answer = self._validate_answer(entry["answer"], task)
            metadata = entry.get("metadata") or {}
            self._validate_response_model(metadata)
            self._record_response_model(metadata, payload["model"])
            self.cache_hits += 1
            self._log(key, task, "cache_hit", response=entry.get("metadata"), cache_hit=True)
            return copy.deepcopy(answer)

        started = time.monotonic()
        response = {}
        try:
            response = self._post(payload, key, task)
            self._record_response_model(response, payload["model"])
            answer = self._parse_response(response, task)
        except (LLMBudgetExceeded, LLMConfigurationError):
            raise
        except LLMError as error:
            if task != "choose_neighbor":
                self._log(key, task, "failed", time.monotonic() - started, response,
                          error=str(error), cache_hit=False)
                raise
            self.fallback_count += 1
            self.last_query_fallback = True
            self._log(key, task, "neighbor_fallback", time.monotonic() - started, response,
                      error=str(error), choice=0, fallback="original_q1", cache_hit=False)
            warnings.warn("LOOP neighbor query failed; using original q1 fallback ({})".format(error),
                          RuntimeWarning, stacklevel=3)
            return {"choice": 0}
        metadata = {"id": self._safe_identifier(response.get("id")),
                    "model": self._safe_identifier(response.get("model")),
                    "usage": self._safe_usage(response.get("usage"))}
        entry = {"version": PROMPT_VERSION, "key": key, "answer": answer, "metadata": metadata}
        self._append_jsonl(self.cache_path, entry)
        self._cache[key] = entry
        self._log(key, task, "completed", time.monotonic() - started, response, cache_hit=False)
        return copy.deepcopy(answer)

    def _post(self, payload, key, task):
        for attempt in range(self.max_retries + 1):
            if self.max_requests is not None and self.requests_made >= self.max_requests:
                self._log(key, task, "budget_exceeded", attempt=attempt + 1, cache_hit=False)
                raise LLMBudgetExceeded("OpenAI max_requests={} reached (HTTP attempts, including retries); training stopped without fallback".format(self.max_requests))
            api_key = self._load_key()
            started = time.monotonic()
            self.requests_made += 1
            try:
                result = requests.post(
                    self.endpoint,
                    headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout,
                    allow_redirects=False,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                self._log(key, task, "network_error", time.monotonic() - started,
                          attempt=attempt + 1, cache_hit=False)
                if attempt == self.max_retries:
                    raise LLMError("OpenAI network/timeout failure after {} attempts".format(attempt + 1)) from None
                time.sleep(min(2 ** attempt, 30))
                continue
            except (requests.exceptions.InvalidURL, requests.exceptions.InvalidSchema,
                    requests.exceptions.MissingSchema, requests.exceptions.InvalidHeader):
                self._log(key, task, "configuration_error", time.monotonic() - started,
                          attempt=attempt + 1, cache_hit=False)
                raise LLMConfigurationError("OpenAI request could not be sent; check HTTPS/API configuration") from None
            except requests.exceptions.RequestException:
                self._log(key, task, "request_error", time.monotonic() - started,
                          attempt=attempt + 1, cache_hit=False)
                raise LLMError("OpenAI request could not be sent; check HTTPS/API configuration") from None

            status = result.status_code
            if not 200 <= status < 300:
                self._log(key, task, "http_error", time.monotonic() - started,
                          http_status=status, attempt=attempt + 1, cache_hit=False)
                retryable = status == 429 or 500 <= status < 600
                if retryable and attempt < self.max_retries:
                    delay = min(2 ** attempt, 30)
                    try:
                        requested_delay = float(result.headers.get("Retry-After", ""))
                        if math.isfinite(requested_delay):
                            delay = min(max(requested_delay, 0), 60)
                    except (ValueError, TypeError):
                        pass
                    time.sleep(delay)
                    continue
                if status in (401, 403):
                    detail = "check API key, project permissions, and access to the configured model"
                elif status == 404:
                    detail = "check endpoint and account access to model {}".format(payload["model"])
                elif status == 429:
                    detail = "check account quota and rate limits"
                else:
                    detail = "check configured GPT-3.5 model and Chat Completions request schema"
                raise LLMError("OpenAI HTTP {}: {}; no model switch was used".format(status, detail))
            try:
                response = result.json()
            except ValueError:
                self._log(key, task, "invalid_json", time.monotonic() - started, cache_hit=False)
                raise LLMError("OpenAI returned a non-JSON response") from None
            if not isinstance(response, dict):
                raise LLMError("OpenAI returned a JSON value instead of a response object")
            return response

    @classmethod
    def _parse_response(cls, response, task):
        cls._validate_response_model(response)
        if response.get("error"):
            raise LLMError("OpenAI response contains an error")
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMError("OpenAI response has no first completion choice")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LLMError("OpenAI response has no completion message")
        text = message.get("content")
        if not isinstance(text, str):
            raise LLMError("OpenAI answer content is not text")
        if task == "choose_neighbor":
            # Exact upstream, case-sensitive substring checks and precedence.
            if "Choice 1" in text:
                return {"choice": 0}
            if "Choice 2" in text:
                return {"choice": 1}
            raise LLMError("OpenAI answer contains neither Choice 1 nor Choice 2")
        if not text.strip():
            raise LLMError("OpenAI returned empty answer content")
        return {"name": text}

    def _record_response_model(self, response, requested_model):
        model = self._safe_identifier(response.get("model"))
        self.last_response_model = model
        self.last_response_model_matches_request = model == requested_model if model else None
        if model is not None:
            self.response_models.add(model)
        if model in SUPPORTED_MODELS and model != requested_model:
            self.model_mismatch_count += 1
            pair = (requested_model, model)
            if pair not in self._warned_model_mismatches:
                self._warned_model_mismatches.add(pair)
                warnings.warn("Requested {}; API reported {}. Backend snapshot is unverified; "
                              "no request model was changed.".format(*pair), RuntimeWarning, stacklevel=3)

    @staticmethod
    def _validate_response_model(response):
        if not isinstance(response, dict) or response.get("model") not in SUPPORTED_MODELS:
            raise LLMError("OpenAI response does not identify a supported GPT-3.5 Turbo model; "
                           "no answer was accepted")

    @staticmethod
    def _validate_answer(answer, task):
        required = "choice" if task == "choose_neighbor" else "name"
        if not isinstance(answer, dict) or set(answer) != {required}:
            raise LLMError("Cached answer does not match the task")
        if required == "choice":
            choice = answer["choice"]
            if type(choice) is not int or choice not in (0, 1):
                raise LLMError("Cached answer has an invalid zero-based choice index")
        else:
            name = answer["name"]
            if not isinstance(name, str) or not name.strip():
                raise LLMError("Cached answer has an empty or nontext name")
        return answer
