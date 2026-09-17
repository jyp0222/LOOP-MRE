import hashlib
import json
from collections import Counter

import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from mre_data import MREData
from mre_protocol import prepare_dataset


class FakeTokenizer:
    def __init__(self, token_types=True):
        self.token_types = token_types
        self.calls = []

    def __call__(self, texts, padding, truncation, max_length, return_tensors):
        assert padding == "max_length" and truncation and return_tensors == "pt"
        self.calls.append(list(texts))
        ids = torch.zeros((len(texts), max_length), dtype=torch.long)
        masks = torch.zeros_like(ids)
        for index, text in enumerate(texts):
            length = min(len(text.split()), max_length)
            ids[index, :length] = torch.arange(1, length + 1)
            masks[index, :length] = 1
        result = {"input_ids": ids, "attention_mask": masks}
        if self.token_types:
            result["token_type_ids"] = torch.zeros_like(ids)
        return result


@pytest.fixture
def prepared(tmp_path):
    source = tmp_path / "source"
    source.mkdir()

    def record(relation, index):
        return {"tokens": ["Head", "relates", "Tail", str(index)],
                "h": {"name": "Head", "pos": [0]},
                "t": {"name": "Tail", "pos": [2]},
                "relation": relation, "img_id": "nonexistent.jpg"}

    for filename, class_sizes in (("train.txt", [("base_a", 10), ("base_b", 15)]),
                                  ("test.txt", [("novel", 6)])):
        rows = [record(relation, "{}_{}".format(relation, i))
                for relation, count in class_sizes for i in range(count)]
        (source / filename).write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    output = tmp_path / "prepared"
    prepare_dataset(source, output, expected_base=2, expected_novel=1)
    return output


def rewrite_split(prepared, name, edit):
    """Simulate a malicious/mistaken but internally rehashed prepared input."""
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    filename = manifest["files"][name]
    path = prepared / filename
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    edit(rows)
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    path.write_bytes(payload)
    manifest["file_sha256"][filename] = hashlib.sha256(payload).hexdigest()
    manifest["splits"][name] = {"count": len(rows), "ids": [row["id"] for row in rows]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_loaders_preserve_unlabeled_boundary_and_evaluation_order(prepared):
    tokenizer = FakeTokenizer()
    data = MREData(prepared, tokenizer, max_length=12, eval_batch_size=7)
    assert (data.n_base, data.n_total) == (2, 3)
    assert len(tokenizer.calls) == 3  # test text reuses unlabeled tokenization
    assert data.labeled_dataset.tensors[0].shape == (9, 12)
    assert data.validation_dataset.tensors[0].shape == (3, 12)
    assert data.test_dataset.tensors[0].shape == (19, 12)
    assert all(set(row) == {"id", "text"} for row in data.unlabeled_records)
    assert torch.equal(data.unlabeled_dataset.tensors[3], torch.full((19,), -1))
    assert torch.equal(data.semi_dataset.tensors[3][:9], data.labeled_dataset.tensors[3])
    assert torch.equal(data.semi_dataset.tensors[3][9:], torch.full((19,), -1))
    assert data.semi_records == data.labeled_records + data.unlabeled_records
    for loader, dataset in ((data.test_loader, data.test_dataset),
                            (data.validation_loader, data.validation_dataset),
                            (data.semi_loader, data.semi_dataset)):
        batches = list(loader)
        assert sum(len(batch[0]) for batch in batches) == len(dataset)
        for column in range(4):
            assert torch.equal(torch.cat([batch[column] for batch in batches]), dataset.tensors[column])
    assert set(data.test_dataset.tensors[3].tolist()) == {0, 1, 2}
    assert set(torch.cat([batch[3] for batch in data.labeled_loader]).tolist()) == {0, 1}


def test_missing_token_type_ids_get_zeros(prepared):
    data = MREData(prepared, FakeTokenizer(token_types=False), max_length=8)
    assert data.semi_dataset.tensors[2].shape[1] == 8
    assert not data.semi_dataset.tensors[2].any()


def test_original_random_sampling_uses_global_torch_rng_without_class_balancing(prepared):
    data = MREData(prepared, FakeTokenizer())
    labels = data.labeled_dataset.tensors[3].tolist()
    sampler = data.labeled_loader.sampler
    assert isinstance(sampler, RandomSampler)
    assert sampler.generator is None and not sampler.replacement
    assert data.labeled_loader.generator is None
    assert isinstance(data.unlabeled_loader.sampler, RandomSampler)
    assert data.unlabeled_loader.sampler.generator is None
    assert data.unlabeled_loader.generator is None
    torch.manual_seed(11)
    epoch_1, epoch_2 = list(sampler), list(sampler)
    torch.manual_seed(11)
    assert epoch_1 == list(sampler)
    assert epoch_2 == list(sampler)
    assert epoch_1 != epoch_2
    assert sorted(epoch_1) == list(range(len(labels)))
    assert Counter(labels[index] for index in epoch_1) == Counter(labels)
    assert len(set(Counter(labels).values())) > 1


def test_original_loader_batch_sizes_and_sequential_semi_pool(prepared):
    data = MREData(prepared, FakeTokenizer())
    assert data.labeled_loader.batch_size == 64
    assert data.unlabeled_loader.batch_size == 128
    assert data.validation_loader.batch_size == data.test_loader.batch_size == 64
    assert data.semi_loader.batch_size == 128
    assert isinstance(data.semi_loader.sampler, SequentialSampler)
    configured = MREData(prepared, FakeTokenizer(), train_batch_size=5, eval_batch_size=7)
    assert configured.semi_loader.batch_size == 5
    assert configured.validation_loader.batch_size == 7


def test_accidental_true_labels_in_unlabeled_file_rejected(prepared):
    rewrite_split(prepared, "train_unlabeled", lambda rows: rows[0].update(label=0))
    with pytest.raises(ValueError, match="exactly"):
        MREData(prepared, FakeTokenizer())


def test_novel_label_cannot_enter_labeled_training(prepared):
    rewrite_split(prepared, "train_labeled", lambda rows: rows[0].update(label=2))
    with pytest.raises(ValueError, match="outside its permitted classes"):
        MREData(prepared, FakeTokenizer())


def test_unlabeled_and_test_inputs_must_match(prepared):
    rewrite_split(prepared, "train_unlabeled", lambda rows: rows[0].update(text="changed text"))
    with pytest.raises(ValueError, match="must equal test inputs"):
        MREData(prepared, FakeTokenizer())


def test_tampered_prepared_file_rejected(prepared):
    path = prepared / "train_labeled.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        MREData(prepared, FakeTokenizer())


def test_invalid_loader_parameters_rejected(prepared):
    with pytest.raises(ValueError, match="eval_batch_size must be a positive integer"):
        MREData(prepared, FakeTokenizer(), eval_batch_size=0)
