"""Text-only MRE data adapter; also reads immutable legacy fixed-class runs."""

import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset


def _read_split(directory, manifest, split, labeled, n_classes):
    filename = manifest["files"][split]
    path = (directory / filename).resolve()
    if path.parent != directory.resolve():
        raise ValueError("prepared split file must be directly inside prepared_dir")
    payload = path.read_bytes()
    expected_hash = manifest["file_sha256"][filename]
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError("prepared split hash mismatch: {}".format(filename))
    rows = []
    keys = {"id", "text", "label"} if labeled else {"id", "text"}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("{}:{} must contain exactly {}".format(filename, line_number, sorted(keys)))
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("{}:{} has an invalid sample ID".format(filename, line_number))
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("{}:{} has empty relation text".format(filename, line_number))
        if labeled and (isinstance(row["label"], bool) or not isinstance(row["label"], int)
                        or not 0 <= row["label"] < n_classes):
            raise ValueError("{}:{} label is outside its permitted classes".format(filename, line_number))
        rows.append(row)
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("{} has repeated sample IDs".format(filename))
    if len(rows) != manifest["splits"][split]["count"] or ids != manifest["splits"][split]["ids"]:
        raise ValueError("{} count or sample order disagrees with manifest".format(filename))
    if not rows:
        raise ValueError("{} is empty".format(filename))
    return rows


def _tensor_dataset(records, tokenizer, max_length, labeled):
    encoded = tokenizer(
        [row["text"] for row in records],
        padding="max_length", truncation=True, max_length=max_length,
        return_tensors="pt",
    )
    input_ids = torch.as_tensor(encoded["input_ids"], dtype=torch.long)
    attention_mask = torch.as_tensor(encoded["attention_mask"], dtype=torch.long)
    token_type_ids = torch.as_tensor(
        encoded.get("token_type_ids", torch.zeros_like(input_ids)), dtype=torch.long
    )
    expected_shape = (len(records), max_length)
    for tensor in (input_ids, attention_mask, token_type_ids):
        if tuple(tensor.shape) != expected_shape:
            raise ValueError("tokenizer must return tensors of shape {}".format(expected_shape))
    labels = torch.tensor([row["label"] if labeled else -1 for row in records], dtype=torch.long)
    return TensorDataset(input_ids, attention_mask, token_type_ids, labels)


class MREData:
    """Keep evaluation labels separate from every unlabeled training tensor.

    The test text pool intentionally equals the unlabeled pool. The labeled
    train/validation/test sample IDs must otherwise be disjoint. Duplicate
    *text* with different source IDs is retained according to the manifest.
    """

    def __init__(self, prepared_dir, tokenizer, max_length=128,
                 labeled_batch_size=64, train_batch_size=128, eval_batch_size=64, seed=0,
                 caption_records=None):
        self.prepared_dir = Path(prepared_dir)
        for name, value in (("max_length", max_length), ("labeled_batch_size", labeled_batch_size),
                            ("train_batch_size", train_batch_size), ("eval_batch_size", eval_batch_size)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("{} must be a positive integer".format(name))
        self.manifest = json.loads((self.prepared_dir / "manifest.json").read_text(encoding="utf-8"))
        protocol = self.manifest.get("protocol")
        if protocol not in ("mre_transductive_random_half", "mre_transductive_fixed_base_novel"):
            raise ValueError("unsupported MRE data protocol")
        self.n_base = self.manifest["n_base"]
        self.n_total = self.manifest["n_total"]
        classes = self.manifest["base_classes"] + self.manifest["novel_classes"]
        if (self.n_base != len(self.manifest["base_classes"])
                or self.n_total != len(classes) or not 0 < self.n_base < self.n_total
                or len(set(classes)) != len(classes) or self.manifest["classes"] != classes
                or self.manifest["class_to_id"] != {name: index for index, name in enumerate(classes)}):
            raise ValueError("manifest class order or class count is inconsistent")
        if protocol == "mre_transductive_random_half" and self.n_base != self.n_total // 2:
            raise ValueError("random-half protocol requires floor(n_total / 2) base classes")
        self.labeled_records = _read_split(self.prepared_dir, self.manifest, "train_labeled", True, self.n_base)
        self.unlabeled_records = _read_split(self.prepared_dir, self.manifest, "train_unlabeled", False, self.n_total)
        self.validation_records = _read_split(self.prepared_dir, self.manifest, "validation", True, self.n_base)
        self.test_records = _read_split(self.prepared_dir, self.manifest, "test", True, self.n_total)
        expected_unlabeled = [{"id": row["id"], "text": row["text"]} for row in self.test_records]
        if self.unlabeled_records != expected_unlabeled:
            raise ValueError("unlabeled records must equal test inputs in the same order, with labels removed")
        id_groups = [set(row["id"] for row in records) for records in
                     (self.labeled_records, self.validation_records, self.test_records)]
        if any(id_groups[i] & id_groups[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("labeled, validation and test sample IDs must be disjoint")
        for name, records, count in (("labeled", self.labeled_records, self.n_base),
                                     ("validation", self.validation_records, self.n_base),
                                     ("test", self.test_records, self.n_total)):
            if {row["label"] for row in records} != set(range(count)):
                raise ValueError("{} records do not cover all expected classes".format(name))

        self.semi_records = self.labeled_records + self.unlabeled_records
        def encoder_rows(records):
            if caption_records is None:
                return records
            return [dict(row, text=caption_records[row['id']]['text']) for row in records]

        if caption_records is not None:
            from mre_captions import SUFFIX, validate_caption
            all_rows = self.semi_records + self.validation_records
            if set(caption_records) != {row['id'] for row in all_rows}:
                raise ValueError('Caption sample IDs differ from prepared dataset')
            for row in all_rows:
                saved = caption_records[row['id']]
                if (saved['original_text'] != row['text'] or
                        saved['text'] != row['text'] + SUFFIX + validate_caption(saved['caption'])):
                    raise ValueError('Caption original/augmented text mismatch: {}'.format(row['id']))
            # The LLM keeps exactly the baseline truncated/decoded original input.
            # Sampling and query logic remain untouched; only encoder input changes.
            self.llm_input_ids = _tensor_dataset(self.semi_records, tokenizer, max_length, False).tensors[0]
        self.labeled_dataset = _tensor_dataset(encoder_rows(self.labeled_records), tokenizer, max_length, True)
        self.unlabeled_dataset = _tensor_dataset(encoder_rows(self.unlabeled_records), tokenizer, max_length, False)
        self.validation_dataset = _tensor_dataset(encoder_rows(self.validation_records), tokenizer, max_length, True)
        # Test and unlabeled inputs are identical; share their tensors to avoid
        # duplicate tokenization. Only this evaluation dataset gets true labels.
        self.test_dataset = TensorDataset(
            *self.unlabeled_dataset.tensors[:3],
            torch.tensor([row["label"] for row in self.test_records], dtype=torch.long)
        )
        self.semi_dataset = TensorDataset(*[
            torch.cat((labeled, unlabeled), dim=0) for labeled, unlabeled in
            zip(self.labeled_dataset.tensors, self.unlabeled_dataset.tensors)
        ])
        self.labeled_loader = DataLoader(
            self.labeled_dataset, batch_size=labeled_batch_size,
            sampler=RandomSampler(self.labeled_dataset),
            num_workers=0, drop_last=False,
        )
        self.unlabeled_loader = DataLoader(
            self.unlabeled_dataset, batch_size=train_batch_size,
            sampler=RandomSampler(self.unlabeled_dataset), num_workers=0, drop_last=False,
        )
        self.validation_loader = DataLoader(self.validation_dataset, batch_size=eval_batch_size,
                                            sampler=SequentialSampler(self.validation_dataset),
                                            num_workers=0, drop_last=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=eval_batch_size,
                                      sampler=SequentialSampler(self.test_dataset),
                                      num_workers=0, drop_last=False)
        self.semi_loader = DataLoader(self.semi_dataset, batch_size=train_batch_size,
                                      sampler=SequentialSampler(self.semi_dataset),
                                      num_workers=0, drop_last=False)
