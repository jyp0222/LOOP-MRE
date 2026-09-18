"""Prepare text-only MRE with its default random-half base/novel classes.

This is TRANSDUCTIVE: test inputs are the unlabeled training pool. Their
ground-truth labels exist only in test.jsonl, never train_unlabeled.jsonl.
Read train.txt then test.txt, preserving first-appearance relation order.
One NumPy RandomState first shuffles relations, then each base class's rows,
matching MRE_learn.utils.split_types and data_loader.split_dataset.
"""

import ast
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


FILTERED_RELATIONS = frozenset(("Other", "None", "none", "NA"))
SCHEMA_VERSION = 3
PROTOCOL = "mre_transductive_random_half"
OUTPUT_FILES = {
    "train_labeled": "train_labeled.jsonl",
    "train_unlabeled": "train_unlabeled.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def _clean_text(value):
    return " ".join(value.split())


def _span(entity, tokens, position_format, name):
    if not isinstance(entity, dict):
        raise ValueError("{} must be an entity dictionary".format(name))
    pos = entity.get("pos")
    if not isinstance(pos, (list, tuple)) or not pos:
        raise ValueError("{}.pos must be a nonempty list".format(name))
    if any(isinstance(x, bool) or not isinstance(x, int) for x in pos):
        raise ValueError("{}.pos must contain integer token positions".format(name))
    if position_format == "indices":
        if list(pos) != list(range(pos[0], pos[-1] + 1)):
            raise ValueError("{}.pos must be sorted contiguous token indices".format(name))
        start, stop = pos[0], pos[-1] + 1
    else:
        if len(pos) != 2:
            raise ValueError("{}.pos must be [start, end) in half_open mode".format(name))
        start, stop = pos
    if not 0 <= start < stop <= len(tokens):
        raise ValueError("{}.pos is empty or outside token boundaries".format(name))
    if not _clean_text(" ".join(tokens[start:stop])):
        raise ValueError("{}.pos [{}, {}) contains only blank tokens; cannot locate entity text".format(
            name, start, stop))
    entity_name = entity.get("name")
    if entity_name is None or entity_name == "":
        entity_name = " ".join(tokens[start:stop])
    if not isinstance(entity_name, str) or not _clean_text(entity_name):
        raise ValueError("{}.name must be nonempty text when supplied".format(name))
    return start, stop, _clean_text(entity_name)


def format_relation_text(record, position_format="indices"):
    """Keep directed entities and sentence text, without ever opening an image.

    `indices` treats [13, 14] as TWO token positions. `half_open` treats it
    as ONE token. This choice is explicit; it is never inferred from length.
    Ordinary marker words survive tokenizer.decode(skip_special_tokens=True).
    """
    if position_format not in ("indices", "half_open"):
        raise ValueError("position_format must be 'indices' or 'half_open'")
    if not isinstance(record, dict):
        raise ValueError("each input row must be a dictionary")
    if "tokens" in record and "token" in record and record["tokens"] != record["token"]:
        raise ValueError("tokens and token disagree")
    tokens = record.get("tokens", record.get("token"))
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokens (or token) must be a nonempty list")
    for index, token in enumerate(tokens):
        if not isinstance(token, str):
            raise ValueError("tokens[{}] must be a string, got {}: {!r}".format(
                index, type(token).__name__, token))
    # Keep ALL original positions until entity markers have been inserted.
    # Filtering empty/whitespace tokens here would shift h.pos and t.pos.
    tokens = [_clean_text(token) for token in tokens]
    if not any(tokens):
        raise ValueError("sentence contains only blank tokens")
    hs, he, hn = _span(record.get("h"), tokens, position_format, "h")
    ts, te, tn = _span(record.get("t"), tokens, position_format, "t")
    marked = []
    for i, token in enumerate(tokens):
        if i == hs:
            marked.append("[HEAD]")
        if i == ts:
            marked.append("[TAIL]")
        if token:
            marked.append(token)
        if i + 1 == te:
            marked.append("[/TAIL]")
        if i + 1 == he:
            marked.append("[/HEAD]")
    return "Head: {}. Tail: {}. Sentence: {}".format(hn, tn, " ".join(marked))


def _read_source(source_path, position_format):
    raw = source_path.read_bytes()
    rows, classes = [], []
    filtered = Counter()
    nonempty_lines = 0
    blank_token_count = 0
    records_with_blank_tokens = 0
    blank_token_examples = []
    for line_number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        nonempty_lines += 1
        context = "{}:{}".format(source_path.name, line_number)
        try:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = ast.literal_eval(line)
            if not isinstance(record, dict):
                raise ValueError("each input row must be a dictionary")
            relation = record.get("relation")
            if not isinstance(relation, str) or not relation.strip():
                raise ValueError("relation must be nonempty text")
            if relation in FILTERED_RELATIONS:
                filtered[relation] += 1
                continue
            text = format_relation_text(record, position_format)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError("{}: {}".format(context, exc)) from exc
        if relation not in classes:
            classes.append(relation)
        tokens = record.get("tokens", record.get("token"))
        blank_positions = [i for i, token in enumerate(tokens) if not _clean_text(token)]
        if blank_positions:
            blank_token_count += len(blank_positions)
            records_with_blank_tokens += 1
            if len(blank_token_examples) < 20:
                blank_token_examples.append({"line": line_number, "positions": blank_positions})
        rows.append({
            "id": "{}:{:08d}".format(source_path.stem, line_number),
            "text": text,
            "relation": relation,
        })
    metadata = {
        "file": source_path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "nonempty_lines": nonempty_lines,
        "retained_records": len(rows),
        "filtered_records": sum(filtered.values()),
        "filtered_relations": dict(sorted(filtered.items())),
        "blank_token_count": blank_token_count,
        "records_with_blank_tokens": records_with_blank_tokens,
        "blank_token_examples": blank_token_examples,
    }
    return rows, classes, metadata


def _audit(rows, split_records):
    # Casefold agrees with the uncased backbone; records are retained unchanged.
    groups = defaultdict(list)
    for row in rows:
        groups[row["text"].casefold()].append(row)
    same_label_duplicates = 0
    conflicts = []
    for group in groups.values():
        labels = Counter(row["relation"] for row in group)
        same_label_duplicates += sum(n - 1 for n in labels.values())
        if len(labels) > 1:
            conflicts.append({
                "ids": [row["id"] for row in group],
                "relations": sorted(labels),
            })
    split_sets = {
        split: {row["text"].casefold() for row in values}
        for split, values in split_records.items()
    }
    return {
        "policy": "retain duplicates and conflicting labels; report only",
        "identity": "casefolded directed Head/Tail/marked-sentence text",
        "same_input_extra_records": sum(len(group) - 1 for group in groups.values()),
        "same_input_same_label_extra_records": same_label_duplicates,
        "conflicting_input_count": len(conflicts),
        "conflicting_record_count": sum(len(group["ids"]) for group in conflicts),
        "conflict_examples": conflicts[:20],
        "labeled_validation_shared_inputs": len(
            split_sets["train_labeled"] & split_sets["validation"]
        ),
        "labeled_test_shared_inputs": len(
            split_sets["train_labeled"] & split_sets["test"]
        ),
        "validation_test_shared_inputs": len(
            split_sets["validation"] & split_sets["test"]
        ),
    }


def prepare_dataset(source_dir, output_dir, seed=0, position_format="indices",
                    expected_total=80):
    """Write a new prepared directory and return its complete manifest dict.

    Existing output paths are always rejected, even with identical inputs;
    this prevents accidentally overwriting an experiment's split. Labels in
    labeled/validation/test files are GLOBAL integer IDs (base first).
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite prepared dataset: {}".format(output_dir))
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1]")
    if isinstance(expected_total, bool) or not isinstance(expected_total, int) or expected_total < 2:
        raise ValueError("expected_total must be an integer of at least 2")
    if position_format not in ("indices", "half_open"):
        raise ValueError("position_format must be 'indices' or 'half_open'")
    train_rows, train_classes, train_info = _read_source(source_dir / "train.txt", position_format)
    test_rows, test_classes, test_info = _read_source(source_dir / "test.txt", position_format)
    all_rows = train_rows + test_rows
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[row["relation"]].append(row)
    # Original MRE builds a dict by first occurrence while reading train then
    # test. Do NOT sort, treat file boundaries as labels, or reseed after this
    # shuffle: each changes the per-seed class/sample assignment.
    source_classes = list(grouped)
    if len(source_classes) != expected_total:
        raise ValueError("expected {} total relations after filtering; found {}".format(
            expected_total, len(source_classes)))
    rng = np.random.RandomState(seed)
    classes = source_classes.copy()
    rng.shuffle(classes)
    n_base = len(classes) // 2
    base_classes, novel_classes = classes[:n_base], classes[n_base:]
    class_to_id = {name: i for i, name in enumerate(classes)}
    splits = {key: [] for key in OUTPUT_FILES}
    class_counts = []

    def labeled(row):
        return {"id": row["id"], "text": row["text"], "label": class_to_id[row["relation"]]}

    for relation in base_classes:
        rows = grouped[relation]
        order = np.arange(len(rows))
        rng.shuffle(order)
        n_train_val = int(len(rows) * 0.5)
        n_train = int(n_train_val * 0.8)
        n_val = n_train_val - n_train
        n_test = len(rows) - n_train_val
        if min(n_train, n_val, n_test) < 1:
            raise ValueError("base class {!r} has {} records: exact MRE split gives "
                             "{}/{}/{} labeled/validation/unlabeled; each must be nonempty".format(
                                 relation, len(rows), n_train, n_val, n_test))
        splits["train_labeled"].extend(labeled(rows[i]) for i in order[:n_train])
        splits["validation"].extend(labeled(rows[i]) for i in order[n_train:n_train_val])
        splits["test"].extend(labeled(rows[i]) for i in order[n_train_val:])
        class_counts.append({"class": relation, "label": class_to_id[relation],
                             "source": len(rows), "train_labeled": n_train,
                             "validation": n_val, "test": n_test, "train_unlabeled": n_test})
    for relation in novel_classes:
        rows = grouped[relation]
        splits["test"].extend(labeled(row) for row in rows)
        class_counts.append({"class": relation, "label": class_to_id[relation],
                             "source": len(rows), "train_labeled": 0,
                             "validation": 0, "test": len(rows), "train_unlabeled": len(rows)})
    splits["train_unlabeled"] = [
        {"id": row["id"], "text": row["text"]} for row in splits["test"]
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "description": "Test inputs are used as unlabeled training data; their true labels "
                       "are reserved for evaluation. Validation IDs never enter training. "
                       "All base training records are labeled; no extra 10% label sampling.",
        "seed": seed,
        "rng": "numpy.random.RandomState(seed); shuffle relations once, then shuffle each "
               "base class in that order with the SAME RNG, without reseeding",
        "class_split": {
            "strategy": "random_half",
            "source_order": ["train.txt", "test.txt"],
            "relations_before_shuffle": source_classes,
            "source_train_classes": train_classes,
            "source_test_classes": test_classes,
            "base_count_rule": "len(relations) // 2",
            "reference": "MRE_learn.utils.split_types + data_loader.split_dataset",
        },
        "position_format": position_format,
        "blank_token_policy": "Keep original token positions for entity spans; omit blank strings "
                              "only when rendering the marked sentence. Retain records; reject "
                              "non-string tokens and entirely blank entity spans.",
        "base_classes": base_classes,
        "novel_classes": novel_classes,
        "classes": classes,
        "class_to_id": class_to_id,
        "n_base": len(base_classes),
        "n_novel": len(novel_classes),
        "n_total": len(classes),
        "sources": {"train": train_info, "test": test_info},
        "files": dict(OUTPUT_FILES),
        "splits": {name: {"count": len(rows), "ids": [row["id"] for row in rows]}
                   for name, rows in splits.items()},
        "class_counts": class_counts,
        "audit": _audit(all_rows, splits),
    }
    manifest["audit"].update({
        "blank_token_count": train_info["blank_token_count"] + test_info["blank_token_count"],
        "records_with_blank_tokens": (train_info["records_with_blank_tokens"]
                                      + test_info["records_with_blank_tokens"]),
    })
    # Input/validation failures above create no output. Exclusive directory
    # creation also protects against an output appearing while preparing.
    output_dir.mkdir(parents=True, exist_ok=False)
    file_hashes = {}
    for split, filename in OUTPUT_FILES.items():
        data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                       for row in splits[split]).encode("utf-8")
        (output_dir / filename).write_bytes(data)
        file_hashes[filename] = hashlib.sha256(data).hexdigest()
    manifest["file_sha256"] = file_hashes
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return manifest
