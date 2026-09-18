import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mre_protocol import format_relation_text, prepare_dataset


def sample(relation, index):
    # Deliberately no img_id or image directory: text-only loading must work.
    return {"tokens": ["Person{}".format(index), "visited", "New", "York"],
            "h": {"name": "Person{}".format(index), "pos": [0]},
            "t": {"name": "New York", "pos": [2, 3]}, "relation": relation}


def write_rows(path, rows):
    path.write_text("\n".join(repr(row) for row in rows) + "\n", encoding="utf-8")


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MREProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, base_count=2, novel_count=2, base_n=5, novel_n=5):
        # These names denote source files, not eventual Base/Novel identities.
        # Non-sorted order tests first appearance before the seeded shuffle.
        bases = ["base-{:03d}".format(i) for i in reversed(range(base_count))]
        novels = ["novel-{:03d}".format(i) for i in reversed(range(novel_count))]
        train = [sample(label, 1000 * i + j) for i, label in enumerate(bases)
                 for j in range(base_n)]
        test = [sample(label, 100000 + 1000 * i + j) for i, label in enumerate(novels)
                for j in range(novel_n)]
        write_rows(self.source / "train.txt", train)
        write_rows(self.source / "test.txt", test)
        return bases, novels, train, test

    def prepare(self, **kwargs):
        kwargs.setdefault("expected_total", 4)
        return prepare_dataset(self.source, self.root / "prepared", **kwargs)

    def test_single_position_contiguous_list_and_direction(self):
        row = sample("relation", 0)
        original = copy.deepcopy(row)
        text = format_relation_text(row)
        self.assertIn("[HEAD] Person0 [/HEAD]", text)
        self.assertIn("[TAIL] New York [/TAIL]", text)
        self.assertTrue(text.startswith("Head: Person0. Tail: New York."))
        self.assertEqual(row, original)
        row["h"], row["t"] = row["t"], row["h"]
        self.assertNotEqual(format_relation_text(row), text)

    def test_explicit_half_open_and_ambiguous_two_positions(self):
        row = sample("r", 0)
        row["h"]["pos"] = [0, 1]
        row["t"]["pos"] = [2, 4]
        self.assertIn("[HEAD] Person0 [/HEAD]", format_relation_text(row, "half_open"))
        row["t"]["pos"] = [2, 3]
        self.assertIn("[TAIL] New [/TAIL] York", format_relation_text(row, "half_open"))
        self.assertIn("[TAIL] New York [/TAIL]", format_relation_text(row, "indices"))

    def test_invalid_positions_and_token_conflict(self):
        for pos in ([], [0, 2], [2, 1], [0, 0], [-1], [4], [True], [1.0]):
            row = sample("r", 0)
            row["h"]["pos"] = pos
            with self.subTest(pos=pos), self.assertRaises(ValueError):
                format_relation_text(row)
        row = sample("r", 0)
        row["token"] = ["different"]
        with self.assertRaisesRegex(ValueError, "disagree"):
            format_relation_text(row)
        with self.assertRaises(ValueError):
            format_relation_text(sample("r", 0), "auto")
        row = sample("r", 0)
        row["token"] = row.pop("tokens")
        self.assertIn("Person0", format_relation_text(row))

    def test_blank_tokens_preserve_entity_indices_and_input(self):
        row = {"tokens": ["", "Alice", " ", "visited", "New", "\t", "York", "\n"],
               "h": {"pos": [1]}, "t": {"pos": [4, 5, 6]}}
        original = copy.deepcopy(row)
        self.assertEqual(format_relation_text(row),
                         "Head: Alice. Tail: New York. Sentence: "
                         "[HEAD] Alice [/HEAD] visited [TAIL] New York [/TAIL]")
        self.assertEqual(row, original)
        row["h"]["pos"] = [0, 3]
        row["t"]["pos"] = [4, 8]
        self.assertEqual(format_relation_text(row, "half_open"), format_relation_text(original))

    def test_blank_sentence_and_blank_entity_rejected(self):
        with self.assertRaisesRegex(ValueError, "sentence contains only blank"):
            format_relation_text({"tokens": ["", "\t"], "h": {"pos": [0]}, "t": {"pos": [1]}})
        with self.assertRaisesRegex(ValueError, "h.pos .* contains only blank"):
            format_relation_text({"tokens": ["", "Paris"],
                                  "h": {"pos": [0], "name": "Alice"}, "t": {"pos": [1]}})

    def test_non_string_tokens_report_index_and_source_line(self):
        self.fixture()
        for bad in (None, 42, ["bad"]):
            row = sample("base", 1)
            row["tokens"][1] = bad
            write_rows(self.source / "train.txt", [sample("base", i) for i in range(72)] + [row])
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, r"train.txt:73: tokens\[1\]"):
                self.prepare()

    def test_blank_token_audit_retains_records_and_split_counts(self):
        _, _, train, test = self.fixture()
        for row in (train[0], test[0]):
            row["tokens"].insert(0, "\t")
            row["tokens"].append("")
            row["h"]["pos"] = [1]
            row["t"]["pos"] = [3, 4]
        write_rows(self.source / "train.txt", train)
        write_rows(self.source / "test.txt", test)
        before = (self.source / "train.txt").read_bytes()
        manifest = self.prepare()
        self.assertEqual(manifest["audit"]["blank_token_count"], 4)
        self.assertEqual(manifest["audit"]["records_with_blank_tokens"], 2)
        self.assertEqual(manifest["sources"]["train"]["blank_token_examples"],
                         [{"line": 1, "positions": [0, 5]}])
        self.assertEqual({key: value["count"] for key, value in manifest["splits"].items()},
                         {"train_labeled": 2, "validation": 2, "train_unlabeled": 16, "test": 16})
        self.assertEqual((self.source / "train.txt").read_bytes(), before)

    def test_exact_sample_split_counts_and_source_unchanged(self):
        source_train, source_test, _, _ = self.fixture()
        before = {name: (self.source / name).read_bytes() for name in ("train.txt", "test.txt")}
        manifest = self.prepare(seed=7)
        rng = np.random.RandomState(7)
        classes = source_train + source_test
        rng.shuffle(classes)
        self.assertEqual(manifest["base_classes"], classes[:2])
        self.assertEqual(manifest["novel_classes"], classes[2:])
        self.assertEqual({key: value["count"] for key, value in manifest["splits"].items()},
                         {"train_labeled": 2, "validation": 2, "train_unlabeled": 16, "test": 16})
        expected_labeled = []
        for relation in classes[:2]:
            source = "train" if relation in source_train else "test"
            source_classes = source_train if source == "train" else source_test
            start = source_classes.index(relation) * 5 + 1
            order = np.arange(5)
            rng.shuffle(order)
            expected_labeled.append("{}:{:08d}".format(source, start + int(order[0])))
        self.assertEqual(manifest["splits"]["train_labeled"]["ids"], expected_labeled)
        for name, original in before.items():
            self.assertEqual((self.source / name).read_bytes(), original)
            self.assertEqual(manifest["sources"][Path(name).stem]["sha256"],
                             hashlib.sha256(original).hexdigest())
        labeled_ids = set(manifest["splits"]["train_labeled"]["ids"])
        val_ids = set(manifest["splits"]["validation"]["ids"])
        unlab_ids = set(manifest["splits"]["train_unlabeled"]["ids"])
        self.assertFalse(labeled_ids & val_ids)
        self.assertFalse(val_ids & unlab_ids)
        self.assertFalse(labeled_ids & unlab_ids)

    def test_default_40_40_and_unlabeled_has_no_labels(self):
        self.fixture(base_count=64, novel_count=16, base_n=4, novel_n=4)
        output = self.root / "prepared"
        manifest = prepare_dataset(self.source, output)
        self.assertEqual((manifest["n_base"], manifest["n_novel"], manifest["n_total"]), (40, 40, 80))
        labeled = read_rows(output / "train_labeled.jsonl")
        test = read_rows(output / "test.jsonl")
        unlab = read_rows(output / "train_unlabeled.jsonl")
        self.assertEqual({row["label"] for row in labeled}, set(range(40)))
        self.assertEqual({row["label"] for row in test}, set(range(80)))
        self.assertEqual(len(unlab), 240)
        self.assertTrue(all(set(row) == {"id", "text"} for row in unlab))
        self.assertEqual(unlab, [{"id": row["id"], "text": row["text"]} for row in test])
        for filename, digest in manifest["file_sha256"].items():
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), digest)

    def test_determinism_and_no_overwrite(self):
        self.fixture()
        first = self.prepare(seed=11)
        second = prepare_dataset(self.source, self.root / "second", seed=11,
                                 expected_total=4)
        self.assertEqual(first, second)
        for name in list(first["files"].values()) + ["manifest.json"]:
            self.assertEqual((self.root / "prepared" / name).read_bytes(),
                             (self.root / "second" / name).read_bytes())
        with self.assertRaises(FileExistsError):
            self.prepare(seed=11)

    def test_small_base_class_rejected_without_output(self):
        self.fixture(base_n=3, novel_n=3)
        with self.assertRaisesRegex(ValueError, "each must be nonempty"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())

    def test_expected_unique_class_count_and_cross_file_class_merging(self):
        bases, novels, _, test = self.fixture()
        with self.assertRaisesRegex(ValueError, "expected 80"):
            prepare_dataset(self.source, self.root / "prepared")
        write_rows(self.source / "test.txt", test + [sample(bases[0], 999)])
        manifest = self.prepare()
        self.assertEqual(set(manifest["classes"]), set(bases + novels))
        self.assertEqual(len(manifest["classes"]), 4)
        counts = {row["class"]: row["source"] for row in manifest["class_counts"]}
        self.assertEqual(counts[bases[0]], 6)
        self.assertEqual(sum(manifest["splits"][key]["count"]
                             for key in ("train_labeled", "validation", "test")), 21)

    def test_duplicates_conflicts_retained_and_other_filtered(self):
        _, _, train, _ = self.fixture(base_n=4)
        train[1] = copy.deepcopy(train[0])
        conflicting_label = train[4]["relation"]
        train[4] = copy.deepcopy(train[0])
        train[4]["relation"] = conflicting_label
        train += [{"relation": name} for name in ("Other", "None", "none", "NA")]
        write_rows(self.source / "train.txt", train)
        manifest = self.prepare()
        self.assertEqual(manifest["sources"]["train"]["retained_records"], 8)
        self.assertEqual(manifest["sources"]["train"]["filtered_records"], 4)
        self.assertEqual(manifest["audit"]["same_input_same_label_extra_records"], 1)
        self.assertEqual(manifest["audit"]["conflicting_input_count"], 1)
        self.assertEqual(manifest["audit"]["conflicting_record_count"], 3)
        self.assertEqual(sum(manifest["splits"][key]["count"]
                             for key in ("train_labeled", "validation", "test")), 18)

    def test_invalid_row_gives_source_line_and_does_not_execute(self):
        self.fixture()
        (self.source / "train.txt").write_text("__import__('os').getcwd()\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "train.txt:1"):
            self.prepare()
        self.assertFalse((self.root / "prepared").exists())


if __name__ == "__main__":
    unittest.main()
