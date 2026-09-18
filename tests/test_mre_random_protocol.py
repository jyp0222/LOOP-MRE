"""Independent parity checks against the actual MRE splitting functions.

The two strings below are frozen, unmodified function source extracted with
ast.get_source_segment from MRE_learn/utils.py and data_loader.py.  Keeping the
small originals here allows the tests to run without TensorFlow, images, or a
second checkout.  Only these two FunctionDef nodes are compiled, not the MRE
modules and their heavyweight imports.  The oracle never calls our splitter.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mre_protocol import prepare_dataset


MRE_SPLIT_TYPES_SOURCE = '''def split_types(types):
    filtered_types = ["Other", "None", "none", "NA"]
    types = [ele for ele in types if ele not in filtered_types]
    np.random.shuffle(types)
    n_types = len(types)
    avg_n_types = n_types // 2
    base_types = types[:avg_n_types]
    novel_types = types[avg_n_types:]
    return base_types, novel_types'''

MRE_SPLIT_DATASET_SOURCE = '''def split_dataset(dataset, train_types, test_types):
    train_dataset = {k:[] for k in train_types}
    val_dataset = {k:[] for k in train_types}
    test_dataset = {k:[] for k in train_types + test_types}
    rel2sampleIdxs = dataset.rel2sampleIdxs

    for train_type in train_types:
        train_type_sampleIdxs = rel2sampleIdxs[train_type]
        n_seen_train_val_sample = int(len(train_type_sampleIdxs) * 0.5)
        n_seen_test_sample = len(train_type_sampleIdxs) - n_seen_train_val_sample
        n_seen_train_sample = int(n_seen_train_val_sample * 0.8)
        n_seen_val_sample = n_seen_train_val_sample - n_seen_train_sample

        np.random.shuffle(train_type_sampleIdxs)

        seen_train_val_sampleIdxs =  train_type_sampleIdxs[:n_seen_train_val_sample]
        seen_test_sampleIdxs = train_type_sampleIdxs[-n_seen_test_sample:]
        assert len(seen_train_val_sampleIdxs) + len(seen_test_sampleIdxs) == len(train_type_sampleIdxs)

        seen_train_sampleIdxs = seen_train_val_sampleIdxs[:n_seen_train_sample]
        seen_val_sampleIdxs = seen_train_val_sampleIdxs[-n_seen_val_sample:]
        assert len(seen_train_sampleIdxs) + len(seen_val_sampleIdxs) == len(seen_train_val_sampleIdxs)

        train_dataset[train_type] += dataset.idx2samples(seen_train_sampleIdxs)
        val_dataset[train_type] += dataset.idx2samples(seen_val_sampleIdxs)
        test_dataset[train_type] += dataset.idx2samples(seen_test_sampleIdxs)

    for test_type in test_types:
        test_dataset[test_type] += dataset.idx2samples(rel2sampleIdxs[test_type])

    return train_dataset, val_dataset, test_dataset'''


class SourceDataset:
    """The ordering behavior of MRelDataset, with IDs in place of image objects."""

    def __init__(self, train, test):
        self.samples = []
        self.rel2sampleIdxs = {}
        for origin, rows in (("train", train), ("test", test)):
            for line, row in enumerate(rows, 1):
                relation = row["relation"]
                self.rel2sampleIdxs.setdefault(relation, []).append(len(self.samples))
                self.samples.append("{}:{:08d}".format(origin, line))
        self.relations = list(self.rel2sampleIdxs)

    def idx2samples(self, indices):
        return [self.samples[index] for index in indices]


def original_functions(seed):
    namespace = {"np": SimpleNamespace(random=np.random.RandomState(seed))}
    for source in (MRE_SPLIT_TYPES_SOURCE, MRE_SPLIT_DATASET_SOURCE):
        parsed = ast.parse(source)
        node = next(node for node in parsed.body if isinstance(node, ast.FunctionDef))
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<original-MRE-function>", "exec"), namespace)
    return namespace


def original_split(train, test, seed, sort_relations=False, reset_after_classes=False):
    dataset = SourceDataset(train, test)
    functions = original_functions(seed)
    relations = sorted(dataset.relations) if sort_relations else dataset.relations
    base, novel = functions["split_types"](relations)
    if reset_after_classes:
        # Deliberately incorrect control used to establish that fixtures detect
        # accidentally re-seeding before the per-class sample shuffle.
        functions["np"].random = np.random.RandomState(seed)
    labeled, validation, test_set = functions["split_dataset"](dataset, base, novel)

    def flattened(mapping):
        return [identifier for values in mapping.values() for identifier in values]

    test_ids = flattened(test_set)
    return base, novel, {
        "train_labeled": flattened(labeled),
        "validation": flattened(validation),
        "test": test_ids,
        "train_unlabeled": list(test_ids),
    }


def record(relation, identity):
    # No image field: preparing splits must not require images.
    return {"relation": relation,
            "tokens": ["Person-{}".format(identity), "visited", "Place-{}".format(identity)],
            "h": {"pos": [0]}, "t": {"pos": [2]}}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class MRERandomProtocolParityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def write_sources(self, train, test):
        for name, rows in (("train", train), ("test", test)):
            (self.source / (name + ".txt")).write_text(
                "".join(repr(row) + "\n" for row in rows), encoding="utf-8")

    def full_fixture(self):
        # A nonalphabetical permutation, with 64 source-train and 16 source-test
        # classes and varied counts to exercise exact RNG consumption/rounding.
        relations = ["relation-{:02d}".format((index * 37) % 80) for index in range(80)]
        train, test = [], []
        for index, relation in enumerate(relations):
            target = train if index < 64 else test
            target.extend(record(relation, "{}-{}".format(index, j))
                          for j in range(5 + index % 8))
        self.write_sources(train, test)
        return train, test

    def assert_matches_oracle(self, manifest, output, train, test, seed):
        base, novel, expected_splits = original_split(train, test, seed)
        self.assertEqual(manifest["protocol"], "mre_transductive_random_half")
        self.assertEqual(manifest["base_classes"], base)
        self.assertEqual(manifest["novel_classes"], novel)
        self.assertEqual(manifest["classes"], base + novel)
        self.assertEqual(manifest["class_to_id"], {name: i for i, name in enumerate(base + novel)})
        id_to_relation = {}
        for origin, rows in (("train", train), ("test", test)):
            for line, row in enumerate(rows, 1):
                id_to_relation["{}:{:08d}".format(origin, line)] = row["relation"]
        for split, expected_ids in expected_splits.items():
            with self.subTest(seed=seed, split=split):
                self.assertEqual(manifest["splits"][split]["ids"], expected_ids)
                self.assertEqual(manifest["splits"][split]["count"], len(expected_ids))
                rows = read_jsonl(output / manifest["files"][split])
                self.assertEqual([row["id"] for row in rows], expected_ids)
                if split == "train_unlabeled":
                    self.assertTrue(all(set(row) == {"id", "text"} for row in rows))
                else:
                    self.assertEqual([row["label"] for row in rows],
                                     [manifest["class_to_id"][id_to_relation[i]] for i in expected_ids])

    def test_default_80_classes_match_real_mre_for_seeds_0_2_3(self):
        train, test = self.full_fixture()
        source_train = {row["relation"] for row in train}
        source_test = {row["relation"] for row in test}
        for seed in (0, 2, 3):
            with self.subTest(seed=seed):
                output = self.root / "seed-{}".format(seed)
                manifest = prepare_dataset(self.source, output, seed=seed)
                self.assertEqual((manifest["n_base"], manifest["n_novel"], manifest["n_total"]),
                                 (40, 40, 80))
                self.assert_matches_oracle(manifest, output, train, test, seed)
                self.assertTrue(source_train & set(manifest["novel_classes"]))
                self.assertTrue(source_test & set(manifest["base_classes"]))

    def test_same_relation_in_both_original_files_is_merged(self):
        relations = ["zebra", "alpha", "zeta", "delta", "beta", "omega"]
        train = [record(relation, "train-{}-{}".format(relation, j))
                 for relation in relations for j in range(4)]
        test = [record(relation, "test-{}-{}".format(relation, j))
                for relation in reversed(relations) for j in range(5)]
        self.write_sources(train, test)
        for seed in (0, 2, 3):
            output = self.root / "overlap-{}".format(seed)
            manifest = prepare_dataset(self.source, output, seed=seed, expected_total=6)
            self.assert_matches_oracle(manifest, output, train, test, seed)
            self.assertTrue(all(count["source"] == 9 for count in manifest["class_counts"]))

    def test_class_order_uses_first_appearance_not_sorted_names(self):
        train, test = self.full_fixture()
        expected = original_split(train, test, 2)
        sorted_control = original_split(train, test, 2, sort_relations=True)
        self.assertNotEqual(expected[0], sorted_control[0])
        output = self.root / "class-order"
        manifest = prepare_dataset(self.source, output, seed=2)
        self.assertEqual(manifest["base_classes"], expected[0])
        self.assertEqual(manifest["novel_classes"], expected[1])

    def test_class_and_sample_shuffles_share_one_unreset_rng(self):
        train, test = self.full_fixture()
        expected = original_split(train, test, 3)
        reset_control = original_split(train, test, 3, reset_after_classes=True)
        self.assertEqual(expected[:2], reset_control[:2])
        self.assertNotEqual(expected[2]["train_labeled"], reset_control[2]["train_labeled"])
        output = self.root / "shared-rng"
        manifest = prepare_dataset(self.source, output, seed=3)
        self.assertEqual(manifest["splits"]["train_labeled"]["ids"], expected[2]["train_labeled"])

    def test_filtered_relations_do_not_consume_class_shuffle_entries(self):
        train, test = self.full_fixture()
        train.insert(0, {"relation": "Other"})
        train.insert(12, {"relation": "None"})
        test.insert(0, {"relation": "none"})
        test.append({"relation": "NA"})
        self.write_sources(train, test)
        output = self.root / "filtered"
        manifest = prepare_dataset(self.source, output, seed=0)
        self.assert_matches_oracle(manifest, output, train, test, 0)
        self.assertEqual(manifest["n_total"], 80)
        self.assertEqual(sum(source["filtered_records"] for source in manifest["sources"].values()), 4)

    def test_seed_changes_category_membership_and_reproduces_same_seed(self):
        train, test = self.full_fixture()
        first = prepare_dataset(self.source, self.root / "first", seed=0)
        again = prepare_dataset(self.source, self.root / "again", seed=0)
        other = prepare_dataset(self.source, self.root / "other", seed=2)
        self.assertEqual(first, again)
        self.assertNotEqual(set(first["base_classes"]), set(other["base_classes"]))
        self.assert_matches_oracle(other, self.root / "other", train, test, 2)

    def test_preparation_does_not_consume_process_global_numpy_randomness(self):
        self.full_fixture()
        previous = np.random.get_state()
        try:
            np.random.seed(987)
            expected_next = np.random.RandomState(987).random_sample(5)
            prepare_dataset(self.source, self.root / "isolated", seed=3)
            np.testing.assert_array_equal(np.random.random_sample(5), expected_next)
        finally:
            np.random.set_state(previous)


if __name__ == "__main__":
    unittest.main()
