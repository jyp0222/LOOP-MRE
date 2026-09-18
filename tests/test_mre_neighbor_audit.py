"""Offline audit checks use hand-counted relation decisions, not training code."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from mre_neighbor_audit import audit_run


class NeighborAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.temp.name)
        self.data_dir = self.run_dir / "data"
        self.data_dir.mkdir()
        self.rows = {
            "train_labeled": [
                {"id": "a", "text": "Alice founded Acme.", "label": 0},
                {"id": "a2", "text": "Ben founded Beta.", "label": 0},
                {"id": "b", "text": "Cara directed Cinema.", "label": 1},
                {"id": "b2", "text": "Dan directed Drama.", "label": 1},
            ],
            "validation": [
                {"id": "va", "text": "Eva founded Echo.", "label": 0},
                {"id": "vb", "text": "Fred directed Film.", "label": 1},
            ],
            "test": [
                # Duplicate text differs in case, so exact-string comparison fails.
                {"id": "a_duplicate", "text": "ALICE FOUNDED ACME.", "label": 0},
                {"id": "n", "text": "Gina married Hugo.", "label": 2},
                {"id": "n2", "text": "Ira married Jade.", "label": 2},
                {"id": "m", "text": "Kai was born in Lima.", "label": 3},
                {"id": "m2", "text": "Mona was born in Nice.", "label": 3},
            ],
        }
        self.rows["train_unlabeled"] = [
            {"id": row["id"], "text": row["text"]} for row in self.rows["test"]
        ]
        classes = ["founder", "director", "spouse", "birthplace"]
        self.manifest = {
            "schema_version": 1,
            "protocol": "mre_transductive_fixed_base_novel",
            "n_base": 2,
            "n_novel": 2,
            "n_total": 4,
            "base_classes": classes[:2],
            "novel_classes": classes[2:],
            "classes": classes,
            "class_to_id": {name: i for i, name in enumerate(classes)},
            "files": {},
            "file_sha256": {},
            "splits": {},
        }
        for split in self.rows:
            self.write_split(split)
        self.write_manifest()
        # Nine decisions deliberately cover zero/one/two correct candidates,
        # repeated queries, every error direction, fallback, self and duplicate.
        self.decisions = [
            self.decision("a", ["b", "a2"], "a2"),
            self.decision("a", ["b", "n"], "n"),
            self.decision("n", ["n2", "a"], "a"),
            self.decision("n", ["a", "b"], "b", fallback=True),
            self.decision("b", ["a", "n"], "a"),
            self.decision("n", ["m", "m2"], "m"),
            self.decision("a", ["a", "b"], "a"),
            self.decision("a", ["a_duplicate", "b"], "a_duplicate"),
            self.decision("m", ["m2", "m"], "m2"),
        ]
        self.write_decisions()

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def decision(query_id, candidate_ids, selected_id, fallback=False):
        return {"query_id": query_id, "candidate_ids": candidate_ids,
                "selected_id": selected_id, "fallback": fallback}

    def write_split(self, split):
        filename = split + ".jsonl"
        content = "".join(json.dumps(row) + "\n" for row in self.rows[split]).encode("utf-8")
        (self.data_dir / filename).write_bytes(content)
        self.manifest["files"][split] = filename
        self.manifest["file_sha256"][filename] = hashlib.sha256(content).hexdigest()
        self.manifest["splits"][split] = {
            "count": len(self.rows[split]), "ids": [row["id"] for row in self.rows[split]]
        }

    def write_manifest(self):
        (self.data_dir / "manifest.json").write_text(
            json.dumps(self.manifest), encoding="utf-8")

    def write_decisions(self):
        (self.run_dir / "neighbor_queries.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.decisions), encoding="utf-8")

    def test_hand_counted_metrics_and_conditional_denominators(self):
        summary = audit_run(self.run_dir)
        metrics = summary["scopes"]["all_decisions"]["overall"]
        self.assertEqual(metrics["decision_count"], 9)
        self.assertEqual(metrics["unique_queries"], 4)
        self.assertAlmostEqual(metrics["candidate_coverage"], 5 / 9)
        self.assertAlmostEqual(metrics["neither_candidate_correct"], 4 / 9)
        self.assertEqual(metrics["exactly_one_candidate_count"], 4)
        self.assertAlmostEqual(metrics["accuracy_given_exactly_one"], 3 / 4)
        self.assertAlmostEqual(metrics["selected_same_label_rate"], 4 / 9)
        # Expectation averages the number of correct choices / 2 per decision.
        # The last decision has TWO same-label candidates and contributes 1.
        self.assertAlmostEqual(metrics["random_two_choice_expected_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["choice1_same_label_rate"], 4 / 9)
        self.assertAlmostEqual(metrics["selected_choice1_rate"], 5 / 9)
        self.assertAlmostEqual(metrics["selected_self_rate"], 1 / 9)
        self.assertAlmostEqual(metrics["candidate_self_rate"], 2 / 9)
        self.assertAlmostEqual(metrics["selected_same_text_rate"], 2 / 9)
        self.assertAlmostEqual(metrics["candidate_same_text_rate"], 3 / 9)
        self.assertAlmostEqual(metrics["fallback_rate"], 1 / 9)

    def test_base_novel_are_grouped_by_query_label(self):
        groups = audit_run(self.run_dir)["scopes"]["all_decisions"]
        base, novel = groups["base"], groups["novel"]
        self.assertEqual(base["decision_count"], 5)
        self.assertEqual(novel["decision_count"], 4)
        self.assertEqual(base["unique_queries"], 2)
        self.assertEqual(novel["unique_queries"], 2)
        self.assertAlmostEqual(base["selected_same_label_rate"], 3 / 5)
        self.assertAlmostEqual(novel["selected_same_label_rate"], 1 / 4)
        self.assertAlmostEqual(base["random_two_choice_expected_rate"], 3 / 10)
        self.assertAlmostEqual(novel["random_two_choice_expected_rate"], 3 / 8)
        self.assertEqual(base["exactly_one_candidate_count"], 3)
        self.assertEqual(novel["exactly_one_candidate_count"], 1)
        self.assertEqual(base["accuracy_given_exactly_one"], 1.0)
        self.assertEqual(novel["accuracy_given_exactly_one"], 0.0)

    def test_scopes_exclude_fallback_then_any_self_or_duplicate_candidate(self):
        scopes = audit_run(self.run_dir)["scopes"]
        llm = scopes["llm_answers_only"]["overall"]
        strict = scopes["llm_answers_without_self_or_duplicate"]["overall"]
        self.assertEqual(llm["decision_count"], 8)
        self.assertAlmostEqual(llm["selected_same_label_rate"], 1 / 2)
        self.assertAlmostEqual(llm["candidate_coverage"], 5 / 8)
        self.assertEqual(llm["fallback_rate"], 0.0)
        # Exclude decision 9 although the selected candidate is not the query:
        # its OTHER candidate is the query, which makes the comparison trivial.
        self.assertEqual(strict["decision_count"], 5)
        self.assertEqual(strict["unique_queries"], 3)
        self.assertAlmostEqual(strict["candidate_coverage"], 2 / 5)
        self.assertAlmostEqual(strict["neither_candidate_correct"], 3 / 5)
        self.assertEqual(strict["exactly_one_candidate_count"], 2)
        self.assertAlmostEqual(strict["accuracy_given_exactly_one"], 1 / 2)
        self.assertAlmostEqual(strict["selected_same_label_rate"], 1 / 5)
        self.assertAlmostEqual(strict["random_two_choice_expected_rate"], 1 / 5)
        for key in ("selected_self_rate", "candidate_self_rate", "selected_same_text_rate",
                    "candidate_same_text_rate", "fallback_rate"):
            self.assertEqual(strict[key], 0.0)

    def test_error_directions_and_per_relation_records(self):
        summary = audit_run(self.run_dir)
        self.assertEqual(summary["errors_by_group"], {
            "base_to_other_base": 1, "base_to_novel": 1,
            "novel_to_base": 2, "novel_to_other_novel": 1,
        })
        by_label = {row["label"]: row for row in summary["per_relation"]}
        self.assertEqual(set(by_label), {0, 1, 2, 3})
        self.assertEqual(by_label[0]["relation"], "founder")
        self.assertEqual(by_label[0]["group"], "base")
        self.assertEqual(by_label[2]["group"], "novel")
        self.assertEqual(by_label[0]["metrics"]["decision_count"], 4)
        self.assertAlmostEqual(by_label[0]["metrics"]["selected_same_label_rate"], 3 / 4)
        self.assertEqual(by_label[2]["metrics"]["decision_count"], 3)
        self.assertEqual(by_label[2]["metrics"]["selected_same_label_rate"], 0.0)

    def test_empty_log_is_no_evidence_not_zero_accuracy(self):
        self.decisions = []
        self.write_decisions()
        summary = audit_run(self.run_dir)
        count_fields = {"decision_count", "unique_queries", "exactly_one_candidate_count"}
        for scope in summary["scopes"].values():
            for group in ("overall", "base", "novel"):
                for key, value in scope[group].items():
                    with self.subTest(group=group, key=key):
                        self.assertEqual(value, 0 if key in count_fields else None)

    def test_conditional_accuracy_is_none_without_exactly_one_correct_candidate(self):
        self.decisions = [self.decision("n", ["a", "b"], "a")]
        self.write_decisions()
        metrics = audit_run(self.run_dir)["scopes"]["all_decisions"]["overall"]
        self.assertEqual(metrics["decision_count"], 1)
        self.assertEqual(metrics["candidate_coverage"], 0.0)
        self.assertEqual(metrics["exactly_one_candidate_count"], 0)
        self.assertIsNone(metrics["accuracy_given_exactly_one"])

    def test_unknown_query_or_candidate_ids_are_rejected(self):
        cases = [self.decision("missing", ["a", "b"], "a"),
                 self.decision("a", ["missing", "b"], "b"),
                 self.decision("a", ["b", "missing"], "b")]
        for row in cases:
            with self.subTest(row=row):
                self.decisions = [row]
                self.write_decisions()
                with self.assertRaises(ValueError):
                    audit_run(self.run_dir)

    def test_choice_not_in_candidates_is_rejected(self):
        self.decisions = [self.decision("a", ["b", "n"], "a2")]
        self.write_decisions()
        with self.assertRaises(ValueError):
            audit_run(self.run_dir)

    def test_duplicate_candidate_ids_are_rejected(self):
        self.decisions = [self.decision("a", ["b", "b"], "b")]
        self.write_decisions()
        with self.assertRaises(ValueError):
            audit_run(self.run_dir)

    def test_data_digest_mismatch_is_rejected(self):
        path = self.data_dir / self.manifest["files"]["test"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            audit_run(self.run_dir)

    def test_manifest_count_and_ids_must_match_actual_split(self):
        for field, bad_value in (("count", 999), ("ids", ["not-the-saved-ids"])):
            original = self.manifest["splits"]["train_labeled"][field]
            with self.subTest(field=field):
                self.manifest["splits"]["train_labeled"][field] = bad_value
                self.write_manifest()
                with self.assertRaises(ValueError):
                    audit_run(self.run_dir)
            self.manifest["splits"]["train_labeled"][field] = original

    def test_missing_log_is_not_treated_as_successful_empty_log(self):
        (self.run_dir / "neighbor_queries.jsonl").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            audit_run(self.run_dir)

    def test_audit_does_not_modify_experiment_files(self):
        paths = [path for path in self.run_dir.rglob("*") if path.is_file()]
        before = {path: path.read_bytes() for path in paths}
        audit_run(self.run_dir)
        self.assertEqual(before, {path: path.read_bytes() for path in paths})


if __name__ == "__main__":
    unittest.main()
