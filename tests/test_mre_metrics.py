import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment

from mre_metrics import mre_accuracy


def reference_mre(y_true, y_pred, n_base, n_total):
    """Literal algorithm of MRE_learn framework.py's two accuracy helpers."""
    w = np.zeros((n_total, n_total), dtype=int)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    ind_map = {j: i for i, j in ind}
    overall = sum(w[i, j] for i, j in ind) / len(y_pred)

    def category_acc(categories):
        correct, instances = 0, 0
        for i in categories:
            correct += w[ind_map[i], i]
            instances += sum(w[:, i])
        return correct / instances

    return {"Base": category_acc(range(n_base)),
            "Novel": category_acc(range(n_base, n_total)), "Overall": overall}


class MREMetricsTests(unittest.TestCase):
    def test_matches_original_mre_for_random_predictions(self):
        for seed in range(10):
            rng = np.random.RandomState(seed)
            truth = np.concatenate((np.arange(80), rng.randint(80, size=120)))
            pred = rng.randint(80, size=truth.size)
            self.assertEqual(mre_accuracy(truth, pred), reference_mre(truth, pred, 40, 80))

    def test_explicit_legacy_64_16_split_remains_supported(self):
        truth = [0, 39, 40, 63, 64, 79]
        pred = [0, 39, 40, 40, 64, 79]
        self.assertEqual(mre_accuracy(truth, pred, n_base=64, n_total=80),
                         reference_mre(truth, pred, 64, 80))

    def test_global_mapping_not_independent_subset_matching(self):
        scores = mre_accuracy([0, 0, 0, 1, 1], [0, 0, 0, 0, 0], n_base=1, n_total=2)
        self.assertEqual(scores, {"Base": 1.0, "Novel": 0.0, "Overall": 0.6})
        # Independently matching each subset would incorrectly give both 1.0.

    def test_base_only_validation_and_absent_ids(self):
        self.assertEqual(mre_accuracy([0, 1, 1], [79, 67, 67]),
                         {"Base": 1.0, "Novel": None, "Overall": 1.0})
        self.assertEqual(mre_accuracy([70, 79], [3, 4]),
                         {"Base": None, "Novel": 1.0, "Overall": 1.0})

    def test_perfect_permutation_and_no_rounding(self):
        self.assertEqual(mre_accuracy([0, 40], [79, 5]),
                         {"Base": 1.0, "Novel": 1.0, "Overall": 1.0})
        result = mre_accuracy([0, 0, 40], [0, 0, 0])
        self.assertEqual(result["Overall"], 2 / 3)

    def test_invalid_arrays(self):
        for truth, pred in [([], []), ([0], [0, 1]), ([[0]], [[0]]),
                            ([0.0], [0]), ([True], [0]), (["0"], [0]),
                            ([0], [-1]), ([80], [0]), ([0], [80])]:
            with self.subTest(truth=truth, pred=pred):
                with self.assertRaises(ValueError):
                    mre_accuracy(truth, pred)

    def test_invalid_class_counts(self):
        for base, total in [(0, 80), (80, 80), (81, 80), (1.0, 80), (True, 80), (1, 2.0)]:
            with self.assertRaises(ValueError):
                mre_accuracy([0], [0], n_base=base, n_total=total)


if __name__ == "__main__":
    unittest.main()
