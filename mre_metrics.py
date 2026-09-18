"""MRE's global Hungarian matched accuracies (fractions, not percentages)."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def _integer_vector(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("{} must be a nonempty one-dimensional array".format(name))
    if values.dtype.kind not in "iu":
        raise ValueError("{} must contain integer IDs (not floats/bools)".format(name))
    return values


def mre_accuracy(y_true, y_pred, n_base=64, n_total=80):
    """Use ONE global matching, then score base/novel ground-truth subsets.

    IDs must be in [0, n_total), with base true IDs in [0, n_base).
    An absent subgroup returns None; Overall is always a float in [0, 1].
    No subgroup is matched separately and no result is rounded.
    """
    if isinstance(n_base, bool) or not isinstance(n_base, (int, np.integer)):
        raise ValueError("n_base must be an integer")
    if isinstance(n_total, bool) or not isinstance(n_total, (int, np.integer)):
        raise ValueError("n_total must be an integer")
    if not 0 < n_base < n_total:
        raise ValueError("require 0 < n_base < n_total")
    y_true = _integer_vector(y_true, "y_true")
    y_pred = _integer_vector(y_pred, "y_pred")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same length")
    for values, name in ((y_true, "y_true"), (y_pred, "y_pred")):
        if np.any(values < 0) or np.any(values >= n_total):
            raise ValueError("{} IDs must be in [0, {})".format(name, n_total))

    # Orientation is deliberately identical to framework.py in MRE_learn.
    counts = np.zeros((n_total, n_total), dtype=np.int64)
    np.add.at(counts, (y_pred, y_true), 1)
    pred_ids, true_ids = linear_sum_assignment(counts.max() - counts)
    matched_by_true = np.zeros(n_total, dtype=np.int64)
    matched_by_true[true_ids] = counts[pred_ids, true_ids]
    true_totals = counts.sum(axis=0)

    def subset_score(start, stop):
        denominator = int(true_totals[start:stop].sum())
        if denominator == 0:
            return None
        return float(matched_by_true[start:stop].sum() / denominator)

    return {
        "Base": subset_score(0, n_base),
        "Novel": subset_score(n_base, n_total),
        "Overall": float(matched_by_true.sum() / y_true.size),
    }
