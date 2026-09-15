"""Offline regression for smoke tests that used to skip LLM supervision."""
import numpy as np
import pytest

import run_mre
from mre_neighbors import cap_query_candidates, mine_neighbors, predict_clusters


def test_smoke_keeps_normal_ranking_pool_and_bounds_queries(monkeypatch):
    monkeypatch.setattr(run_mre.defaults, 'QUERY_POOL_SIZE', 500)
    cfg = run_mre.experiment_config(run_mre.build_parser().parse_args(['--smoke']))
    assert (cfg['pretrain_epochs'], cfg['train_epochs']) == (2, 2)
    assert cfg['query_pool_size'] == 500
    assert cfg['max_queries_per_refresh'] == 5

    features = np.random.RandomState(7).normal(size=(64, 8))
    centers = features[::8]
    pseudo = predict_clusters(features, centers)
    indices, old_selected = mine_neighbors(features, pseudo, centers, 10, 5)
    assert old_selected == []
    indices, normal_selected = mine_neighbors(features, pseudo, centers, 10, cfg['query_pool_size'])
    selected = cap_query_candidates(indices, pseudo, normal_selected, cfg['max_queries_per_refresh'])
    assert len(selected) == 5 and set(selected).issubset(normal_selected)
    for i in selected:
        assert len(set(pseudo[indices[i][indices[i] != i]])) >= 2


def test_cap_skips_unqueryable_anchor_and_does_not_count_self_as_candidate():
    indices = np.array([[0, 1, 2], [1, 2, 3], [2, 1, 3], [3, 1, 2]])
    pseudo = np.array([0, 1, 1, 2])
    # Anchor 0 only has class-1 neighbors; its own class must not make it eligible.
    assert cap_query_candidates(indices, pseudo, [0, 2, 1, 3], 1) == [2]
    assert cap_query_candidates(indices, pseudo, [], 5) == []
    assert cap_query_candidates(indices, np.zeros(4), [0, 1, 2], 5) == []
    assert cap_query_candidates(indices, pseudo, [0, 1, 2], 0) == []


def test_formal_run_has_no_query_cap_and_preserves_selection():
    cfg = run_mre.experiment_config(run_mre.build_parser().parse_args([]))
    assert cfg['max_queries_per_refresh'] is None
    assert cap_query_candidates([[0, 1], [1, 0]], [0, 0], [1, 0], None) == [1, 0]


@pytest.mark.parametrize('limit', [-1, True, 1.5])
def test_invalid_cap_rejected(limit):
    with pytest.raises(ValueError):
        cap_query_candidates([[0, 1], [1, 0]], [0, 1], [0], limit)
