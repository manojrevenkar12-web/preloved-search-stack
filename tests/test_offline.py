import numpy as np

from preloved.measure.offline import OfflineSet, metrics, ndcg


def test_ndcg():
    assert ndcg([1, 2, 3], {1: 1.0}) == 1.0
    assert 0 < ndcg([2, 1, 3], {1: 1.0}) < 1
    assert ndcg([4, 5], {1: 1.0}) == 0.0


def test_point_in_time_availability():
    off = OfflineSet(
        [], {}, {}, [], np.array([0, 2, 0]), np.array([10**6, 10**6, 1]), np.array([10**6, 10**6, 10**6]), {}
    )
    assert off.available(1).tolist() == [True, False, True]  # item 2 sold on day 1: for sale before it sold
    assert off.available(2).tolist() == [True, True, False]


def test_metrics_weight_clicks_by_position():
    off = OfflineSet([], {7: [(1, 0), (2, 5)]}, {7: 2}, [], np.zeros(3), np.zeros(3), np.zeros(3), {})
    exam = 1.0 / (1 + np.arange(20)) ** 0.9
    top_shown_first = metrics({7: [1, 2]}, off, exam)
    deep_click_first = metrics({7: [2, 1]}, off, exam)
    assert top_shown_first["ndcg_clicks"] == deep_click_first["ndcg_clicks"]
    assert deep_click_first["ndcg_ips"] > top_shown_first["ndcg_ips"]
    assert deep_click_first["recall_pit"] == 1.0
