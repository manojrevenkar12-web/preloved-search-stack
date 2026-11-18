import numpy as np

from preloved.measure.interleave import preference, team_draft


def test_team_draft_mixes_without_duplicates():
    a, b = list(range(0, 40)), list(range(20, 60))
    page, team = team_draft(a, b, 20, seed=1)
    assert len(page) == 20 == len(set(page))
    assert set(team) == set(page)
    counts = [sum(v == t for v in team.values()) for t in "ab"]
    assert abs(counts[0] - counts[1]) <= 1


def test_identical_rankings_give_no_preference():
    a = list(range(30))
    wins = []
    for s in range(400):
        page, team = team_draft(a, a, 20, seed=s)
        wins.append(team[page[0]])
    assert 0.4 < np.mean([w == "a" for w in wins]) < 0.6


def test_preference_interval():
    p = preference(np.array([0, 1, 0, 0, 2]), np.array([1, 0, 1, 1, 2]))
    assert p["decided"] == 4 and p["b_share"] == 0.75 and p["low"] < 0.75 < p["high"]
