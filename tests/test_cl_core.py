from types import SimpleNamespace as R

import numpy as np
import pytest

from agricl.cl.core import average_incremental_accuracy, forgetting, herding_select, make_tasks


def fake_records():
    recs = []
    for p, n in [("apple", 6), ("wheat", 5), ("corn", 4), ("tomato", 4), ("grape", 3), ("pea", 2), ("bean", 2)]:
        for k in range(n):
            recs.append(R(plant=p, disease=f"{p} d{k}"))
            recs.append(R(plant=p, disease=f"{p} d{k}"))  # duplicates are fine
    return recs


def test_make_tasks_partition_and_plant_grouping():
    diseases, plants = make_tasks(fake_records(), n_tasks=3, base_frac=0.5, seed=0)
    flat = [d for t in diseases for d in t]
    assert len(flat) == len(set(flat)) == 26  # every disease exactly once
    assert sum(len(t) for t in diseases[:1]) >= 13  # base holds at least half
    for ds, ps in zip(diseases, plants):  # a plant's diseases stay in one round
        assert {d.split()[0] for d in ds} == set(ps)


def test_make_tasks_deterministic():
    assert make_tasks(fake_records(), 3, seed=1) == make_tasks(fake_records(), 3, seed=1)


def test_make_tasks_too_many_rounds():
    with pytest.raises(ValueError):
        make_tasks(fake_records(), n_tasks=20)


def test_herding_first_pick_is_closest_to_mean_and_unique():
    f = np.array([[1, 0], [0, 1], [1, 1]], float)
    sel = herding_select(f, 3)
    assert sel[0] == 2  # (1,1)/sqrt2 is clearly closest to the normalised mean (0.57, 0.57)
    assert len(sel) == len(set(sel)) == 3


def test_herding_m_larger_than_class():
    assert sorted(herding_select(np.eye(3), 10)) == [0, 1, 2]


def test_cl_metrics():
    acc = np.array([[0.9, np.nan, np.nan],
                    [0.7, 0.8, np.nan],
                    [0.6, 0.5, 0.9]])
    assert forgetting(acc) == pytest.approx(((0.9 - 0.6) + (0.8 - 0.5)) / 2)
    assert average_incremental_accuracy([0.9, 0.75, 0.66]) == pytest.approx(0.77)
    assert forgetting(np.array([[0.5]])) == 0.0
