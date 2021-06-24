from collections import Counter
import operator

import pytest

import numpy as np
import numba as nb

import cluscheck


@pytest.mark.parametrize("random_seed", tuple(range(5)))
@pytest.mark.parametrize("max_count,iterations,expected_checks,expected_checks_cmp", (
    (1, 1, 1, operator.eq,),
    (1, 2, 2, operator.eq,),
    (2, 1, 1, operator.ge,),
    (5, 2, 2, operator.ge,),
))
def test_all_checked(
    random_seed,
    max_count,
    iterations,
    expected_checks,
    expected_checks_cmp,
):
    rs = np.random.RandomState(random_seed)
    dp = rs.uniform(-1,1,(64,1000))
    ndp = np.arange(1000, dtype="int32").reshape((1,1000))

    checked_items = Counter()

    @nb.jit(nopython=True)
    def check(ndp_):
        # actual assertions have non-constant messages, which nopython mode
        # doesn't like
        if ndp_.shape[0] != ndp.shape[0]:
            raise AssertionError("ndp dimension 0 incorrect")
        if ndp_.shape[1] > max_count:
            raise AssertionError("ndp dimension 1 greater than max_count")

        with nb.objmode():
            checked_items.update(ndp_.flat)

        return False

    finder = cluscheck.get_finder_for_cluster_obeying(
        check,
        min_count=1,
        max_count=max_count,
        # ridiculous depth should make it unlikely we miss any
        max_depth=100,
    )

    finder(dp, ndp, random_seed=random_seed, iterations=iterations)

    ref_counter = Counter({i: expected_checks for i in range(1000)})
    assert checked_items.keys() == ref_counter.keys()
    for k in ref_counter.keys():
        assert expected_checks_cmp(checked_items[k], ref_counter[k])


@pytest.mark.parametrize("random_seed", tuple(range(5)))
def test_abort_branch(random_seed):
    rs = np.random.RandomState(random_seed)
    dp = rs.uniform(-1,1,(64,1000))
    ndp = np.arange(1000, dtype="int32").reshape((1,1000))

    max_count = 8
    checked_items = Counter()

    @nb.jit(nopython=True)
    def check(ndp_):
        # actual assertions have non-constant messages, which nopython mode
        # doesn't like
        if ndp_.shape[0] != ndp.shape[0]:
            raise AssertionError("ndp dimension 0 incorrect")
        if ndp_.shape[1] > max_count:
            raise AssertionError("ndp dimension 1 greater than max_count")

        with nb.objmode():
            checked_items.update(ndp_.flat)

        for i in ndp_.flat:
            if i % 2:
                return -1
        return 0

    finder = cluscheck.get_finder_for_cluster_obeying(
        check,
        min_count=1,
        max_count=max_count,
        # ridiculous depth should make it unlikely we miss any
        max_depth=100,
    )

    finder(dp, ndp, random_seed=random_seed, iterations=1)

    assert checked_items.keys() == set(range(1000))
    for k, v in checked_items.items():
        if k % 2:
            # any clusters with odd numbers should have been aborted after
            # their first check call
            assert v == 1
