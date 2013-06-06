"""
Test functions related to the Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx
import scipy.linalg

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._util import (
        sparse_expm,
        get_dense_rate_matrix,
        )


def _check_sparse_expm(Q_sparse, t):

    # Compute the expm directly, using the scipy linalg expm function.
    states, Q_dense = get_dense_rate_matrix(Q_sparse)
    P_expected = scipy.linalg.expm(Q_dense * t)

    # Compute the expm indirectly,
    # using custom functions for specially structured rate matrices.
    P_sparse = sparse_expm(Q_sparse, t)
    P_observed = np.zeros_like(P_expected)
    for sa_index, sa in enumerate(states):
        for sb_index, sb in enumerate(states):
            if P_sparse.has_edge(sa, sb):
                prob = P_sparse[sa][sb]['weight']
                P_observed[sa_index, sb_index] = prob

    # Check that both results give the same answer.
    assert_allclose(P_expected, P_observed)


class TestSmallExpm(TestCase):

    def test_small_expm_restriction_a(self):
        np.random.seed(1234)
        nsamples = 10
        for t in np.logspace(-5, 5, 10, base=2):
            for i in range(nsamples):
                a, w, r = np.random.exponential(size=3)
                Q = nx.DiGraph()
                Q.add_weighted_edges_from([
                    (0, 1, a),
                    (1, 0, w),
                    (1, 2, r)])
                _check_sparse_expm(Q, t)

    def test_small_expm_restriction_b(self):
        np.random.seed(1234)
        nsamples = 10
        for t in np.logspace(-5, 5, 10, base=2):
            for i in range(nsamples):
                a, r = np.random.exponential(size=2)
                Q = nx.DiGraph()
                Q.add_weighted_edges_from([
                    (0, 1, a),
                    (1, 2, r)])
                _check_sparse_expm(Q, t)

    def test_small_expm_restriction_c(self):
        np.random.seed(1234)
        nsamples = 10
        for t in np.logspace(-5, 5, 10, base=2):
            for i in range(nsamples):
                a = np.random.exponential()
                Q = nx.DiGraph()
                Q.add_weighted_edges_from([
                    (0, 1, a),
                    (1, 2, a)])
                _check_sparse_expm(Q, t)


if __name__ == '__main__':
    run_module_suite()

