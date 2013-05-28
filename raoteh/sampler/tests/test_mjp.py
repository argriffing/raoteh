"""
Test functions related to the Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mjp import (
            get_history_dwell_times, get_history_root_state_and_transitions,
            get_total_rates, get_conditional_transition_matrix)


class TestMarkovJumpProcess(TestCase):

    def test_get_total_rates(self):
        Q = nx.DiGraph()
        Q.add_weighted_edges_from([
            (0, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (2, 1, 1)])
        observed = get_total_rates(Q)
        expected = {
                0 : 1,
                1 : 2,
                2 : 1}
        assert_equal(observed, expected)


if __name__ == '__main__':
    run_module_suite()

