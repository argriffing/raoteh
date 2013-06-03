"""
Test functions related to the Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mjp import (
            get_history_dwell_times, get_history_root_state_and_transitions,
            get_total_rates, get_conditional_transition_matrix,
            get_expected_history_statistics,
            )

from raoteh.sampler._conditional_expectation import (
        get_jukes_cantor_rate_matrix,
        get_jukes_cantor_interaction,
        get_jukes_cantor_probability,
        )


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

    def test_jukes_cantor_conditional_expectation(self):
        # Compare conditional expectations to the true values.

        # Define the elapsed time.
        t = 0.5

        # Define the tree, which in this case is only a path.
        T = nx.Graph()
        T.add_edge(0, 1, weight=0.1*t)
        T.add_edge(1, 2, weight=0.2*t)
        T.add_edge(2, 3, weight=0.3*t)
        T.add_edge(3, 4, weight=0.4*t)

        # Define the initial state, final state, and elapsed time.
        nstates = 4
        for a in range(nstates):
            for b in range(nstates):

                # Define the states at the two ends of the path.
                #node_to_state = {min(T):a, max(T):b}
                node_to_allowed_states = {
                        0 : {a},
                        1 : set(range(nstates)),
                        2 : set(range(nstates)),
                        3 : set(range(nstates)),
                        4 : {b},
                        }

                # Define the Jukes-Cantor rate matrix.
                Q = get_jukes_cantor_rate_matrix(nstates)

                # Compute the expected dwell times.
                expected_dwell_times = np.zeros(nstates, dtype=float)
                for i in range(nstates):
                    interaction = get_jukes_cantor_interaction(
                            a, b, i, i, t, nstates)
                    probability = get_jukes_cantor_probability(
                            a, b, t, nstates)
                    expected_dwell_times[i] = interaction / probability

                # Get the MJP expected history statistics.
                # Check the expected dwell times for various roots.
                for root in T:

                    # Get the expected dwell times by brute force.
                    info = get_expected_history_statistics(
                            T, node_to_allowed_states, root, Q_default=Q)
                    mjp_expected_dwell_times, mjp_expected_transitions = info

                    # Compare to the expected dwell times.
                    for i in range(nstates):
                        assert_allclose(
                                expected_dwell_times[i],
                                mjp_expected_dwell_times[i])


if __name__ == '__main__':
    run_module_suite()

