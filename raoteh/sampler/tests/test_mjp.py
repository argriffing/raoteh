"""
Test functions related to the Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler import _density, _mjp, _mjp_dense

from raoteh.sampler._density import (
        dict_to_numpy_array,
        rate_matrix_to_numpy_array,
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
        observed = _mjp.get_total_rates(Q)
        expected = {
                0 : 1,
                1 : 2,
                2 : 1}
        assert_equal(observed, expected)

        # Check dense total rates.
        nstates = 3
        nodelist = range(nstates)
        Q_dense = rate_matrix_to_numpy_array(Q, nodelist=nodelist)
        desired_dense = dict_to_numpy_array(expected, nodelist=nodelist)
        actual_dense = _mjp_dense.get_total_rates(Q_dense)
        assert_allclose(actual_dense, desired_dense)

    def test_get_likelihood(self):
        T = nx.Graph()
        T.add_weighted_edges_from([
            (0, 1, 2.0),
            (0, 2, 3.0),
            (0, 3, 4.0),
            (1, 4, 5.0),
            (1, 5, 6.0),
            ])
        nstates = 4
        distn = {
                0 : 0.1,
                1 : 0.2,
                2 : 0.3,
                3 : 0.4,
                }
        Q = nx.DiGraph()
        Q.add_weighted_edges_from([
            (0, 1, 1.0 * distn[1]),
            (1, 0, 1.0 * distn[0]),
            (1, 2, 2.0 * distn[2]),
            (2, 1, 2.0 * distn[1]),
            (2, 3, 1.0 * distn[3]),
            (3, 2, 1.0 * distn[2]),
            (3, 0, 2.0 * distn[0]),
            (0, 3, 2.0 * distn[3]),
            ])
        node_to_allowed_states = {
                0 : set(range(nstates)),
                1 : set(range(nstates)),
                2 : {0},
                3 : {1},
                4 : {2},
                5 : {3},
                }
        distn_dense = _density.dict_to_numpy_array(
                distn, nodelist=range(nstates))
        Q_dense = _density.rate_matrix_to_numpy_array(
                Q, nodelist=range(nstates))
        for root in range(6):
            lk = _mjp.get_likelihood(
                    T, node_to_allowed_states, root,
                    root_distn=distn, Q_default=Q)
            lk_dense = _mjp_dense.get_likelihood(
                    T, node_to_allowed_states, root, nstates,
                    root_distn=distn_dense, Q_default=Q_dense)
            assert_allclose(lk, lk_dense)
        lk_marginal = lk
        lk_dense_marginal = lk_dense

        # Compute the likelihood by naively summing over all combinations
        # of ancestral states.
        lk_m = 0
        lk_dense_m = 0
        for s0 in range(4):
            for s1 in range(4):
                root = 0
                nodemap = dict(node_to_allowed_states)
                nodemap[0] = {s0}
                nodemap[1] = {s1}
                for root in range(6):
                    lk = _mjp.get_likelihood(
                            T, nodemap, root,
                            root_distn=distn, Q_default=Q)
                    lk_dense = _mjp_dense.get_likelihood(
                            T, nodemap, root, nstates,
                            root_distn=distn_dense, Q_default=Q_dense)
                    assert_allclose(lk, lk_dense)
                lk_m += lk
                lk_dense_m += lk_dense

        # Check that the answers are the same.
        assert_allclose(lk_marginal, lk_m)
        assert_allclose(lk_dense_marginal, lk_dense_m)

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

                # Sparse testing.

                # Get the MJP expected history statistics.
                # Check the expected dwell times for various roots.
                for root in T:

                    # Get the expected dwell times by brute force.
                    info = _mjp.get_expected_history_statistics(
                            T, node_to_allowed_states, root, Q_default=Q)
                    mjp_dwell, mjp_init, mjp_trans = info

                    # Compare to the expected dwell times.
                    for i in range(nstates):
                        assert_allclose(
                                expected_dwell_times[i],
                                mjp_dwell[i])

                # Dense testing.
                Q_dense = rate_matrix_to_numpy_array(Q)

                # Get the MJP expected history statistics.
                # Check the expected dwell times for various roots.
                for root in T:

                    # Get the expected dwell times by brute force.
                    info = _mjp_dense.get_expected_history_statistics(
                            T, node_to_allowed_states, root, nstates,
                            Q_default=Q_dense)
                    mjp_dwell_dense, mjp_init_dense, mjp_trans_dense = info

                    # Compare to the expected dwell times.
                    for i in range(nstates):
                        assert_allclose(
                                expected_dwell_times[i],
                                mjp_dwell_dense[i])


if __name__ == '__main__':
    run_module_suite()

