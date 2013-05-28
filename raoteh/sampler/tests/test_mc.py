"""
Test Markov chain functions that do not require random sampling.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mc import (
        get_history_log_likelihood, get_node_to_distn_naive,
        )


class TestMarkovChain(TestCase):

    def test_history_log_likelihood_P_default(self):
        T = nx.Graph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(0, 3)
        root = 0
        node_to_state = {0:0, 1:0, 2:0, 3:0}
        root_distn = {0 : 0.5, 1 : 0.5, 2 : 0, 3 : 0}
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            (0, 0, 0.5),
            (0, 1, 0.25),
            (0, 2, 0.25),
            (1, 1, 0.5),
            (1, 2, 0.25),
            (1, 0, 0.25),
            (2, 2, 0.5),
            (2, 0, 0.25),
            (2, 1, 0.25)])
        actual = get_history_log_likelihood(
                T, node_to_state, root, root_distn, P_default=P)
        desired = 4 * np.log(0.5)
        assert_equal(actual, desired)

    def test_node_to_distn_naive(self):
        # Test the marginal distributions of node states.

        # Try an example where no state is initially known,
        # but for which the transition matrix will cause
        # the joint distribution of states at all nodes to be
        # all in the same state.
        # This will cause all states to have the same distribution
        # as the root distribution.
        nstates = 3
        states = range(nstates)
        P = nx.DiGraph()
        P.add_weighted_edges_from([(s, s, 1) for s in states])
        T = nx.Graph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(0, 3)
        root = 0
        node_to_allowed_states = dict((n, set(states)) for n in T)
        for root_distn in (
                {0 : 0.10, 1 : 0.40, 2 : 0.50},
                {0 : 0.25, 1 : 0.50, 2 : 0.25},
                ):
            node_to_distn = get_node_to_distn_naive(T, node_to_allowed_states,
                    root, root_distn, P)
            for node, distn in node_to_distn.items():
                assert_equal(distn, root_distn)

    def test_marginal_distributions(self):
        # Test the marginal distributions of node states.
        pass

    def test_joint_endpoint_distributions(self):
        # Test the marginal distributions of node states.
        pass


if __name__ == '__main__':
    run_module_suite()

