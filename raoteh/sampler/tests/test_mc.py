"""
Test Markov chain functions that do not require random sampling.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mc import (
        get_history_log_likelihood,
        )


class TestMJP(TestCase):

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

    def test_marginal_distributions(self):
        # Test the marginal distributions of node states.
        pass

    def test_joint_endpoint_distributions(self):
        # Test the marginal distributions of node states.
        pass


if __name__ == '__main__':
    run_module_suite()

