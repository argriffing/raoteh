"""
Test Markov chain functions that do not require random sampling.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mjp import (
            get_history_dwell_times, get_history_root_state_and_transitions,
            get_total_rates, get_conditional_transition_matrix)


class TestMJP(TestCase):

    def test_marginal_distributions(self):
        # Test the marginal distributions of node states.
        pass

    def test_joint_endpoint_distributions(self):
        # Test the marginal distributions of node states.
        pass


if __name__ == '__main__':
    run_module_suite()

