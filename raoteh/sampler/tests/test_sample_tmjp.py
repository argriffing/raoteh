"""
Test functions that sample Markov jump trajectories on a tree.

These tests currently spam too much and are too slow to be useful,
so they are disabled.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)


from raoteh.sampler import _tmjp, _sample_tmjp

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)


class TestGetFeasibleHistory(TestCase):

    def test_get_feasible_history(self):

        # Define a base tree and root.
        T = nx.Graph()
        T.add_edge(0, 1, weight=2.0)
        T.add_edge(0, 2, weight=2.0)
        T.add_edge(0, 3, weight=2.0)
        root = 0

        # Define observed primary states at the leaves.
        node_to_primary_state = {1:0, 2:2, 3:4}

        # Define the tolerance transition rates.
        rate_on = 0.5
        rate_off = 1.5

        # Get the compound tolerance process toy model.
        info = _tmjp.get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q_primary, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize the compound process.
        nparts = len(set(primary_to_part.values()))

        # Get a feasible history.
        # This includes the primary trajectory and the tolerance trajectories.
        feasible_history = _sample_tmjp.get_feasible_history(
                T, root,
                Q_primary, primary_distn,
                primary_to_part, rate_on, rate_off,
                node_to_primary_state)
        primary_trajectory, tolerance_trajectories = feasible_history

        # Assert that the number of tolerance trajectories is correct.
        assert_equal(len(tolerance_trajectories), nparts)


if __name__ == '__main__':
    run_module_suite()

