"""
Test functions related to the tolerance Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import random
import itertools

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

from raoteh.sampler import _sampler
from raoteh.sampler._util import (
                StructuralZeroProb, NumericalZeroProb, get_first_element)


class TestFullyAugmentedLikelihood(TestCase):

    def test_fully_augmented_likelihood_sufficient_statistics(self):
        # If we fix all of the parameters of the model except for the two
        # parameters that correspond to the tolerance transition rates,
        # then this model has low-dimensional sufficient statistics.
        # I think that these two parameters are associated
        # with three sufficient statistics.

        # Define a distribution over some primary states.
        nprimary_states = 4
        primary_distn = {
                0 : 0.1,
                1 : 0.2,
                2 : 0.3,
                4 : 0.4}

        # Define the transition rates.
        transition_rates = [
                (0, 1, 2 * primary_distn[1]),
                (1, 0, 2 * primary_distn[0]),
                (1, 2, primary_distn[2]),
                (2, 1, primary_distn[1]),
                (2, 3, 2 * primary_distn[3]),
                (3, 2, 2 * primary_distn[2]),
                (3, 0, primary_distn[0]),
                (0, 3, primary_distn[3]),
                ]

        # Define the primary process through its transition rate matrix.
        Q = nx.DiGraph()
        Q.add_weighted_edges_from(transition_rates)

        # Define the tolerance birth and death rates.
        tolerance_rate_on = 0.5
        tolerance_rate_off = 1.5
        total_tolerance_rate = tolerance_rate_on + tolerance_rate_off
        tolerance_distn = {
                0 : tolerance_rate_off / total_tolerance_rate,
                1 : tolerance_rate_on / total_tolerance_rate}

        # Define a couple of tolerance classes.
        nparts = 2
        state_to_part = {
                0 : 0,
                1 : 0,
                2 : 1,
                3 : 1}

        # Define a compound state space.
        compound_to_primary = []
        compound_to_tolerances = []
        for i, (primary, tolerances) in itertools.product(
                range(nprimary_states),
                itertools.product((0, 1), repeat=nparts)):
            compound_to_primary.append(primary)
            compound_to_tolerances.append(tolerances)

        # Define the distribution over compound states.
        compound_distn = {}
        for i, (primary, tolerances) in enumerate(
                zip(compound_to_primary, compound_to_tolerances)):
            p_primary = primary_distn[primary]
            p_tolerances = np.product(
                    tolerance_distn[tol] for tol in tolerances)
            compound_distn[i] = p_primary * p_tolerances

        # Define a tree with edge weights.
        T = nx.Graph()
        T.add_edge(0, 1, weight=0.1)
        T.add_edge(2, 1, weight=0.2)
        T.add_edge(1, 3, weight=5.3)
        T.add_edge(3, 4, weight=0.4)
        T.add_edge(3, 5, weight=0.5)

        # Randomly assign compound leaf states.
        node_to_state = dict(
                (n, random.randrange(ncompound)) for n in (0, 2, 4, 5))

        # Sample a few histories on the tree.
        # XXX under construction
        for foo in bar:
            Q_compound = nx.DiGraph()
            compound_rates = []
            Q_compound.add_weighted_edges_from(compound_rates)


if __name__ == '__main__':
    run_module_suite()

