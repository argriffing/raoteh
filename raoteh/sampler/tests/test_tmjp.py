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

from scipy import special

from raoteh.sampler import _sampler
from raoteh.sampler._util import (
                StructuralZeroProb, NumericalZeroProb, get_first_element)
from raoteh.sampler._mjp import (
            get_history_dwell_times, get_history_root_state_and_transitions,
            get_total_rates, get_conditional_transition_matrix)


class TestFullyAugmentedLikelihood(TestCase):

    def test_fully_augmented_likelihood_sufficient_statistics(self):
        # If we fix all of the parameters of the model except for the two
        # parameters that correspond to the tolerance transition rates,
        # then this model has low-dimensional sufficient statistics.
        # I think that these two parameters are associated
        # with three sufficient statistics.

        # Define a distribution over some primary states.
        nprimary = 4
        primary_distn = {
                0 : 0.1,
                1 : 0.2,
                2 : 0.3,
                3 : 0.4}

        # Define the transition rates.
        primary_transition_rates = [
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
        Q.add_weighted_edges_from(primary_transition_rates)

        # Define the tolerance birth and death rates.
        tolerance_rate_on = 0.5
        tolerance_rate_off = 1.5
        total_tolerance_rate = tolerance_rate_on + tolerance_rate_off
        tolerance_distn = {
                0 : tolerance_rate_off / total_tolerance_rate,
                1 : tolerance_rate_on / total_tolerance_rate}

        # Define a couple of tolerance classes.
        nparts = 2
        primary_to_part = {
                0 : 0,
                1 : 0,
                2 : 1,
                3 : 1}

        # Define a compound state space.
        compound_to_primary = []
        compound_to_tolerances = []
        for primary, tolerances in itertools.product(
                range(nprimary),
                itertools.product((0, 1), repeat=nparts)):
            compound_to_primary.append(primary)
            compound_to_tolerances.append(tolerances)

        # Define the sparse distribution over compound states.
        compound_distn = {}
        for i, (primary, tolerances) in enumerate(
                zip(compound_to_primary, compound_to_tolerances)):
            part = primary_to_part[primary]
            if tolerances[part] == 1:
                p_primary = primary_distn[primary]
                p_tolerances = 1.0
                for tolerance_class, tolerance_state in enumerate(tolerances):
                    if tolerance_class != part:
                        p_tolerances *= tolerance_distn[tolerance_state]
                compound_distn[i] = p_primary * p_tolerances

        # Check the number of entries in the compound state distribution.
        assert_equal(len(compound_distn), nprimary * (1 << (nparts - 1)))

        # Check that the distributions have the correct normalization.
        assert_allclose(sum(primary_distn.values()), 1)
        assert_allclose(sum(tolerance_distn.values()), 1)
        assert_allclose(sum(compound_distn.values()), 1)

        # Define the compound transition rate matrix.
        # Use compound_distn to avoid formal states with zero probability.
        # This is slow, but we do not need to be fast.
        Q_compound = nx.DiGraph()
        for i in compound_distn:
            for j in compound_distn:
                if i == j:
                    continue
                i_prim = compound_to_primary[i]
                j_prim = compound_to_primary[j]
                i_tols = compound_to_tolerances[i]
                j_tols = compound_to_tolerances[j]
                tol_pairs = list(enumerate(zip(i_tols, j_tols)))
                tol_diffs = [(k, x, y) for k, (x, y) in tol_pairs if x != y]
                tol_hdist = len(tol_diffs)

                # Look for a tolerance state change.
                # Do not allow simultaneous primary and tolerance changes.
                # Do not allow more than one simultaneousl tolerance change.
                # Do not allow changes to the primary tolerance class.
                if tol_hdist > 0:
                    if i_prim != j_prim:
                        continue
                    if tol_hdist > 1:
                        continue
                    part, i_tol, j_tol = tol_diffs[0]
                    if part == primary_to_part[i_prim]:
                        continue

                    # Add the transition rate.
                    if j_tol:
                        weight = tolerance_rate_on
                    else:
                        weight = tolerance_rate_off
                    Q_compound.add_edge(i, j, weight=weight)

                # Look for a primary state change.
                # Do not allow simultaneous primary and tolerance changes.
                # Do not allow a change to a non-tolerated primary class.
                # Do not allow transitions that have zero rate
                # in the primary process.
                if i_prim != j_prim:
                    if tol_hdist > 0:
                        continue
                    if not i_tols[primary_to_part[j_prim]]:
                        continue
                    if not Q.has_edge(i_prim, j_prim):
                        continue
                    
                    # Add the primary state transition rate.
                    weight = Q[i_prim][j_prim]['weight']
                    Q_compound.add_edge(i, j, weight=weight)

        # Define a tree with edge weights.
        T = nx.Graph()
        T.add_edge(0, 1, weight=0.1)
        T.add_edge(2, 1, weight=0.2)
        T.add_edge(1, 3, weight=5.3)
        T.add_edge(3, 4, weight=0.4)
        T.add_edge(3, 5, weight=0.5)

        # Summarize the total tree length.
        total_tree_length = sum(T[a][b]['weight'] for a, b in T.edges())

        # Randomly assign compound leaf states.
        choices = list(compound_distn)
        node_to_compound_state = dict(
                (n, random.choice(choices)) for n in (0, 2, 4, 5))

        # Test the likelihood calculations
        # for a few conditionally sampled histories on the tree.
        nhistories = 10
        for compound_process_history in itertools.islice(
                _sampler.gen_histories(T, Q_compound, node_to_compound_state),
                nhistories):

            # Summarize the compound process history.
            dwell_times = get_history_dwell_times(
                    compound_process_history)
            root_state, transitions = get_history_root_state_and_transitions(
                    compound_process_history)

            # Get the total rate away from each compound state.
            total_rates = get_total_rates(Q_compound)

            # Construct a transition matrix conditional on a state change.
            P = get_conditional_transition_matrix(Q_compound, total_rates)

            # Directly compute the log likelihood of the history.
            log_likelihood = 0.0
            log_likelihood += np.log(compound_distn[root_state])
            for compound_state, dwell_time in dwell_times.items():
                log_likelihood -= dwell_time * total_rates[compound_state]
            for a, b in transitions.edges():
                ntrans = transitions[a][b]['weight']
                ptrans = P[a][b]['weight']
                log_likelihood += special.xlogy(ntrans, ptrans)
            direct_log_likelihood = log_likelihood

            print('directly computed log_likelihood:', direct_log_likelihood)

            # Compute the log likelihood through sufficient statistics.

            # Get the number of tolerance gains
            # plus the number of initial tolerances,
            # and the number of tolerance losses
            # plus the number of initial lack-of-tolerances.
            ngains_stat = 0
            nlosses_stat = 0
            for tol in compound_to_tolerances[root_state]:
                if tol == 1:
                    ngains_stat += 1
                elif tol == 0:
                    ngains_stat += 1
                else:
                    raise Exception('invalid root tolerance state')
            for a, b in transitions.edges():
                if a == b:
                    continue
                ntransitions = transitions[a][b]['weight']
                prim_a = compound_to_primary[a]
                prim_b = compound_to_primary[b]
                tols_a = compound_to_tolerances[a]
                tols_b = compound_to_tolerances[b]
                tols_diff = [y-x for x, y in zip(tols_a, tols_b)]
                ndiffs = sum(1 for x in tols_diff if x)
                if prim_a == prim_b:
                    if ndiffs == 0:
                        raise Exception(
                                'expected each non-self transition '
                                'to have either a primary state change '
                                'or a tolerance state change')
                    elif ndiffs > 1:
                        raise Exception(
                                'expected at most one tolerance state '
                                'difference but observed %d' % ndiffs)
                    elif ndiffs != 1:
                        raise Exception('internal error')
                    signed_hdist = sum(tols_diff)
                    if signed_hdist == 1:
                        ngains_stat += ntransitions
                    elif signed_hdist == -1:
                        nlosses_stat += ntransitions
                    else:
                        raise Exception('invalid tolerance process transition')

            # Get the total amount of time spent in tolerated states,
            # summed over each tolerance class.
            tolerance_duration_stat = 0.0
            for compound_state, dwell_time in dwell_times.items():
                ntols = sum(compound_to_tolerances[compound_state])
                tolerance_duration_stat += dwell_time * ntols

            # Initialize the log likelihood for this more clever approach.
            log_likelihood = 0.0

            # Add the log likelihood contributions that involve
            # the sufficient statistics and the on/off tolerance rates.
            log_likelihood -= special.xlogy(nparts, total_tolerance_rate)
            log_likelihood += special.xlogy(ngains_stat, tolerance_rate_on)
            log_likelihood += special.xlogy(nlosses_stat, tolerance_rate_off)
            log_likelihood -= tolerance_rate_off * (
                    tolerance_duration_stat - total_tree_length)
            log_likelihood -= tolerance_rate_on * (
                    total_tree_length * nparts - tolerance_duration_stat)

            # Add the log likelihood contributions that involve
            # general functions of the data and not the on/off tolerance rates.
            # On the other hand, they do involve the tolerance state.
            root_primary_state = compound_to_primary[root_state]
            log_likelihood += np.log(primary_distn[root_primary_state])
            """
            for compound_state, dwell_time in dwell_times.items():
                primary_state = compound_to_primary[compound_state]
                primary_rate_out = 0.0
                for sink in Q_compound[compound_state]:
                    if compound_to_primary[sink] != primary_state:
                        rate = Q_compound[compound_state][sink]['weight']
                        primary_rate_out += rate
                log_likelihood -= dwell_time * primary_rate_out
            """
            for a, b in transitions.edges():
                edge = transitions[a][b]
                ntransitions = edge['weight']
                prim_a = compound_to_primary[a]
                prim_b = compound_to_primary[b]
                if prim_a != prim_b:
                    rate = Q_compound[a][b]['weight']
                    log_likelihood += special.xlogy(ntransitions, rate)

            print('cleverly computed log_likelihood:', log_likelihood)
            print()


        raise Exception('raise an exception to print some stuff')


if __name__ == '__main__':
    run_module_suite()

