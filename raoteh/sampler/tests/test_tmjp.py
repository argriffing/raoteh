"""
Test functions related to the tolerance Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import random
import itertools
import functools
from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.linalg
from scipy import special
from scipy import optimize

from numpy.testing import (
        run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises, decorators,
        )

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element,
        )

from raoteh.sampler._graph_transform import(
        get_redundant_degree_two_nodes,
        remove_redundant_nodes,
        )

from raoteh.sampler._mc import (
        get_restricted_likelihood,
        construct_node_to_restricted_pmap,
        get_node_to_distn,
        )

from raoteh.sampler._mjp import (
        get_history_dwell_times,
        get_history_root_state_and_transitions,
        get_history_statistics,
        get_total_rates,
        get_conditional_transition_matrix,
        get_expm_augmented_tree,
        get_expected_history_statistics,
        )

from raoteh.sampler._tmjp import (
        get_tolerance_process_log_likelihood,
        )

from raoteh.sampler._sampler import (
        gen_histories,
        gen_restricted_histories,
        gen_mh_histories,
        get_forward_sample,
        )

from raoteh.sampler._sample_tree import (
        get_random_branching_tree, get_random_agglom_tree,
        )


def _get_tolerance_process_info(tolerance_rate_on, tolerance_rate_off):
    """

    Returns
    -------
    primary_distn : dict
        Primary process state distribution.
    Q : weighted directed networkx graph
        Sparse primary process transition rate matrix.
    primary_to_part : dict
        Maps primary state to tolerance class.
    compound_to_primary : list
        foo
    compound_to_tolerances : list
        foo
    compound_distn : dict
        foo
    Q_compound : weighted directed networkx graph
        Sparse compound process transition rate matrix.

    """
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

    # Define some tolerance process stuff.
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
            # Do not allow more than one simultaneous tolerance change.
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

    return (primary_distn, Q, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound)


def _neg_log_likelihood_for_minimization(
        T_seq, root, node_to_allowed_states_seq, X):
    """
    This is intended to be functools.partial'd and sent to scipy.optimize.

    Parameters
    ----------
    T_seq : sequence of weighted undirected acyclic networkx graphs
        Tree with edge weights and annotated with allowed states on edges.
    root : integer
        Root node.
    node_to_allowed_states_seq : sequence
        List of maps from node to set of allowed compound states.
    X : one dimensional ndarray containing two floats
        This is the vector of parameters for minimization.
        It contains the log of the tolerance_rate_on
        and the log of the tolerance_rate_off.

    Returns
    -------
    neg_log_likelihood : float
        Negative of log likelihood, for minimization.

    """

    # Extract the tolerance rate args which are being optimized.
    rate_on, rate_off = np.exp(X)

    # Get the compound rate matrix and stationary distribution.
    (primary_distn, Q, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound) = _get_tolerance_process_info(rate_on, rate_off)

    # Return the negative log likelihood for minimization.
    log_likelihood = 0.0
    for T, node_to_allowed_states in zip(T_seq, node_to_allowed_states_seq):

        # Compute some matrix exponentials
        # and put them on the edges of the tree,
        # respecting edge state restrictions.
        T_aug = get_expm_augmented_tree(T, root, Q_default=Q_compound)
        likelihood = get_restricted_likelihood(
                T_aug, root, node_to_allowed_states, compound_distn)
        log_likelihood += np.log(likelihood)
    neg_log_likelihood = -log_likelihood
    return neg_log_likelihood


class TestExpectationMaximization(TestCase):

    def xfail_expectation_maximization(self):
        # It should be possible to use expectation maximization
        # to find locally optimal rate-on and rate-off values
        # in the sense of maximum likelihood.
        # Compare to a brute force search using scipy.optimize.

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = _get_tolerance_process_info(rate_on, rate_off)

        # Summarize properties of the process.
        ncompound = len(compound_to_primary)
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        total_tolerance_rate = rate_on + rate_off
        tolerance_distn = {
                0 : rate_off / total_tolerance_rate,
                1 : rate_on / total_tolerance_rate}

        # Sample a non-tiny random tree without branch lengths.
        #branching_distn = [0.4, 0.3, 0.2, 0.1]
        #T = get_random_branching_tree(branching_distn, maxnodes=50)
        T = get_random_agglom_tree(maxnodes=20)
        root = 0

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            #scale = 1.0
            scale = 0.1
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Track the primary process.
        ncols = 5
        T_seq = []
        node_to_allowed_states_seq = []
        for col in range(ncols):

            # Use forward sampling to jointly sample compound states
            # on the tree, according to the compound process
            # and its stationary distribution at the root.
            T_forward_sample = get_forward_sample(
                    T, Q_compound, root, compound_distn)

            # Construct the tree with edge-restricted compound states.
            T_restricted = nx.Graph()
            for na, nb in nx.bfs_edges(T_forward_sample, root):
                compound_state = T_forward_sample[na][nb]['state']
                primary_state = compound_to_primary[compound_state]
                allowed = set()
                for s in range(ncompound):
                    if compound_to_primary[s] == primary_state:
                        allowed.add(s)
                weight = T_forward_sample[na][nb]['weight']
                T_restricted.add_edge(na, nb, weight=weight, allowed=allowed)

            # Construct the compound state restrictions at the nodes.
            node_to_allowed_states = {}
            for na in T_forward_sample:
                neighbor_primary_states = set()
                for nb in T_forward_sample[na]:
                    edge = T_forward_sample[na][nb]
                    compound_state = edge['state']
                    primary_state = compound_to_primary[compound_state]
                    neighbor_primary_states.add(primary_state)
                allowed = set()
                for s in range(ncompound):
                    if compound_to_primary[s] in neighbor_primary_states:
                        allowed.add(s)
                node_to_allowed_states[na] = allowed

            # Append to the state restriction sequences.
            T_seq.append(T_restricted)
            node_to_allowed_states_seq.append(node_to_allowed_states)

        # Report simulation parameter value neg log likelihood.
        X_true = np.log([rate_on, rate_off])
        true_neg_ll = _neg_log_likelihood_for_minimization(
                T_seq, root, node_to_allowed_states_seq, X_true)
        print('neg ll for true simulation parameter values:', true_neg_ll)

        # Report initial parameter value neg log likelihood.
        X0 = np.zeros(2, dtype=float)
        initial_neg_ll = _neg_log_likelihood_for_minimization(
                T_seq, root, node_to_allowed_states_seq, X0)
        print('neg ll for initial parameter values:', initial_neg_ll)

        # Report the results of the optimization.
        f = functools.partial(
                _neg_log_likelihood_for_minimization,
                T_seq, root, node_to_allowed_states_seq)
        method = 'BFGS'
        results = scipy.optimize.minimize(f, X0, method=method)
        print('results of', method, 'minimization:', results)

        raise Exception('print stuff')


class TestFullyAugmentedLikelihood(TestCase):

    def test_fully_augmented_likelihood_sufficient_statistics(self):
        # If we fix all of the parameters of the model except for the two
        # parameters that correspond to the tolerance transition rates,
        # then this model has low-dimensional sufficient statistics.
        # I think that these two parameters are associated
        # with three sufficient statistics.

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = _get_tolerance_process_info(rate_on, rate_off)

        # Summarize the other properties.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        total_tolerance_rate = rate_on + rate_off
        tolerance_distn = {
                0 : rate_off / total_tolerance_rate,
                1 : rate_on / total_tolerance_rate}

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
        for compound_process_history in gen_histories(
                T, Q_compound, node_to_compound_state, nhistories=nhistories):

            # Summarize the compound process history.
            dwell_times = get_history_dwell_times(
                    compound_process_history)
            root_state, transitions = get_history_root_state_and_transitions(
                    compound_process_history)

            # Get the total rate away from each compound state.
            total_rates = get_total_rates(Q_compound)

            # Construct a transition matrix conditional on a state change.
            #P = get_conditional_transition_matrix(Q_compound, total_rates)

            # Directly compute the log likelihood of the history.
            ll_initial = np.log(compound_distn[root_state])
            ll_dwell = 0.0
            for compound_state, dwell_time in dwell_times.items():
                ll_dwell -= dwell_time * total_rates[compound_state]
            ll_transitions = 0.0
            for a, b in transitions.edges():
                ntrans = transitions[a][b]['weight']
                rate = Q_compound[a][b]['weight']
                ll_transitions += special.xlogy(ntrans, rate)

            direct_ll_initrans = ll_initial + ll_transitions
            direct_ll_dwell = ll_dwell

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
                    nlosses_stat += 1
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
            ll_initrans = 0.0
            ll_dwell = 0.0

            # Add the log likelihood contributions that involve
            # the sufficient statistics and the on/off tolerance rates.
            ll_initrans -= special.xlogy(nparts-1, total_tolerance_rate)
            ll_initrans += special.xlogy(ngains_stat-1, rate_on)
            ll_initrans += special.xlogy(nlosses_stat, rate_off)
            ll_dwell -= rate_off * (
                    tolerance_duration_stat - total_tree_length)
            ll_dwell -= rate_on * (
                    total_tree_length * nparts - tolerance_duration_stat)

            # Add the log likelihood contributions that involve
            # general functions of the data and not the on/off tolerance rates.
            # On the other hand, they do involve the tolerance state.
            root_primary_state = compound_to_primary[root_state]
            ll_initrans += np.log(primary_distn[root_primary_state])
            for compound_state, dwell_time in dwell_times.items():
                primary_state = compound_to_primary[compound_state]
                primary_rate_out = 0.0
                for sink in Q_compound[compound_state]:
                    if compound_to_primary[sink] != primary_state:
                        rate = Q_compound[compound_state][sink]['weight']
                        primary_rate_out += rate
                ll_dwell -= dwell_time * primary_rate_out
            for a, b in transitions.edges():
                edge = transitions[a][b]
                ntransitions = edge['weight']
                prim_a = compound_to_primary[a]
                prim_b = compound_to_primary[b]
                if prim_a != prim_b:
                    rate = Q_compound[a][b]['weight']
                    ll_initrans += special.xlogy(ntransitions, rate)

            clever_ll_initrans = ll_initrans
            clever_ll_dwell = ll_dwell

            # Compare the two log likelihood calculations.
            assert_allclose(direct_ll_initrans, clever_ll_initrans)
            assert_allclose(direct_ll_dwell, clever_ll_dwell)


class TestToleranceProcessMarginalLogLikelihood(TestCase):

    def xfail_tolerance_process_log_likelihood(self):

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = _get_tolerance_process_info(rate_on, rate_off)

        # Summarize properties of the process.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        total_tolerance_rate = rate_on + rate_off
        tolerance_distn = {
                0 : rate_off / total_tolerance_rate,
                1 : rate_on / total_tolerance_rate}

        # Sample a non-tiny random tree without branch lengths.
        T = get_random_agglom_tree(maxnodes=20)
        root = 0

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            scale = 0.1
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Use forward sampling to jointly sample compound states on the tree,
        # according to the compound process and its stationary distribution
        # at the root.
        T_forward_sample = get_forward_sample(
                T, Q_compound, root, compound_distn)

        # Track only the primary process information at leaves.
        # Do not track the any tolerance process information at leaves,
        # and track neither primary nor tolerance process information
        # at non-leaf vertices or on any edge.
        node_to_allowed_states = {}
        for node in T:
            if len(T[node]) == 1:
                # Allow only compound states
                # that have the the observed primary state.
                neighbor = get_first_element(T_forward_sample[node])
                compound_state = T_forward_sample[node][neighbor]['state']
                primary_state = compound_to_primary[compound_state]
                allowed_states = set()
                for comp, prim in enumerate(compound_to_primary):
                    if prim == primary_state:
                        allowed_states.add(comp)
            else:
                # Allow all compound states.
                allowed_states = set(range(len(compound_to_primary)))
            node_to_allowed_states[node] = allowed_states

        # Compute the log likelihood, directly using the compound process.
        X_true = np.log([rate_on, rate_off])
        neg_ll_direct = _neg_log_likelihood_for_minimization(
                T, root, node_to_allowed_states_seq, X_true)
        neg_ll_clever = -get_tolerance_process_log_likelihood(
                Q, state_to_part, T, node_to_tmap,
                rate_off, rate_on, primary_distn, root)


#TODO move this function into _tmjp
def get_tolerance_expectations(
        primary_to_part, rate_on, rate_off, Q_primary, T_primary, root):
    """
    Get tolerance process expectations conditional on a primary trajectory.

    Given a primary process trajectory,
    compute some log likelihood contributions
    related to the tolerance process.

    Parameters
    ----------
    primary_to_part : x
        x
    T_primary : x
        x
    Q_primary : x
        x
    root : x
        x
    rate_on : x
        x
    rate_off : x
        x

    Returns
    -------
    dwell_ll_contrib : float
        Log likelihood contribution of the conditional
        tolerance state dwell times.
    init_ll_contrib : float
        Log likelihood contribution of the conditional
        initial distribution of tolerance states.
    trans_ll_contrib : float
        Log likelihood contribution of the conditional
        tolerance process transition counts and types.

    Notes
    -----
    This function assumes that no tolerance process data is observed
    at the leaves, other than what could be inferred through
    the primary process observations at the leaves.
    The ordering of the arguments of this function was chosen haphazardly.
    The ordering of the return values corresponds to the ordering of
    the return values of the _mjp function that returns
    statistics of a trajectory.

    """
    # Summarize the tolerance process.
    nparts = len(set(primary_to_part.values()))
    total_tolerance_rate = rate_on + rate_off
    tolerance_distn = {
            0 : rate_off / total_tolerance_rate,
            1 : rate_on / total_tolerance_rate}

    # Compute conditional expectations of statistics
    # of the tolerance process.
    # This requires constructing independent piecewise homogeneous
    # Markov jump processes for each tolerance class.
    expected_ngains = 0.0
    expected_nlosses = 0.0
    expected_dwell_on = 0.0
    expected_initial_on = 0.0
    for tolerance_class in range(nparts):

        # Define the set of allowed tolerances at each node.
        # These may be further constrained
        # by the sampled primary process trajectory.
        node_to_allowed_tolerances = dict(
                (n, {0, 1}) for n in T_primary)

        # Construct the tree whose edges are in correspondence
        # to the edges of the sampled primary trajectory,
        # and whose edges are annotated with weights
        # and with edge-specific 3-state transition rate matrices.
        # The third state of each edge-specific rate matrix is an
        # absorbing state which will never be entered.
        T_tol = nx.Graph()
        for na, nb in nx.bfs_edges(T_primary, root):
            edge = T_primary[na][nb]
            primary_state = edge['state']
            local_tolerance_class = primary_to_part[primary_state]
            weight = edge['weight']

            # Define the local on->off rate, off->on rate,
            # and absorption rate.
            local_rate_on = rate_on
            if tolerance_class == local_tolerance_class:
                local_rate_off = 0
            else:
                local_rate_off = rate_off
            absorption_rate = 0
            if primary_state in Q_primary:
                for sb in Q_primary[primary_state]:
                    if primary_to_part[sb] == tolerance_class:
                        rate = Q_primary[primary_state][sb]['weight']
                        absorption_rate += rate

            # Construct the local tolerance rate matrix.
            Q_tol = nx.DiGraph()
            if local_rate_on:
                Q_tol.add_edge(0, 1, weight=local_rate_on)
            if local_rate_off:
                Q_tol.add_edge(1, 0, weight=local_rate_off)
            if absorption_rate:
                Q_tol.add_edge(1, 2, weight=absorption_rate)

            # Add the edge.
            T_tol.add_edge(na, nb, weight=weight, Q=Q_tol)

            # Possibly restrict the set of allowed tolerances
            # at the endpoints of the edge.
            if tolerance_class == local_tolerance_class:
                for n in (na, nb):
                    node_to_allowed_tolerances[n].discard(0)

        # Compute conditional expectations of dwell times
        # and transitions for this tolerance class.
        expectation_info = get_expected_history_statistics(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=tolerance_distn)
        dwell_times, post_root_distn, transitions = expectation_info

        # Get the dwell time log likelihood contribution.
        if 1 in dwell_times:
            expected_dwell_on += dwell_times[1]

        # Get the transition log likelihood contribution.
        if transitions.has_edge(0, 1):
            expected_ngains += transitions[0][1]['weight']
        if transitions.has_edge(1, 0):
            expected_nlosses += transitions[1][0]['weight']

        # Get the initial state log likelihood contribution.
        if 1 in post_root_distn:
            expected_initial_on += post_root_distn[1]

    # Summarize expectations.
    expected_initial_off = nparts - expected_initial_on

    # Return the log likelihood contributions.
    dwell_ll_contrib = 0 #FIXME
    init_ll_contrib = (
            special.xlogy(expected_initial_on - 1, tolerance_distn[1]) +
            special.xlogy(expected_initial_off, tolerance_distn[0]))
    trans_ll_contrib = (
            special.xlogy(expected_ngains, rate_on) +
            special.xlogy(expected_nlosses, rate_off))
    return dwell_ll_contrib, init_ll_contrib, trans_ll_contrib


class TestToleranceProcessExpectedLogLikelihood(TestCase):

    #TODO split this function into multiple parts
    #@decorators.skipif(True)
    def test_tmjp_monte_carlo_rao_teh_differential_entropy(self):
        # In this test, we look at conditional expected log likelihoods.
        # These are computed in two ways.
        # The first way is by exponential integration using expm_frechet.
        # The second way is by Rao-Teh sampling.

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        (primary_distn, Q_primary, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = _get_tolerance_process_info(rate_on, rate_off)

        # Summarize properties of the process.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        total_rates = get_total_rates(Q_compound)
        total_tolerance_rate = rate_on + rate_off
        tolerance_distn = {
                0 : rate_off / total_tolerance_rate,
                1 : rate_on / total_tolerance_rate}

        # Sample a non-tiny random tree without branch lengths.
        T = get_random_agglom_tree(maxnodes=30)
        root = 0

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            scale = 0.4
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Use forward sampling to jointly sample compound states on the tree,
        # according to the compound process and its stationary distribution
        # at the root.
        T_forward_sample = get_forward_sample(
                T, Q_compound, root, compound_distn)

        # Track only the primary process information at leaves.
        # Do not track the any tolerance process information at leaves,
        # and track neither primary nor tolerance process information
        # at non-leaf vertices or on any edge.
        node_to_allowed_states = {}
        for node in T:

            # For nodes of degree 1, allow only compound states
            # that share the primary state with the sampled compound state.
            # For the remaining nodes, allow all possible compound states.
            deg = len(T[node])
            if deg == 1:
                neighbor = get_first_element(T_forward_sample[node])
                compound_state = T_forward_sample[node][neighbor]['state']
                primary_state = compound_to_primary[compound_state]
                allowed_states = set()
                for comp, prim in enumerate(compound_to_primary):
                    if prim == primary_state:
                        allowed_states.add(comp)
            else:
                allowed_states = set(range(len(compound_to_primary)))
            node_to_allowed_states[node] = allowed_states

        # Compute the conditional expected log likelihood explicitly
        # using some Markov jump process functions.

        # Annotate the edges of the tree
        # transition probability matrices which account for
        # the compound process transition rate matrix
        # and the length of the edge.
        T_trans = get_expm_augmented_tree(T, root, Q_default=Q_compound)
        #node_to_pmap = construct_node_to_restricted_pmap(
                #T_trans, root, node_to_allowed_states)
        #posterior_node_to_distn = get_node_to_distn(
                #T_trans, node_to_allowed_states, node_to_pmap,
                #root, prior_root_distn=compound_distn)
        #posterior_root_distn = posterior_node_to_distn[root]

        # Get some posterior expectations.
        expectation_info = get_expected_history_statistics(
                T, node_to_allowed_states,
                root, root_distn=compound_distn, Q_default=Q_compound)
        dwell_times, post_root_distn, transitions = expectation_info

        # Use some posterior expectations
        # to get the root distribution contribution to differential entropy.
        diff_ent_init = 0.0
        for state, prob in post_root_distn.items():
            diff_ent_init -= special.xlogy(prob, compound_distn[state])

        # Use some posterior expectations
        # to get the dwell time contribution to differential entropy.
        diff_ent_dwell = 0.0
        for s, rate in total_rates.items():
            diff_ent_dwell += dwell_times[s] * rate

        # Use some posterior expectations
        # to get the transition contribution to differential entropy.
        diff_ent_trans = 0.0
        for sa in set(Q_compound) & set(transitions):
            for sb in set(Q_compound[sa]) & set(transitions[sa]):
                rate = Q_compound[sa][sb]['weight']
                ntrans_expected = transitions[sa][sb]['weight']
                diff_ent_trans -= ntrans_expected * np.log(rate)

        # Define the number of samples.
        nsamples = 500
        sqrt_nsamp = np.sqrt(nsamples)

        # Do some Rao-Teh conditional samples,
        # and get the negative expected log likelihood.
        #
        # statistics for the full process samples
        sampled_root_distn = defaultdict(float)
        neg_ll_contribs_init = []
        neg_ll_contribs_dwell = []
        neg_ll_contribs_trans = []
        #
        # Statistics for the partial process samples.
        # The idea for the pm_ prefix is that we can use the compound
        # process Rao-Teh to sample the pure primary process without bias
        # if we throw away the tolerance process information.
        # Then with these unbiased primary process history samples,
        # we can compute conditional expectations of statistics of interest
        # by integrating over the possible tolerance trajectories.
        # This allows us to test this integration without worrying that
        # the primary process history samples are biased.
        #pm_neg_ll_contribs_dwell = []
        pm_neg_ll_contribs_init = []
        pm_neg_ll_contribs_trans = []
        #
        for T_aug in gen_restricted_histories(
                T, Q_compound, node_to_allowed_states,
                root=root, root_distn=compound_distn,
                uniformization_factor=2, nhistories=nsamples):

            # Get some stats of the histories.
            info = get_history_statistics(T_aug, root=root)
            dwell_times, root_state, transitions = info

            # contribution of root state to log likelihood
            sampled_root_distn[root_state] += 1.0 / nsamples
            ll = np.log(compound_distn[root_state])
            neg_ll_contribs_init.append(-ll)

            # contribution of dwell times
            ll = 0.0
            for state, dwell in dwell_times.items():
                ll -= dwell * total_rates[state]
            neg_ll_contribs_dwell.append(-ll)

            # contribution of transitions
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q_compound[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            neg_ll_contribs_trans.append(-ll)

            # Get a tree annotated with only the primary process,
            # after having thrown away the sampled tolerance
            # process data.

            # First copy the unbiased compound state trajectory tree.
            # Then convert the state annotation from compound state
            # to primary state.
            # Then use graph transformations to detect and remove
            # degree-2 vertices whose adjacent states are identical.
            T_primary_aug = T_aug.copy()
            for na, nb in nx.bfs_edges(T_primary_aug, root):
                edge = T_primary_aug[na][nb]
                compound_state = edge['state']
                primary_state = compound_to_primary[compound_state]
                edge['state'] = primary_state
            extras = get_redundant_degree_two_nodes(T_primary_aug) - {root}
            T_primary_aug = remove_redundant_nodes(T_primary_aug, extras)

            # Get primary trajectory stats.
            primary_info = get_history_statistics(T_primary_aug, root=root)
            dwell_times, root_state, transitions = primary_info

            # Add primary process initial contribution.
            init_prim_ll = np.log(primary_distn[root_state])

            # Add the transition stat contribution of the primary process.
            trans_prim_ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q_primary[sa][sb]['weight']
                trans_prim_ll += special.xlogy(ntransitions, rate)

            # Get pm_ ll contributions of expectations of
            # tolerance process transitions.
            tol_info = get_tolerance_expectations(
                    primary_to_part, rate_on, rate_off,
                    Q_primary, T_primary_aug, root)
            dwell_tol_ll, init_tol_ll, trans_tol_ll = tol_info

            # Append the pm_ neg ll trans component of log likelihood.
            pm_trans_ll = trans_prim_ll + trans_tol_ll
            pm_neg_ll_contribs_trans.append(-pm_trans_ll)

            # Append the pm_ neg ll init component of log likelihood.
            pm_init_ll = init_prim_ll + init_tol_ll
            pm_neg_ll_contribs_init.append(-pm_init_ll)

        # Define a rate matrix for a primary process proposal distribution.
        # This is intended to define a Markov jump process for primary states
        # which approximates the non-Markov jump process for primary states
        # defined by the marginal primary component of the compound process.
        # This biased proposal primary process can be used for either
        # importance sampling or for a Metropolis-Hastings step
        # within the Rao-Teh sampling.
        Q_primary_proposal = nx.DiGraph()
        for sa, sb in Q_primary.edges():
            primary_rate = Q_primary[sa][sb]['weight']
            if primary_to_part[sa] == primary_to_part[sb]:
                proposal_rate = primary_rate * tolerance_distn[1]
            else:
                proposal_rate = primary_rate
            Q_primary_proposal.add_edge(sa, sb, weight=proposal_rate)

        # Summarize the primary proposal rates.
        primary_proposal_total_rates = get_total_rates(Q_primary_proposal)

        # Summarize the primary rates.
        primary_total_rates = get_total_rates(Q_primary)

        # Get the map from leaf node to primary state.
        leaf_to_primary_state = {}
        for node in T:
            deg = len(T[node])
            if deg == 1:
                neighbor = get_first_element(T_forward_sample[node])
                compound_state = T_forward_sample[node][neighbor]['state']
                primary_state = compound_to_primary[compound_state]
                leaf_to_primary_state[node] = primary_state

        # Get Rao-Teh samples of primary process trajectories
        # conditional on the primary states at the leaves.
        # The imp_ prefix denotes importance sampling.
        imp_sampled_root_distn = defaultdict(float)
        imp_neg_ll_contribs_init = []
        imp_neg_ll_contribs_dwell = []
        imp_neg_ll_contribs_trans = []
        importance_weights = []
        for T_aug in gen_histories(
                T, Q_primary_proposal, leaf_to_primary_state,
                root=root, root_distn=primary_distn,
                uniformization_factor=2, nhistories=nsamples):

            # Compute primary process statistics.
            # These will be used for two purposes.
            # One of the purposes is as the denominator of the
            # importance sampling ratio.
            # The second purpose is to compute contributions
            # to the neg log likelihood estimate.
            info = get_history_statistics(T_aug, root=root)
            dwell_times, root_state, transitions = info


            # Proposal primary process summary.

            # contribution of root state to log likelihood
            proposal_init_ll = np.log(primary_distn[root_state])

            # contribution of dwell times
            ll = 0.0
            for state, dwell in dwell_times.items():
                ll -= dwell * primary_proposal_total_rates[state]
            proposal_dwell_ll = ll

            # contribution of transitions
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q_primary_proposal[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            proposal_trans_ll = ll

            # Get the proposal log likelihood
            # by summing the log likelihood contributions.
            proposal_ll = sum((
                proposal_init_ll,
                proposal_dwell_ll,
                proposal_trans_ll))


            # Non-proposal primary process summary.

            # contribution of root state to log likelihood
            primary_init_ll = np.log(primary_distn[root_state])

            # contribution of dwell times
            ll = 0.0
            for state, dwell in dwell_times.items():
                ll -= dwell * primary_total_rates[state]
            primary_dwell_ll = ll

            # contribution of transitions
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q_primary[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            primary_trans_ll = ll

            # Get the proposal log likelihood
            # by summing the log likelihood contributions.
            primary_ll = sum((
                primary_init_ll,
                primary_dwell_ll,
                primary_trans_ll))


            # Compute the importance sampling ratio.
            # This requires computing the primary proposal likelihood,
            # and also computing the marginal likelihood
            # of the primary process by integrating over
            # tolerance histories.
            # For now, we will assume that the tolerance histories
            # are completely unobserved, although they are somewhat
            # constrained by the sampled primary trajectory.
            # If we begin to include disease data into the analysis,
            # the tolerance histories will become more constrained.
            node_to_tmap = {}
            tolerance_ll = get_tolerance_process_log_likelihood(
                    Q_primary, primary_to_part, T_aug, node_to_tmap,
                    rate_off, rate_on, primary_distn, root)
            importance_weight = np.exp(tolerance_ll - proposal_ll)

            # Append the importance weight to the list.
            importance_weights.append(importance_weight)

            # Get log likelihood contributions of expectations of
            # tolerance process transitions.
            tol_info = get_tolerance_expectations(
                    primary_to_part, rate_on, rate_off,
                    Q_primary, T_aug, root)
            dwell_tol_ll, init_tol_ll, trans_tol_ll = tol_info

            # Add the log likelihood contribution
            # of the initial state distribution.
            imp_init_ll = primary_init_ll + init_tol_ll
            imp_neg_ll_contribs_init.append(-imp_init_ll)

            # Add the log likelihood contributions of the primary transitions.
            imp_trans_ll = primary_trans_ll + trans_tol_ll
            imp_neg_ll_contribs_trans.append(-imp_trans_ll)

            # XXX for now ignore this contribution
            # Add the log likelihood contributions
            # of the dwell times.
            imp_neg_ll_contribs_dwell.append(0.0)

        # define some expectations
        normalized_imp_trans = (np.array(importance_weights) * np.array(
                imp_neg_ll_contribs_trans)) / np.mean(importance_weights)
        normalized_imp_init = (np.array(importance_weights) * np.array(
                imp_neg_ll_contribs_init)) / np.mean(importance_weights)


        # Get Rao-Teh samples of primary process trajectories
        # conditional on the primary states at the leaves.
        # The met_ prefix denotes Metropolis-Hastings.
        met_sampled_root_distn = defaultdict(float)
        met_neg_ll_contribs_init = []
        met_neg_ll_contribs_dwell = []
        met_neg_ll_contribs_trans = []
        naccepted = 0
        nrejected = 0

        # Define the callback.
        def target_log_likelihood_callback(T_aug):
            node_to_tmap = {}
            return get_tolerance_process_log_likelihood(
                    Q_primary, primary_to_part, T_aug, node_to_tmap,
                    rate_off, rate_on, primary_distn, root)

        # Convert between restricted state formats.
        node_to_allowed_primary_states = {}
        for node in T:
            if node in leaf_to_primary_state:
                allowed = {leaf_to_primary_state[node]}
            else:
                allowed = set(primary_to_part)
            node_to_allowed_primary_states[node] = allowed

        # Sample the histories.
        for T_aug, accept_flag in gen_mh_histories(
                T, Q_primary_proposal, node_to_allowed_primary_states,
                target_log_likelihood_callback,
                root, root_distn=primary_distn,
                uniformization_factor=2, nhistories=nsamples):

            if accept_flag:
                naccepted += 1
            else:
                nrejected += 1

            # Compute primary process statistics.
            info = get_history_statistics(T_aug, root=root)
            dwell_times, root_state, transitions = info

            # Contribution of root state to log likelihood.
            primary_init_ll = np.log(primary_distn[root_state])

            # Count the contribution of primary process transitions.
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q_primary[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            primary_trans_ll = ll

            # Get log likelihood contributions of expectations of
            # tolerance process transitions.
            tol_info = get_tolerance_expectations(
                    primary_to_part, rate_on, rate_off,
                    Q_primary, T_aug, root)
            dwell_tol_ll, init_tol_ll, trans_tol_ll = tol_info

            # Add the log likelihood contributions.
            met_trans_ll = primary_trans_ll + trans_tol_ll
            met_neg_ll_contribs_trans.append(-met_trans_ll)
            met_init_ll = primary_init_ll + init_tol_ll
            met_neg_ll_contribs_init.append(-met_init_ll)


        print()
        print('--- tmjp experiment ---')
        print('nsamples:', nsamples)
        print()
        print('sampled root distn :', sampled_root_distn)
        print('analytic root distn:', post_root_distn)
        print()
        print('diff ent init :', diff_ent_init)
        print('neg ll init   :', np.mean(neg_ll_contribs_init))
        print('error         :', np.std(neg_ll_contribs_init) / sqrt_nsamp)
        print('pm neg ll init:', np.mean(pm_neg_ll_contribs_init))
        print('error         :', np.std(pm_neg_ll_contribs_init) / sqrt_nsamp)
        print('imp init      :', np.mean(normalized_imp_init))
        print('error         :', np.std(normalized_imp_init) / sqrt_nsamp)
        print('met init      :', np.mean(met_neg_ll_contribs_init))
        print('error         :', np.std(met_neg_ll_contribs_init) / sqrt_nsamp)
        print()
        print('diff ent dwell:', diff_ent_dwell)
        print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
        print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamp)
        print()
        print('diff ent trans:', diff_ent_trans)
        print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
        print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamp)
        print('pm neg ll tran:', np.mean(pm_neg_ll_contribs_trans))
        print('error         :', np.std(pm_neg_ll_contribs_trans) / sqrt_nsamp)
        print('imp trans     :', np.mean(normalized_imp_trans))
        print('error         :', np.std(normalized_imp_trans) / sqrt_nsamp)
        print('met trans     :', np.mean(met_neg_ll_contribs_trans))
        print('error         :', np.std(met_neg_ll_contribs_trans) / sqrt_nsamp)
        print()
        print('number of accepted M-H samples:', naccepted)
        print('number of rejected M-H samples:', nrejected)
        print()
        print('importance weights:', importance_weights)
        print('mean of importance weights:', np.mean(importance_weights))
        print()
        raise Exception('print tmjp entropy stuff')

