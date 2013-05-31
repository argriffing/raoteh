"""
Test functions related to the tolerance Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import random
import itertools
import functools

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

import scipy.linalg
from scipy import special
from scipy import optimize

from raoteh.sampler._sampler import (
        gen_histories, get_forward_sample)
from raoteh.sampler._sample_tree import (
        get_random_branching_tree, get_random_agglom_tree)
from raoteh.sampler._util import (
                StructuralZeroProb, NumericalZeroProb, get_first_element)
from raoteh.sampler._mjp import (
            get_history_dwell_times, get_history_root_state_and_transitions,
            get_total_rates, get_conditional_transition_matrix)
from raoteh.sampler._mc import get_restricted_likelihood
from raoteh.sampler._tmjp import get_tolerance_process_log_likelihood


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


#XXX this belongs in the Markov jump chain module
def get_expm_augmented_tree(T, Q, root):
    """
    This is a helper function.

    Parameters
    ----------
    T : weighted undirected networkx graph
        This tree may possibly be annotated with edge state restrictions.
        If an edge is annotated with the 'allowed' attribution,
        then it is interpreted as a whitelist of states that are
        allowed along the edge.  If the annotation is missing,
        then no edge state restriction is imposed.
    Q : weighted directed networkx graph
        Sparse rate matrix.
    root : integer
        Root node.

    Returns
    -------
    T_aug : weighted undirected networkx graph
        Tree annotated with transition probability matrices.
        If state restrictions along edges are in place,
        then these probabilities are joint with the condition
        that no restricted state was entered along the edge.

    """
    # Convert the sparse rate matrix to a dense ndarray rate matrix.
    states = sorted(Q)
    nstates = len(states)
    Q_dense = np.zeros((nstates, nstates), dtype=float)
    for a, sa in enumerate(states):
        for b, sb in enumerate(states):
            if Q.has_edge(sa, sb):
                edge = Q[sa][sb]
                Q_dense[a, b] = edge['weight']
    Q_dense = Q_dense - np.diag(np.sum(Q_dense, axis=1))

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        weight = edge['weight']
        if 'allowed' in edge:
            allowed_edge_states = edge['allowed']
            Q_dense_local = Q_dense.copy()
            for a, sa in enumerate(states):
                for b, sb in enumerate(states):
                    if sb not in allowed_edge_states:
                        Q_dense_local[a, b] = 0
        else:
            Q_dense_local = Q_dense
        P_dense = scipy.linalg.expm(weight * Q_dense_local)
        P_nx = nx.DiGraph()
        for a, sa in enumerate(states):
            for b, sb in enumerate(states):
                if ('allowed' not in edge) or (sb in edge['allowed']):
                    P_nx.add_edge(sa, sb, weight=P_dense[a, b])
        T_aug.add_edge(na, nb, P=P_nx)

    # Return the augmented tree.
    return T_aug


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
        T_aug = get_expm_augmented_tree(T, Q_compound, root)
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
        for compound_process_history in itertools.islice(
                gen_histories(T, Q_compound, node_to_compound_state),
                nhistories):

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


if __name__ == '__main__':
    run_module_suite()

