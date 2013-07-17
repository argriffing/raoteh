"""
Test functions related to the tolerance Markov jump process.

"""
from __future__ import division, print_function, absolute_import

import random
import itertools
import functools

import numpy as np
import networkx as nx
from scipy import special
from scipy import optimize

from numpy.testing import (
        run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises, decorators,
        )

from raoteh.sampler import _util, _mjp, _tmjp, _sampler

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element,
        )

from raoteh.sampler._mjp import (
        get_history_dwell_times,
        get_history_root_state_and_transitions,
        get_total_rates,
        get_expm_augmented_tree,
        get_trajectory_log_likelihood,
        )

from raoteh.sampler._tmjp import (
        get_tolerance_process_log_likelihood,
        get_example_tolerance_process_info,
        get_tolerance_distn,
        )

from raoteh.sampler._sampler import (
        gen_histories,
        gen_restricted_histories,
        get_forward_sample,
        )

from raoteh.sampler._sample_tree import (
        get_random_agglom_tree,
        )


def test_primary_trajectory_log_likelihood():
    # Compare two ways to compute the marginal log likelihood
    # of the primary trajectory.
    # The first way directly computes the log likelihood
    # using only matrix exponentials of small 3x3 tolerance rate matrices.
    # The second way computes some expectations of statistics
    # using matrix exponentials and frechet derivatives of matrix exponentials
    # of the small 3x3 tolerance rate matrices,
    # and then computes the log likelihood through these expectations.
    # The two methods should give the same log likelihood.

    tolerance_rate_on = 0.5
    tolerance_rate_off = 1.5
    tolerance_distn = _tmjp.get_tolerance_distn(
            tolerance_rate_off, tolerance_rate_on)

    # Get an arbitrary tolerance process.
    ctm = _tmjp.get_example_tolerance_process_info(
            tolerance_rate_on=tolerance_rate_on,
            tolerance_rate_off=tolerance_rate_off)

    # Get slow and fast blinking tolerance processes.
    ctm_slow = _tmjp.get_example_tolerance_process_info(
            tolerance_rate_on=1e-5*tolerance_rate_on,
            tolerance_rate_off=1e-5*tolerance_rate_off)
    ctm_fast = _tmjp.get_example_tolerance_process_info(
            tolerance_rate_on=1e2*tolerance_rate_on,
            tolerance_rate_off=1e2*tolerance_rate_off)

    # Define an arbitrary tree with edges.
    T = nx.Graph()
    T.add_weighted_edges_from([
        (0, 1, 0.1),
        (0, 2, 0.2),
        (0, 3, 0.3),
        (3, 4, 0.4),
        (3, 5, 0.5)])
    root = 0

    # Sample a primary process trajectory
    # using forward sampling of the primary process rate matrix.
    T_primary = _sampler.get_forward_sample(
            T, ctm.Q_primary, root, ctm.primary_distn)

    # Get the primary process log likelihood,
    # under the compound tolerance model.
    ll_direct = _tmjp.get_tolerance_process_log_likelihood(
            ctm, T_primary, root)

    # Get the primary process log likelihood,
    # under the fast blinking compound tolerance model.
    ll_direct_fast = _tmjp.get_tolerance_process_log_likelihood(
            ctm_fast, T_primary, root)

    # Get the primary process log likelihood,
    # under the fast blinking compound tolerance model.
    ll_direct_slow = _tmjp.get_tolerance_process_log_likelihood(
            ctm_slow, T_primary, root)

    # Get the log likelihood through the summary.
    cnll = _tmjp.ll_expectation_helper(ctm, T_primary, root)
    ll_indirect = -(cnll.init_prim + cnll.dwell_prim + cnll.trans_prim)

    # Get the log likelihood through the summary.
    cnll = _tmjp.ll_expectation_helper(ctm_fast, T_primary, root)
    ll_indirect_fast = -(cnll.init_prim + cnll.dwell_prim + cnll.trans_prim)

    # Get the log likelihood through the summary.
    cnll = _tmjp.ll_expectation_helper(ctm_slow, T_primary, root)
    ll_indirect_slow = -(cnll.init_prim + cnll.dwell_prim + cnll.trans_prim)

    # Get the log likelihood for the fast blinking limit.
    Q_proposal = _tmjp.get_primary_proposal_rate_matrix(
            ctm.Q_primary, ctm.primary_to_part, tolerance_distn)
    ll_fast_limit = _mjp.get_trajectory_log_likelihood(
            T_primary, root, ctm.primary_distn, Q_default=Q_proposal)

    # Get the log likelihood for the slow blinking limit.
    slow_mixture = []
    slow_rate_matrices = []
    slow_distns = []
    total_blink_rate = tolerance_rate_on + tolerance_rate_off
    p_off = tolerance_rate_off / total_blink_rate
    p_on = tolerance_rate_on / total_blink_rate
    for tol_states in itertools.product((0, 1), repeat=ctm.nparts):

        if not any(tol_states):
            continue

        # Define the mixture probability.
        # Append it to the list of mixture probabilities.
        mixture_p = np.prod([(p_off, p_on)[s] for s in tol_states])
        mixture_p /= (1 - p_off ** ctm.nparts)
        #n_on = sum(1 for s in tol_states if s == 1)
        #n_off = sum(1 for s in tol_states if s == 0)
        #assert_allclose(n_on + n_off, ctm.nparts)
        #mixture_p = (p_on ** (n_on - 1)) * (p_off ** n_off)
        slow_mixture.append(mixture_p)

        # Construct a slow rate matrix.
        # Append it to the list of rate matrix mixture components.
        Q_slow = nx.DiGraph()
        for sa, sb in ctm.Q_primary.edges():
            weight = ctm.Q_primary[sa][sb]['weight']
            sa_part = ctm.primary_to_part[sa]
            sb_part = ctm.primary_to_part[sb]
            if tol_states[sa_part] and tol_states[sb_part]:
                Q_slow.add_edge(sa, sb, weight=weight)
        slow_rate_matrices.append(Q_slow)

        # Define the distribution.
        slow_weights = {}
        for sa, p in ctm.primary_distn.items():
            part = ctm.primary_to_part[sa]
            if tol_states[part]:
                slow_weights[sa] = p
        slow_distn = _util.get_normalized_dict_distn(slow_weights)
        slow_distns.append(slow_distn)

    assert_allclose(sum(slow_mixture), 1)

    slow_likelihood = 0
    for info in zip(slow_mixture, slow_rate_matrices, slow_distns):
        p, Q_slow, slow_distn = info
        ll = _mjp.get_trajectory_log_likelihood(
                T_primary, root, slow_distn, Q_default=Q_slow)
        slow_likelihood += p * np.exp(ll)
    ll_slow_limit = np.log(slow_likelihood)


    # Print some stuff.
    print('ll_direct  :', ll_direct)
    print('ll_indirect:', ll_indirect)
    print()
    print('ll_direct_fast  :', ll_direct_fast)
    print('ll_indirect_fast:', ll_indirect_fast)
    print('ll_fast_limit   :', ll_fast_limit)
    print()
    print('ll_direct_slow  :', ll_direct_slow)
    print('ll_indirect_slow:', ll_indirect_slow)
    print('ll_slow_limit   :', ll_slow_limit)
    print()

    # Check that the two methods give the same answer.
    assert_allclose(ll_indirect, ll_direct)


class TestMonteCarloLikelihoodRatio(TestCase):

    @decorators.slow
    def test_tmjp_primary_leaf_marginal_distn_sum(self):
        # Test that a set of marginal probabilities adds up to 1.

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        info = get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q_primary, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize properties of the process.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        tolerance_distn = get_tolerance_distn(rate_off, rate_on)
        ordered_primary_states = sorted(primary_distn)
        ordered_compound_states = sorted(compound_distn)

        # Use an arbitrary tree with few enough leaves for enumeration.
        T = nx.Graph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(0, 3)
        T.add_edge(3, 4)
        T.add_edge(3, 5)

        # Define the set of leaf nodes.
        # Define the root of the tree.
        root = 0
        ordered_leaves = [1, 2, 4, 5]
        leaf_set = set(ordered_leaves)
        nleaves = len(ordered_leaves)

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            scale = 0.1
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Sum over all possible combinations of leaf states.
        obs_likelihoods = []
        for ordered_leaf_states in itertools.product(
                ordered_primary_states, repeat=nleaves):

            # Define the leaf states.
            leaf_to_state = dict(zip(ordered_leaves, ordered_leaf_states))

            # Define the primary and compound state restrictions
            # at all nodes on the tree.
            node_to_allowed_compound_states = {}
            node_to_allowed_primary_states = {}
            for node in T:
                if node in leaf_set:
                    primary_state = leaf_to_state[node]
                    allowed_primary = {primary_state}
                    allowed_compound = set()
                    for c in ordered_compound_states:
                        if compound_to_primary[c] == primary_state:
                            allowed_compound.add(c)
                else:
                    allowed_primary = set(ordered_primary_states)
                    allowed_compound = set(ordered_compound_states)
                node_to_allowed_primary_states[node] = allowed_primary
                node_to_allowed_compound_states[node] = allowed_compound

            # Compute the observation likelihood under the compound model.
            obs_likelihood_target = _mjp.get_likelihood(
                    T, node_to_allowed_compound_states, root,
                    #root_distn=compound_distn_unnormal,
                    root_distn=compound_distn,
                    Q_default=Q_compound)

            # Add the observation likelihood to the list.
            obs_likelihoods.append(obs_likelihood_target)

        # Check the sum of observation likelihoods.
        # Because this is a sum of marginal leaf data probabilities
        # over all possible leaf data combinations,
        # the sum should be 1.
        assert_allclose(sum(obs_likelihoods), 1)

    @decorators.slow
    def test_monte_carlo_likelihood_ratio(self):
        # Define a tree with some branch lengths.
        # Use forward sampling from some process to get primary state
        # observations at the leaves.
        # Define a tolerance Markov jump compound process.
        # Independently define some reference primary Markov jump process.
        # For each of these two processes,
        # compute the likelihood associated with the observed
        # primary state at the leaves.

        # Define the tolerance process rates.
        rate_on = 0.5
        rate_off = 1.5

        # Define some other properties of the process,
        # in a way that is not object-oriented.
        info = get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q_primary, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize properties of the process.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        tolerance_distn = get_tolerance_distn(rate_off, rate_on)
        all_primary_states = set(primary_distn)
        all_compound_states = set(compound_distn)

        # Sample a random tree without branch lengths.
        #T = get_random_agglom_tree(maxnodes=10)
        # Use an arbitrary tree with few enough leaves for enumeration.
        T = nx.Graph()
        T.add_edge(0, 1)
        #T.add_edge(0, 2)
        #T.add_edge(0, 3)
        #T.add_edge(3, 4)
        #T.add_edge(3, 5)
        root = 0

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            scale = 1.0
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Sample a single unconditional history on the tree
        # using some arbitrary process.
        # The purpose is really to sample the states at the leaves.
        T_forward_sample = get_forward_sample(
                T, Q_primary, root, primary_distn)

        # Get the sampled leaf states from the forward sample.
        leaf_to_primary_state = {}
        for node in T_forward_sample:
            if len(T_forward_sample[node]) == 1:
                nb = get_first_element(T_forward_sample[node])
                edge = T_forward_sample[node][nb]
                primary_state = edge['state']
                leaf_to_primary_state[node] = primary_state

        # Get the state restrictions
        # associated with the sampled leaf states.
        node_to_allowed_compound_states = {}
        node_to_allowed_primary_states = {}
        for node in T:
            if node in leaf_to_primary_state:
                primary_state = leaf_to_primary_state[node]
                allowed_primary = {primary_state}
                allowed_compound = set()
                for c in all_compound_states:
                    if compound_to_primary[c] == primary_state:
                        allowed_compound.add(c)
            else:
                allowed_primary = all_primary_states
                allowed_compound = all_compound_states
            node_to_allowed_primary_states[node] = allowed_primary
            node_to_allowed_compound_states[node] = allowed_compound

        # Compute the observation likelihood under the compound model.
        obs_likelihood_target = _mjp.get_likelihood(
                T, node_to_allowed_compound_states, root,
                root_distn=compound_distn,
                Q_default=Q_compound)

        # Compute the observation likelihood under the primary model.
        obs_likelihood_proposal = _mjp.get_likelihood(
                T, node_to_allowed_primary_states, root,
                root_distn=primary_distn,
                Q_default=Q_primary)

        # Sample primary state trajectories
        # according to the pure primary process.
        nhistories = 1000
        importance_weights = []
        for T_primary_traj in gen_restricted_histories(
                T, Q_primary, node_to_allowed_primary_states,
                root, root_distn=primary_distn, nhistories=nhistories):
            
            # Compute the trajectory log likelihood
            # under the inhomogeneous primary component of the
            # tolerance Markov jump process.
            traj_ll_target = get_tolerance_process_log_likelihood(
                    Q_primary, primary_to_part, T_primary_traj,
                    rate_off, rate_on, primary_distn, root)

            # Compute the trajectory log likelihood
            # under the distribution used for sampling the histories.
            traj_ll_proposal = _mjp.get_trajectory_log_likelihood(
                    T_primary_traj, root, primary_distn,
                    Q_default=Q_primary)

            # Calculate the importance weight as a trajectory likelihood ratio.
            traj_likelihood_ratio = np.exp(traj_ll_target - traj_ll_proposal)

            # Append the importance weight to the list.
            importance_weights.append(traj_likelihood_ratio)

        # Report the likelihood ratios.
        print()
        print('--- monte carlo likelihood test ---')
        print('sample tree:')
        for na, nb in nx.bfs_edges(T, root):
            print(na, nb, T[na][nb]['weight'])
        print('leaf primary state samples:')
        for n, s in sorted(leaf_to_primary_state.items()):
            print(n, s)
        print('obs likelihood target  :', obs_likelihood_target)
        print('obs likelihood proposal:', obs_likelihood_proposal)
        print('obs likelihood ratio   :',
                obs_likelihood_target / obs_likelihood_proposal)
        print('sample average of importance weights:',
                np.mean(importance_weights))
        print('sample average error:',
                np.std(importance_weights) / np.sqrt(nhistories))
        print()



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
    info = get_example_tolerance_process_info(rate_on, rate_off)
    (primary_distn, Q, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound) = info

    # Return the negative log likelihood for minimization.
    log_likelihood = 0.0
    for T, node_to_allowed_states in zip(T_seq, node_to_allowed_states_seq):

        # Compute some matrix exponentials
        # and put them on the edges of the tree,
        # respecting edge state restrictions.
        T_aug = get_expm_augmented_tree(T, root, Q_default=Q_compound)
        likelihood = _mcy.get_likelihood(T_aug, root,
                node_to_allowed_states=node_to_allowed_states,
                root_distn=compound_distn)
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
        info = get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize properties of the process.
        ncompound = len(compound_to_primary)
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        tolerance_distn = get_tolerance_distn(rate_off, rate_on)

        # Sample a non-tiny random tree without branch lengths.
        T = get_random_agglom_tree(maxnodes=20)
        root = 0

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
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

    @decorators.slow
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
        info = get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize the other properties.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        total_tolerance_rate = rate_on + rate_off
        tolerance_distn = get_tolerance_distn(rate_off, rate_on)

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
        info = get_example_tolerance_process_info(rate_on, rate_off)
        (primary_distn, Q, primary_to_part,
                compound_to_primary, compound_to_tolerances, compound_distn,
                Q_compound) = info

        # Summarize properties of the process.
        nprimary = len(primary_distn)
        nparts = len(set(primary_to_part.values()))
        tolerance_distn = get_tolerance_distn(rate_off, rate_on)

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
                Q, state_to_part, T,
                rate_off, rate_on, primary_distn, root)

