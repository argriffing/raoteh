"""
Test functions that sample tolerance Markov jump trajectories on a tree.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx
from scipy import special

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)


from raoteh.sampler import (
        _mjp,
        _mjp_dense,
        _tmjp,
        _tmjp_dense,
        _sampler,
        _sample_tmjp,
        _sample_tree,
        )

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._density import (
        dict_to_numpy_array,
        rate_matrix_to_numpy_array,
        check_square_dense,
        )

from raoteh.sampler._graph_transform import(
        get_redundant_degree_two_nodes,
        remove_redundant_nodes,
        )

from raoteh.sampler._mjp import (
        get_history_statistics,
        get_expected_history_statistics,
        )

from raoteh.sampler._tmjp import (
        get_tolerance_process_log_likelihood,
        get_tolerance_summary,
        get_primary_proposal_rate_matrix,
        get_example_tolerance_process_info,
        )



#TODO this should go into the _tmjp module.
def _compound_ll_expectation_helper(
        primary_to_part, rate_on, rate_off,
        Q_primary, primary_distn, T_primary_aug, root):
    """
    Get contributions to the expected log likelihood of the compound process.

    The primary process trajectory is fully observed,
    but the binary tolerance states are unobserved.

    Parameters
    ----------
    primary_to_part : x
        x
    rate_on : x
        x
    rate_off : x
        x
    Q_primary : x
        x
    primary_distn : x
        x
    T_primary_aug : x
        x
    root : integer
        The root node.

    Returns
    -------
    init_ll : float
        x
    dwell_ll : float
        x
    trans_ll : float
        x

    """
    total_tree_length = T_primary_aug.size(weight='weight')
    primary_info = get_history_statistics(T_primary_aug, root=root)
    dwell_times, root_state, transitions = primary_info
    post_root_distn = {root_state : 1}

    neg_ll_info = _differential_entropy_helper_sparse(
            Q_primary, primary_distn,
            post_root_distn, dwell_times, transitions)
    neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

    # This function call is the speed bottleneck for this function.
    tol_summary = get_tolerance_summary(
            primary_to_part, rate_on, rate_off,
            Q_primary, T_primary_aug, root)

    tol_info = _tmjp.get_tolerance_ll_contribs(
            rate_on, rate_off, total_tree_length, *tol_summary)
    init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

    init_ll = -neg_init_prim_ll + init_tol_ll
    dwell_ll = dwell_tol_ll
    trans_ll = -neg_trans_prim_ll + trans_tol_ll

    return init_ll, dwell_ll, trans_ll


#TODO this should go into the _tmjp_dense module.
def _compound_ll_expectation_helper_dense(
        primary_to_part, rate_on, rate_off,
        Q_primary, primary_distn, T_primary_aug, root):
    """
    Get contributions to the expected log likelihood of the compound process.

    The primary process trajectory is fully observed,
    but the binary tolerance states are unobserved.

    Parameters
    ----------
    primary_to_part : x
        x
    rate_on : x
        x
    rate_off : x
        x
    Q_primary : 2d ndarray
        Primary rate matrix.
    primary_distn : 1d ndarray
        Primary process state distribution.
    T_primary_aug : x
        x
    root : integer
        The root node.

    Returns
    -------
    init_ll : float
        x
    dwell_ll : float
        x
    trans_ll : float
        x

    """
    check_square_dense(Q_primary)
    nprimary = Q_primary.shape[0]
    if primary_distn.shape[0] != nprimary:
        raise ValueError('inconsistency in the number of primary states')

    total_tree_length = T_primary_aug.size(weight='weight')
    primary_info = _mjp_dense.get_history_statistics(
            T_primary_aug, nprimary, root=root)
    dwell_times, root_state, transitions = primary_info
    post_root_distn = np.zeros(nprimary, dtype=float)
    post_root_distn[root_state] = 1

    neg_ll_info = _differential_entropy_helper_dense(
            Q_primary, primary_distn,
            post_root_distn, dwell_times, transitions)
    neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

    tol_summary = _tmjp_dense.get_tolerance_summary(
            primary_to_part, rate_on, rate_off,
            Q_primary, T_primary_aug, root)

    tol_info = _tmjp_dense.get_tolerance_ll_contribs(
            rate_on, rate_off, total_tree_length, *tol_summary)
    init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

    init_ll = -neg_init_prim_ll + init_tol_ll
    dwell_ll = dwell_tol_ll
    trans_ll = -neg_trans_prim_ll + trans_tol_ll

    return init_ll, dwell_ll, trans_ll


#TODO this should be just a log likelihood helper function in the _mjp module.
def _differential_entropy_helper_sparse(
        Q, prior_root_distn,
        post_root_distn, post_dwell_times, post_transitions,
        ):
    """
    Use posterior expectations to help compute differential entropy.

    Parameters
    ----------
    Q : weighted directed networkx graph
        Rate matrix.
    prior_root_distn : dict
        Prior distribution at the root.
        If Q is a time-reversible rate matrix,
        then the prior root distribution
        could be the stationary distribution associated with Q.
    post_root_distn : dict
        Posterior state distribution at the root.
    post_dwell_times : dict
        Posterior expected dwell time for each state.
    post_transitions : weighted directed networkx graph
        Posterior expected count of each transition type.

    Returns
    -------
    diff_ent_init : float
        Initial state distribution contribution to differential entropy.
    diff_ent_dwell : float
        Dwell time contribution to differential entropy.
    diff_ent_trans : float
        Transition contribution to differential entropy.

    """
    # Get the total rates.
    total_rates = _mjp.get_total_rates(Q)

    # Initial state distribution contribution to differential entropy.
    diff_ent_init = 0.0
    for state, prob in post_root_distn.items():
        diff_ent_init -= special.xlogy(prob, prior_root_distn[state])

    # Dwell time contribution to differential entropy.
    diff_ent_dwell = 0.0
    for s in set(total_rates) & set(post_dwell_times):
        diff_ent_dwell += post_dwell_times[s] * total_rates[s]

    # Transition contribution to differential entropy.
    diff_ent_trans = 0.0
    for sa in set(Q) & set(post_transitions):
        for sb in set(Q[sa]) & set(post_transitions[sa]):
            rate = Q[sa][sb]['weight']
            ntrans_expected = post_transitions[sa][sb]['weight']
            diff_ent_trans -= special.xlogy(ntrans_expected, rate)

    # Return the contributions to differential entropy.
    return diff_ent_init, diff_ent_dwell, diff_ent_trans


#TODO this should be a log likelihood helper function in the _mjp_dense module.
def _differential_entropy_helper_dense(
        Q, prior_root_distn,
        post_root_distn, post_dwell_times, post_transitions,
        ):
    """
    Use posterior expectations to help compute differential entropy.

    Parameters
    ----------
    Q : 2d ndarray
        Rate matrix.
    prior_root_distn : 1d ndarray
        Prior distribution at the root.
        If Q is a time-reversible rate matrix,
        then the prior root distribution
        could be the stationary distribution associated with Q.
    post_root_distn : 1d ndarray
        Posterior state distribution at the root.
    post_dwell_times : 1d ndarray
        Posterior expected dwell time for each state.
    post_transitions : 2d ndarray
        Posterior expected count of each transition type.

    Returns
    -------
    diff_ent_init : float
        Initial state distribution contribution to differential entropy.
    diff_ent_dwell : float
        Dwell time contribution to differential entropy.
    diff_ent_trans : float
        Transition contribution to differential entropy.

    """
    check_square_dense(Q)
    check_square_dense(post_transitions)
    nstates = Q.shape[0]

    # Get the total rates.
    total_rates = _mjp_dense.get_total_rates(Q)

    # Initial state distribution contribution to differential entropy.
    diff_ent_init = -special.xlogy(post_root_distn, prior_root_distn).sum()

    # Dwell time contribution to differential entropy.
    diff_ent_dwell = post_dwell_times.dot(total_rates)

    # Transition contribution to differential entropy.
    diff_ent_trans = -special.xlogy(post_transitions, Q).sum()

    # Return the contributions to differential entropy.
    return diff_ent_init, diff_ent_dwell, diff_ent_trans


def test_get_feasible_history():

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


@decorators.slow
def test_tmjp_monte_carlo_rao_teh_differential_entropy():
    # In this test, we look at conditional expected log likelihoods.
    # These are computed in two ways.
    # The first way is by exponential integration using expm_frechet.
    # The second way is by Rao-Teh sampling.

    # Define the tolerance process rates.
    rate_on = 0.5
    rate_off = 1.5

    # Define some other properties of the process,
    # in a way that is not object-oriented.
    info = _tmjp.get_example_tolerance_process_info(rate_on, rate_off)
    (primary_distn, Q_primary, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound) = info

    # Summarize properties of the process.
    nprimary = len(primary_distn)
    nparts = len(set(primary_to_part.values()))
    compound_total_rates = _mjp.get_total_rates(Q_compound)
    primary_total_rates = _mjp.get_total_rates(Q_primary)
    tolerance_distn = _tmjp.get_tolerance_distn(rate_off, rate_on)

    # Sample a non-tiny random tree without branch lengths.
    T = _sample_tree.get_random_agglom_tree(maxnodes=5)
    root = 0

    # Add some random branch lengths onto the edges of the tree.
    for na, nb in nx.bfs_edges(T, root):
        scale = 2.6
        T[na][nb]['weight'] = np.random.exponential(scale=scale)

    # Sample a single unconditional history on the tree
    # using some arbitrary process.
    # The purpose is really to sample the states at the leaves.
    T_forward_sample = _sampler.get_forward_sample(
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
            for comp, prim in enumerate(compound_to_primary):
                if prim == primary_state:
                    allowed_compound.add(comp)
        else:
            allowed_primary = set(primary_distn)
            allowed_compound = set(compound_distn)
        node_to_allowed_primary_states[node] = allowed_primary
        node_to_allowed_compound_states[node] = allowed_compound

    # Compute the marginal likelihood of the leaf distribution
    # of the forward sample, according to the compound process.
    target_marginal_likelihood = _mjp.get_likelihood(
            T, node_to_allowed_compound_states,
            root, root_distn=compound_distn,
            Q_default=Q_compound)

    # Compute the conditional expected log likelihood explicitly
    # using some Markov jump process functions.

    # Get some posterior expectations.
    expectation_info = get_expected_history_statistics(
            T, node_to_allowed_compound_states,
            root, root_distn=compound_distn, Q_default=Q_compound)
    dwell_times, post_root_distn, transitions = expectation_info

    # Compute contributions to differential entropy.
    diff_ent_info = _differential_entropy_helper_sparse(
        Q_compound, compound_distn,
        post_root_distn, dwell_times, transitions)
    diff_ent_init, diff_ent_dwell, diff_ent_trans = diff_ent_info

    # Define the number of samples.
    nsamples = 100
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
    pm_neg_ll_contribs_dwell = []
    pm_neg_ll_contribs_init = []
    pm_neg_ll_contribs_trans = []
    #
    for T_aug in _sampler.gen_restricted_histories(
            T, Q_compound, node_to_allowed_compound_states,
            root=root, root_distn=compound_distn,
            uniformization_factor=2, nhistories=nsamples):

        # Get some stats of the histories.
        info = get_history_statistics(T_aug, root=root)
        dwell_times, root_state, transitions = info

        sampled_root_distn[root_state] += 1.0 / nsamples

        # Compound process log likelihoods contributions.
        post_root_distn = {root_state : 1}
        neg_ll_info = _differential_entropy_helper_sparse(
                Q_compound, compound_distn,
                post_root_distn, dwell_times, transitions)
        neg_ll_init, neg_ll_dwell, neg_ll_trans = neg_ll_info
        neg_ll_contribs_init.append(neg_ll_init)
        neg_ll_contribs_dwell.append(neg_ll_dwell)
        neg_ll_contribs_trans.append(neg_ll_trans)

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

        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        pm_neg_ll_contribs_init.append(-ll_init)
        pm_neg_ll_contribs_dwell.append(-ll_dwell)
        pm_neg_ll_contribs_trans.append(-ll_trans)

    # Define a rate matrix for a primary process proposal distribution.
    Q_proposal = get_primary_proposal_rate_matrix(
            Q_primary, primary_to_part, tolerance_distn)

    # Summarize the primary proposal rates.
    proposal_total_rates = _mjp.get_total_rates(Q_proposal)

    # Get the probability of leaf states according
    # to the proposal distribution.
    proposal_marginal_likelihood = _mjp.get_likelihood(
            T, node_to_allowed_primary_states,
            root, root_distn=primary_distn,
            Q_default=Q_proposal)

    # Get Rao-Teh samples of primary process trajectories
    # conditional on the primary states at the leaves.
    # The imp_ prefix denotes importance sampling.
    imp_sampled_root_distn = defaultdict(float)
    imp_neg_ll_contribs_init = []
    imp_neg_ll_contribs_dwell = []
    imp_neg_ll_contribs_trans = []
    importance_weights = []
    for T_primary_aug in _sampler.gen_histories(
            T, Q_proposal, leaf_to_primary_state,
            root=root, root_distn=primary_distn,
            uniformization_factor=2, nhistories=nsamples):

        # Compute primary process statistics.
        # These will be used for two purposes.
        # One of the purposes is as the denominator of the
        # importance sampling ratio.
        # The second purpose is to compute contributions
        # to the neg log likelihood estimate.
        info = get_history_statistics(T_primary_aug, root=root)
        dwell_times, root_state, transitions = info


        # Proposal primary process summary.

        # contribution of root state to log likelihood
        proposal_init_ll = np.log(primary_distn[root_state])

        # contribution of dwell times
        ll = 0.0
        for state, dwell in dwell_times.items():
            ll -= dwell * proposal_total_rates[state]
        proposal_dwell_ll = ll

        # contribution of transitions
        ll = 0.0
        for sa, sb in transitions.edges():
            ntransitions = transitions[sa][sb]['weight']
            rate = Q_proposal[sa][sb]['weight']
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

        # Get the primary log likelihood
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
        tolerance_ll = get_tolerance_process_log_likelihood(
                Q_primary, primary_to_part, T_primary_aug,
                rate_off, rate_on, primary_distn, root)
        importance_weight = np.exp(tolerance_ll - proposal_ll)

        # Append the importance weight to the list.
        importance_weights.append(importance_weight)

        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        imp_neg_ll_contribs_init.append(-ll_init)
        imp_neg_ll_contribs_dwell.append(-ll_dwell)
        imp_neg_ll_contribs_trans.append(-ll_trans)

    # define some expectations
    normalized_imp_trans = (np.array(importance_weights) * np.array(
            imp_neg_ll_contribs_trans)) / np.mean(importance_weights)
    normalized_imp_init = (np.array(importance_weights) * np.array(
            imp_neg_ll_contribs_init)) / np.mean(importance_weights)
    normalized_imp_dwell = (np.array(importance_weights) * np.array(
            imp_neg_ll_contribs_dwell)) / np.mean(importance_weights)


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
        return get_tolerance_process_log_likelihood(
                Q_primary, primary_to_part, T_aug,
                rate_off, rate_on, primary_distn, root)

    # Sample the histories.
    for T_primary_aug, accept_flag in _sampler.gen_mh_histories(
            T, Q_proposal, node_to_allowed_primary_states,
            target_log_likelihood_callback,
            root, root_distn=primary_distn,
            uniformization_factor=2, nhistories=nsamples):

        if accept_flag:
            naccepted += 1
        else:
            nrejected += 1

        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        met_neg_ll_contribs_init.append(-ll_init)
        met_neg_ll_contribs_dwell.append(-ll_dwell)
        met_neg_ll_contribs_trans.append(-ll_trans)

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
    print('pm neg ll dwel:', np.mean(pm_neg_ll_contribs_dwell))
    print('error         :', np.std(pm_neg_ll_contribs_dwell) / sqrt_nsamp)
    print('imp dwell     :', np.mean(normalized_imp_dwell))
    print('error         :', np.std(normalized_imp_dwell) / sqrt_nsamp)
    print('met dwell     :', np.mean(met_neg_ll_contribs_dwell))
    print('error         :', np.std(met_neg_ll_contribs_dwell) / sqrt_nsamp)
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
    print('mean of weights:', np.mean(importance_weights))
    print('error          :', np.std(importance_weights) / sqrt_nsamp)
    print()
    print('target marginal likelihood  :', target_marginal_likelihood)
    print('proposal marginal likelihood:', proposal_marginal_likelihood)
    print('likelihood ratio            :', (target_marginal_likelihood /
        proposal_marginal_likelihood))


#TODO this is just for debugging a dense vs. non-dense
#TODO rao-blackwellization discrepancy
def _tmjp_clever_sample_helper_debug(
        T, root, Q_primary, primary_to_part,
        leaf_to_primary_state, rate_on, rate_off,
        primary_distn, nhistories):
    """
    A helper function for speed profiling.

    The args are the same as for _sample_tmjp.gen_histories_v1().

    """
    # init dense transition matrix stuff
    nprimary = len(primary_to_part)
    if set(primary_to_part) != set(range(nprimary)):
        raise NotImplementedError
    primary_states = range(nprimary)
    primary_distn_dense = dict_to_numpy_array(
            primary_distn, nodelist=primary_states)
    Q_primary_dense = rate_matrix_to_numpy_array(
            Q_primary, nodelist=primary_states)

    # init non-dense transition matrix process summary lists
    neg_ll_contribs_init = []
    neg_ll_contribs_dwell = []
    neg_ll_contribs_trans = []

    # init dense transition matrix process summary lists
    d_neg_ll_contribs_init = []
    d_neg_ll_contribs_dwell = []
    d_neg_ll_contribs_trans = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp.gen_histories_v1(
            T, root, Q_primary, primary_to_part,
            leaf_to_primary_state, rate_on, rate_off,
            primary_distn, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        total_tree_length = T_primary_aug.size(weight='weight')

        # non-dense log likelihood contribution

        primary_info = get_history_statistics(T_primary_aug, root=root)
        dwell_times, root_state, transitions = primary_info
        post_root_distn = {root_state : 1}

        neg_ll_info = _differential_entropy_helper_sparse(
                Q_primary, primary_distn,
                post_root_distn, dwell_times, transitions)
        neg_init_prim_ll = neg_ll_info[0]
        neg_dwell_prim_ll = neg_ll_info[1]
        neg_trans_prim_ll = neg_ll_info[2]

        tol_summary = get_tolerance_summary(
                primary_to_part, rate_on, rate_off,
                Q_primary, T_primary_aug, root)

        tol_info = _tmjp.get_tolerance_ll_contribs(
                rate_on, rate_off, total_tree_length, *tol_summary)
        init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

        init_ll = -neg_init_prim_ll + init_tol_ll
        dwell_ll = dwell_tol_ll
        trans_ll = -neg_trans_prim_ll + trans_tol_ll

        neg_ll_contribs_init.append(-init_ll)
        neg_ll_contribs_dwell.append(-dwell_ll)
        neg_ll_contribs_trans.append(-trans_ll)

        # dense log likelihood contribution

        d_primary_info = _mjp_dense.get_history_statistics(
                T_primary_aug, nprimary, root=root)
        d_dwell_times, d_root_state, d_transitions = d_primary_info
        d_post_root_distn = np.zeros(nprimary, dtype=float)
        d_post_root_distn[d_root_state] = 1

        d_neg_ll_info = _differential_entropy_helper_dense(
                Q_primary_dense, primary_distn_dense,
                d_post_root_distn, d_dwell_times, d_transitions)
        d_neg_init_prim_ll = d_neg_ll_info[0]
        d_neg_dwell_prim_ll = d_neg_ll_info[1]
        d_neg_trans_prim_ll = d_neg_ll_info[2]

        tol_summary = _tmjp_dense.get_tolerance_summary(
                primary_to_part, rate_on, rate_off,
                Q_primary_dense, T_primary_aug, root)

        d_tol_info = _tmjp_dense.get_tolerance_ll_contribs(
                rate_on, rate_off, total_tree_length, *tol_summary)
        d_init_tol_ll, d_dwell_tol_ll, d_trans_tol_ll = d_tol_info

        d_init_ll = -d_neg_init_prim_ll + d_init_tol_ll
        d_dwell_ll = d_dwell_tol_ll
        d_trans_ll = -d_neg_trans_prim_ll + d_trans_tol_ll

        d_neg_ll_contribs_init.append(-d_init_ll)
        d_neg_ll_contribs_dwell.append(-d_dwell_ll)
        d_neg_ll_contribs_trans.append(-d_trans_ll)

        # check the dense vs. non-dense primary process expectations
        assert_allclose(
                d_dwell_times,
                dict_to_numpy_array(dwell_times, primary_states))
        assert_equal(d_root_state, root_state)
        assert_allclose(
                d_transitions,
                nx.to_numpy_matrix(transitions, primary_states).A)
        assert_allclose(
                d_post_root_distn,
                dict_to_numpy_array(post_root_distn, primary_states))

        # check the dense vs. non-dense primary process log likeilhoods
        assert_allclose(d_neg_ll_info, neg_ll_info)

        # check the dense vs. non-dense log likelihood contributions
        assert_allclose(
                (d_init_tol_ll, d_dwell_tol_ll, d_trans_tol_ll),
                (init_tol_ll, dwell_tol_ll, trans_tol_ll))

        # check the dense vs. non-dense entries to be added to the lists
        assert_allclose(
                (d_init_ll, d_dwell_ll, d_trans_ll),
                (init_ll, dwell_ll, trans_ll))

    neg_ll_contribs = (
            neg_ll_contribs_init,
            neg_ll_contribs_dwell,
            neg_ll_contribs_trans)

    d_neg_ll_contribs = (
            d_neg_ll_contribs_init,
            d_neg_ll_contribs_dwell,
            d_neg_ll_contribs_trans)
    
    return neg_ll_contribs, d_neg_ll_contribs


def _tmjp_clever_sample_helper_dense(
        T, root, Q_primary, primary_to_part,
        leaf_to_primary_state, rate_on, rate_off,
        primary_distn, nhistories):
    """
    A helper function for speed profiling.

    The args are the same as for _sample_tmjp.gen_histories_v1().

    """
    # init dense transition matrix stuff
    nprimary = len(primary_to_part)
    if set(primary_to_part) != set(range(nprimary)):
        raise NotImplementedError
    primary_states = range(nprimary)
    primary_distn_dense = dict_to_numpy_array(
            primary_distn, nodelist=primary_states)
    Q_primary_dense = rate_matrix_to_numpy_array(
            Q_primary, nodelist=primary_states)

    # init dense transition matrix process summary lists
    d_neg_ll_contribs_init = []
    d_neg_ll_contribs_dwell = []
    d_neg_ll_contribs_trans = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp.gen_histories_v1(
            T, root, Q_primary, primary_to_part,
            leaf_to_primary_state, rate_on, rate_off,
            primary_distn, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        # use the dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper_dense(
                primary_to_part, rate_on, rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        d_neg_ll_contribs_init.append(-ll_init)
        d_neg_ll_contribs_dwell.append(-ll_dwell)
        d_neg_ll_contribs_trans.append(-ll_trans)

    d_neg_ll_contribs = (
            d_neg_ll_contribs_init,
            d_neg_ll_contribs_dwell,
            d_neg_ll_contribs_trans)
    
    return d_neg_ll_contribs


def _tmjp_clever_sample_helper(
        T, root, Q_primary, primary_to_part,
        leaf_to_primary_state, rate_on, rate_off,
        primary_distn, nhistories):
    """
    A helper function for speed profiling.

    The args are the same as for _sample_tmjp.gen_histories_v1().

    """
    # init dense transition matrix stuff
    nprimary = len(primary_to_part)
    if set(primary_to_part) != set(range(nprimary)):
        raise NotImplementedError
    primary_states = range(nprimary)
    primary_distn_dense = dict_to_numpy_array(
            primary_distn, nodelist=primary_states)
    Q_primary_dense = rate_matrix_to_numpy_array(
            Q_primary, nodelist=primary_states)

    # init non-dense transition matrix process summary lists
    neg_ll_contribs_init = []
    neg_ll_contribs_dwell = []
    neg_ll_contribs_trans = []

    # init dense transition matrix process summary lists
    d_neg_ll_contribs_init = []
    d_neg_ll_contribs_dwell = []
    d_neg_ll_contribs_trans = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp.gen_histories_v1(
            T, root, Q_primary, primary_to_part,
            leaf_to_primary_state, rate_on, rate_off,
            primary_distn, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        # use the non-dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        neg_ll_contribs_init.append(-ll_init)
        neg_ll_contribs_dwell.append(-ll_dwell)
        neg_ll_contribs_trans.append(-ll_trans)

        # use the dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper_dense(
                primary_to_part, rate_on, rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        d_neg_ll_contribs_init.append(-ll_init)
        d_neg_ll_contribs_dwell.append(-ll_dwell)
        d_neg_ll_contribs_trans.append(-ll_trans)

    neg_ll_contribs = (
            neg_ll_contribs_init,
            neg_ll_contribs_dwell,
            neg_ll_contribs_trans)

    d_neg_ll_contribs = (
            d_neg_ll_contribs_init,
            d_neg_ll_contribs_dwell,
            d_neg_ll_contribs_trans)
    
    return neg_ll_contribs, d_neg_ll_contribs


def _tmjp_dumb_sample_helper(
        T, primary_to_part, compound_to_primary,
        Q_compound, compound_distn,
        Q_primary, primary_distn,
        node_to_allowed_compound_states,
        root, rate_on, rate_off,
        nsamples):
    """
    """
    # Precompute some stuff.
    compound_total_rates = _mjp.get_total_rates(Q_compound)

    # Do some Rao-Teh conditional samples,
    # and get the negative expected log likelihood.
    #
    # statistics for the full process samples
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
    pm_neg_ll_contribs_init = []
    pm_neg_ll_contribs_dwell = []
    pm_neg_ll_contribs_trans = []
    #
    for T_aug in _sampler.gen_restricted_histories(
            T, Q_compound, node_to_allowed_compound_states,
            root=root, root_distn=compound_distn,
            uniformization_factor=2, nhistories=nsamples):

        # Get some stats of the histories.
        info = get_history_statistics(T_aug, root=root)
        dwell_times, root_state, transitions = info

        # log likelihood contribution of initial state
        ll = np.log(compound_distn[root_state])
        neg_ll_contribs_init.append(-ll)

        # log likelihood contribution of dwell times
        ll = 0.0
        for state, dwell in dwell_times.items():
            ll -= dwell * compound_total_rates[state]
        neg_ll_contribs_dwell.append(-ll)

        # log likelihood contribution of transitions
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

        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        init_ll, dwell_ll, trans_ll = ll_info
        pm_neg_ll_contribs_init.append(-init_ll)
        pm_neg_ll_contribs_dwell.append(-dwell_ll)
        pm_neg_ll_contribs_trans.append(-trans_ll)
    neg_ll_info = (
            neg_ll_contribs_init,
            neg_ll_contribs_dwell,
            neg_ll_contribs_trans)
    pm_neg_ll_info = (
            pm_neg_ll_contribs_init,
            pm_neg_ll_contribs_dwell,
            pm_neg_ll_contribs_trans)
    return neg_ll_info, pm_neg_ll_info


@decorators.slow
def test_sample_tmjp_v1():
    # Compare summaries of samples from the product space
    # of the compound process to summaries of samples that uses
    # gibbs sampling enabled by conditional independence
    # of some components of the compound process.

    # Define the tolerance process rates.
    rate_on = 0.5
    rate_off = 1.5

    # Define some other properties of the process,
    # in a way that is not object-oriented.
    info = _tmjp.get_example_tolerance_process_info(rate_on, rate_off)
    (primary_distn, Q_primary, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound) = info

    # Summarize properties of the process.
    nprimary = len(primary_distn)
    nparts = len(set(primary_to_part.values()))
    compound_total_rates = _mjp.get_total_rates(Q_compound)
    primary_total_rates = _mjp.get_total_rates(Q_primary)
    tolerance_distn = _tmjp.get_tolerance_distn(rate_off, rate_on)

    # Define the number of samples per repeat.
    nsamples = 1000
    sqrt_nsamp = np.sqrt(nsamples)

    nrepeats = 1
    for repeat_index in range(nrepeats):

        # Sample a non-tiny random tree without branch lengths.
        maxnodes = 5
        T = _sample_tree.get_random_agglom_tree(maxnodes=maxnodes)
        root = 0

        # Check for the requested number of nodes.
        nnodes = len(T)
        assert_equal(nnodes, maxnodes)

        # Add some random branch lengths onto the edges of the tree.
        for na, nb in nx.bfs_edges(T, root):
            scale = 0.6
            T[na][nb]['weight'] = np.random.exponential(scale=scale)

        # Get the total tree length.
        total_tree_length = T.size(weight='weight')

        # Sample a single unconditional history on the tree
        # using some arbitrary process.
        # The purpose is really to sample the states at the leaves.
        T_forward_sample = _sampler.get_forward_sample(
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
                for comp, prim in enumerate(compound_to_primary):
                    if prim == primary_state:
                        allowed_compound.add(comp)
            else:
                allowed_primary = set(primary_distn)
                allowed_compound = set(compound_distn)
            node_to_allowed_primary_states[node] = allowed_primary
            node_to_allowed_compound_states[node] = allowed_compound

        # Compute the conditional expected log likelihood explicitly
        # using some Markov jump process functions.

        # Get some posterior expectations.
        expectation_info = get_expected_history_statistics(
                T, node_to_allowed_compound_states,
                root, root_distn=compound_distn, Q_default=Q_compound)
        dwell_times, post_root_distn, transitions = expectation_info

        # Compute contributions to differential entropy.
        diff_ent_info = _differential_entropy_helper_sparse(
            Q_compound, compound_distn,
            post_root_distn, dwell_times, transitions)
        diff_ent_init, diff_ent_dwell, diff_ent_trans = diff_ent_info

        # Get neg ll contribs using the clever sampler.
        # This calls a separate function for more isolated profiling.
        d_info = _tmjp_clever_sample_helper_dense(
                T, root, Q_primary, primary_to_part,
                leaf_to_primary_state, rate_on, rate_off,
                primary_distn, nsamples)
        #v1_neg_ll_contribs_init = v1_info[0]
        #v1_neg_ll_contribs_dwell = v1_info[1]
        #v1_neg_ll_contribs_trans = v1_info[2]
        d_neg_ll_contribs_init = d_info[0]
        d_neg_ll_contribs_dwell = d_info[1]
        d_neg_ll_contribs_trans = d_info[2]

        # Get neg ll contribs using the dumb sampler.
        # This calls a separate function for more isolated profiling.
        neg_ll_info, pm_neg_ll_info = _tmjp_dumb_sample_helper(
                T, primary_to_part, compound_to_primary,
                Q_compound, compound_distn,
                Q_primary, primary_distn,
                node_to_allowed_compound_states,
                root, rate_on, rate_off,
                nsamples)
        neg_ll_contribs_init = neg_ll_info[0]
        neg_ll_contribs_dwell = neg_ll_info[1]
        neg_ll_contribs_trans = neg_ll_info[2]
        pm_neg_ll_contribs_init = pm_neg_ll_info[0]
        pm_neg_ll_contribs_dwell = pm_neg_ll_info[1]
        pm_neg_ll_contribs_trans = pm_neg_ll_info[2]

    print()
    print('--- tmjp v1 test ---')
    print('nsamples:', nsamples)
    print()
    print('diff ent init :', diff_ent_init)
    print('neg ll init   :', np.mean(neg_ll_contribs_init))
    print('error         :', np.std(neg_ll_contribs_init) / sqrt_nsamp)
    print('pm neg ll init:', np.mean(pm_neg_ll_contribs_init))
    print('error         :', np.std(pm_neg_ll_contribs_init) / sqrt_nsamp)
    #print('v1 neg ll init:', np.mean(v1_neg_ll_contribs_init))
    #print('error         :', np.std(v1_neg_ll_contribs_init) / sqrt_nsamp)
    print('d neg ll init :', np.mean(d_neg_ll_contribs_init))
    print('error         :', np.std(d_neg_ll_contribs_init) / sqrt_nsamp)
    print()
    print('diff ent dwell:', diff_ent_dwell)
    print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
    print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamp)
    print('pm neg ll dwel:', np.mean(pm_neg_ll_contribs_dwell))
    print('error         :', np.std(pm_neg_ll_contribs_dwell) / sqrt_nsamp)
    #print('v1 neg ll dwel:', np.mean(v1_neg_ll_contribs_dwell))
    #print('error         :', np.std(v1_neg_ll_contribs_dwell) / sqrt_nsamp)
    print('d neg ll dwell:', np.mean(d_neg_ll_contribs_dwell))
    print('error         :', np.std(d_neg_ll_contribs_dwell) / sqrt_nsamp)
    print()
    print('diff ent trans:', diff_ent_trans)
    print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
    print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamp)
    print('pm neg ll tran:', np.mean(pm_neg_ll_contribs_trans))
    print('error         :', np.std(pm_neg_ll_contribs_trans) / sqrt_nsamp)
    #print('v1 neg ll tran:', np.mean(v1_neg_ll_contribs_trans))
    #print('error         :', np.std(v1_neg_ll_contribs_trans) / sqrt_nsamp)
    print('d neg ll trans:', np.mean(d_neg_ll_contribs_trans))
    print('error         :', np.std(d_neg_ll_contribs_trans) / sqrt_nsamp)



if __name__ == '__main__':
    run_module_suite()

