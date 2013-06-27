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
        _tmjp,
        _sampler,
        _sample_tmjp,
        _sample_tree,
        )

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

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
        get_tolerance_expectations,
        get_primary_proposal_rate_matrix,
        get_example_tolerance_process_info,
        )


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
    # Initial state distribution contribution to differential entropy.
    diff_ent_init = 0.0
    for state, prob in post_root_distn.items():
        diff_ent_init -= special.xlogy(prob, prior_root_distn[state])

    # Dwell time contribution to differential entropy.
    diff_ent_dwell = 0.0
    for s, rate in compound_total_rates.items():
        diff_ent_dwell += post_dwell_times[s] * rate

    # Transition contribution to differential entropy.
    diff_ent_trans = 0.0
    for sa in set(Q) & set(transitions):
        for sb in set(Q[sa]) & set(post_transitions[sa]):
            rate = Q[sa][sb]['weight']
            ntrans_expected = post_transitions[sa][sb]['weight']
            diff_ent_trans -= special.xlogy(ntrans_expected, rate)

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

    # Use some posterior expectations
    # to get the root distribution contribution to differential entropy.
    diff_ent_init = 0.0
    for state, prob in post_root_distn.items():
        diff_ent_init -= special.xlogy(prob, compound_distn[state])

    # Use some posterior expectations
    # to get the dwell time contribution to differential entropy.
    diff_ent_dwell = 0.0
    for s, rate in compound_total_rates.items():
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

        # contribution of root state to log likelihood
        sampled_root_distn[root_state] += 1.0 / nsamples
        ll = np.log(compound_distn[root_state])
        neg_ll_contribs_init.append(-ll)

        # contribution of dwell times
        ll = 0.0
        for state, dwell in dwell_times.items():
            ll -= dwell * compound_total_rates[state]
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

        # Append the pm_ neg ll dwell component of log likelihood.
        pm_neg_ll_contribs_dwell.append(-dwell_tol_ll)

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
    for T_aug in _sampler.gen_histories(
            T, Q_proposal, leaf_to_primary_state,
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
                Q_primary, primary_to_part, T_aug,
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

        # Add the log likelihood contributions
        # of the dwell times.
        imp_neg_ll_contribs_dwell.append(-dwell_tol_ll)

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
    for T_aug, accept_flag in _sampler.gen_mh_histories(
            T, Q_proposal, node_to_allowed_primary_states,
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
        met_neg_ll_contribs_dwell.append(-dwell_tol_ll)


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


#TODO for profiling, break this into two parts.
#TODO the first part would be the explicit expectations
#TODO the second part would be the expectations estimated from rao teh sampling
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

        # Use some posterior expectations
        # to get the root distribution contribution to differential entropy.
        diff_ent_init = 0.0
        for state, prob in post_root_distn.items():
            diff_ent_init -= special.xlogy(prob, compound_distn[state])

        # Use some posterior expectations
        # to get the dwell time contribution to differential entropy.
        diff_ent_dwell = 0.0
        for s, rate in compound_total_rates.items():
            diff_ent_dwell += dwell_times[s] * rate

        # Use some posterior expectations
        # to get the transition contribution to differential entropy.
        diff_ent_trans = 0.0
        for sa in set(Q_compound) & set(transitions):
            for sb in set(Q_compound[sa]) & set(transitions[sa]):
                rate = Q_compound[sa][sb]['weight']
                ntrans_expected = transitions[sa][sb]['weight']
                diff_ent_trans -= ntrans_expected * np.log(rate)

        # Initialize some statistics lists that will be analogous
        # to the pm_ lists.
        v1_neg_ll_contribs_dwell = []
        v1_neg_ll_contribs_init = []
        v1_neg_ll_contribs_trans = []
        for history_info in _sample_tmjp.gen_histories_v1(
                T, root, Q_primary, primary_to_part,
                leaf_to_primary_state, rate_on, rate_off,
                primary_distn, nhistories=nsamples):

            # Unpack the sampled trajectories.
            prim_trajectory, tol_trajectories = history_info

            # Get primary trajectory stats.
            primary_info = get_history_statistics(
                    prim_trajectory, root=root)
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
                    Q_primary, prim_trajectory, root)
            dwell_tol_ll, init_tol_ll, trans_tol_ll = tol_info

            # Append the pm_ neg ll trans component of log likelihood.
            v1_trans_ll = trans_prim_ll + trans_tol_ll
            v1_neg_ll_contribs_trans.append(-v1_trans_ll)

            # Append the pm_ neg ll init component of log likelihood.
            v1_init_ll = init_prim_ll + init_tol_ll
            v1_neg_ll_contribs_init.append(-v1_init_ll)

            # Append the pm_ neg ll dwell component of log likelihood.
            v1_neg_ll_contribs_dwell.append(-dwell_tol_ll)

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

            # Update the sampled root distribution.
            sampled_root_distn[root_state] += 1.0 / nsamples

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

            # Append the pm_ neg ll dwell component of log likelihood.
            pm_neg_ll_contribs_dwell.append(-dwell_tol_ll)

    print()
    print('--- tmjp v1 test ---')
    print('nsamples:', nsamples)
    print()
    print('diff ent init :', diff_ent_init)
    print('neg ll init   :', np.mean(neg_ll_contribs_init))
    print('error         :', np.std(neg_ll_contribs_init) / sqrt_nsamp)
    print('pm neg ll init:', np.mean(pm_neg_ll_contribs_init))
    print('error         :', np.std(pm_neg_ll_contribs_init) / sqrt_nsamp)
    print('v1 neg ll init:', np.mean(v1_neg_ll_contribs_init))
    print('error         :', np.std(v1_neg_ll_contribs_init) / sqrt_nsamp)
    print()
    print('diff ent dwell:', diff_ent_dwell)
    print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
    print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamp)
    print('pm neg ll dwel:', np.mean(pm_neg_ll_contribs_dwell))
    print('error         :', np.std(pm_neg_ll_contribs_dwell) / sqrt_nsamp)
    print('v1 neg ll dwel:', np.mean(v1_neg_ll_contribs_dwell))
    print('error         :', np.std(v1_neg_ll_contribs_dwell) / sqrt_nsamp)
    print()
    print('diff ent trans:', diff_ent_trans)
    print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
    print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamp)
    print('pm neg ll tran:', np.mean(pm_neg_ll_contribs_trans))
    print('error         :', np.std(pm_neg_ll_contribs_trans) / sqrt_nsamp)
    print('v1 neg ll tran:', np.mean(v1_neg_ll_contribs_trans))
    print('error         :', np.std(v1_neg_ll_contribs_trans) / sqrt_nsamp)



if __name__ == '__main__':
    run_module_suite()

