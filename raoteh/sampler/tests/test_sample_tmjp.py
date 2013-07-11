"""
Test functions that sample tolerance Markov jump trajectories on a tree.

In this module,
the disease_data variable is a list, indexed by tolerance class,
of maps from a node to a set of allowed tolerance states.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import random
import textwrap

import numpy as np
import networkx as nx
from scipy import special

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)


from raoteh.sampler import (
        _graph_transform,
        _mjp,
        _mjp_dense,
        _tmjp,
        _tmjp_dense,
        _tmjp_util,
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

from raoteh.sampler._tmjp import (
        get_tolerance_process_log_likelihood,
        get_primary_proposal_rate_matrix,
        get_example_tolerance_process_info,
        )


def test_get_feasible_history():

    # Define a base tree and root.
    T = nx.Graph()
    T.add_edge(0, 1, weight=2.0)
    T.add_edge(0, 2, weight=2.0)
    T.add_edge(0, 3, weight=2.0)
    root = 0

    # Define observed primary states at the leaves.
    node_to_primary_state = {1:0, 2:2, 3:4}

    # Get the compound tolerance process toy model.
    ctm = _tmjp.get_example_tolerance_process_info()

    # Get a feasible history.
    # This includes the primary trajectory and the tolerance trajectories.
    feasible_history = _sample_tmjp.get_feasible_history(
            ctm, T, root, node_to_primary_state)
    primary_trajectory, tolerance_trajectories = feasible_history

    # Assert that the number of tolerance trajectories is correct.
    assert_equal(len(tolerance_trajectories), ctm.nparts)


@decorators.slow
def test_tmjp_monte_carlo_rao_teh_differential_entropy():
    # In this test, we look at conditional expected log likelihoods.
    # These are computed in two ways.
    # The first way is by exponential integration using expm_frechet.
    # The second way is by Rao-Teh sampling.
    # Importance sampling and Metropolis-Hastings sampling are also used.

    # Define some other properties of the process,
    # in a way that is not object-oriented.
    ctm = _tmjp.get_example_tolerance_process_info()

    # Simulate some data for testing.
    info = get_simulated_data(ctm, sample_disease_data=False)
    (T, root, leaf_to_primary_state,
            node_to_allowed_primary_states, node_to_allowed_compound_states,
            reference_leaf, reference_disease_parts) = info

    # Compute the marginal likelihood of the leaf distribution
    # of the forward sample, according to the compound process.
    target_marginal_likelihood = _mjp.get_likelihood(
            T, node_to_allowed_compound_states,
            root, root_distn=ctm.compound_distn,
            Q_default=ctm.Q_compound)

    # Compute the conditional expected log likelihood explicitly
    # using some Markov jump process functions.

    # Get some posterior expectations.
    expectation_info = _mjp.get_expected_history_statistics(
            T, node_to_allowed_compound_states, root, 
            root_distn=ctm.compound_distn, Q_default=ctm.Q_compound)
    dwell_times, post_root_distn, transitions = expectation_info

    # Compute contributions to differential entropy.
    diff_ent_cnll = _tmjp.differential_entropy_helper(
            ctm, post_root_distn, dwell_times, transitions)

    # Define the number of samples.
    nsamples = 100
    sqrt_nsamp = np.sqrt(nsamples)

    # Do some Rao-Teh conditional samples,
    # and get the negative expected log likelihood.
    # The idea for the pm_ prefix is that we can use the compound
    # process Rao-Teh to sample the pure primary process without bias
    # if we throw away the tolerance process information.
    # Then with these unbiased primary process history samples,
    # we can compute conditional expectations of statistics of interest
    # by integrating over the possible tolerance trajectories.
    # This allows us to test this integration without worrying that
    # the primary process history samples are biased.

    sampled_root_distn = defaultdict(float)
    plain_cnlls = []
    pm_cnlls = []
    for T_aug in _sampler.gen_restricted_histories(
            T, ctm.Q_compound, node_to_allowed_compound_states,
            root=root, root_distn=ctm.compound_distn,
            uniformization_factor=2, nhistories=nsamples):

        # Compound process log likelihood contributions.
        info = _mjp.get_history_statistics(T_aug, root=root)
        dwell_times, root_state, transitions = info
        post_root_distn = {root_state : 1}
        plain_cnll = _tmjp.differential_entropy_helper(
                ctm, post_root_distn, dwell_times, transitions)
        plain_cnlls.append(plain_cnll)

        # Update the root distribution.
        sampled_root_distn[root_state] += 1.0 / nsamples

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
            primary_state = ctm.compound_to_primary[compound_state]
            edge['state'] = primary_state
        extras = get_redundant_degree_two_nodes(T_primary_aug) - set(T)
        T_primary_aug = remove_redundant_nodes(T_primary_aug, extras)

        pm_cnll = _tmjp.ll_expectation_helper(ctm, T_primary_aug, root)
        pm_cnlls.append(pm_cnll)

    # Define a rate matrix for a primary process proposal distribution.
    Q_proposal = get_primary_proposal_rate_matrix(
            ctm.Q_primary, ctm.primary_to_part, ctm.tolerance_distn)

    # Summarize the primary proposal rates.
    proposal_total_rates = _mjp.get_total_rates(Q_proposal)

    # Get the probability of leaf states according
    # to the proposal distribution.
    proposal_marginal_likelihood = _mjp.get_likelihood(
            T, node_to_allowed_primary_states,
            root, root_distn=ctm.primary_distn,
            Q_default=Q_proposal)

    # Get Rao-Teh samples of primary process trajectories
    # conditional on the primary states at the leaves.
    # The imp_ prefix denotes importance sampling.
    imp_sampled_root_distn = defaultdict(float)
    imp_cnlls = []
    importance_weights = []
    for T_primary_aug in _sampler.gen_histories(
            T, Q_proposal, leaf_to_primary_state,
            root=root, root_distn=ctm.primary_distn,
            uniformization_factor=2, nhistories=nsamples):

        # Compute primary process statistics.
        info = _mjp.get_history_statistics(T_primary_aug, root=root)
        dwell_times, root_state, transitions = info

        # The primary process statistics will be used for two purposes.
        # One of the purposes is as the denominator of the
        # importance sampling ratio.
        # The second purpose is to compute contributions
        # to the neg log likelihood estimate.
        post_root_distn = {root_state : 1}

        # Proposal primary process log likelihood.
        neg_ll_info = _mjp.differential_entropy_helper(
                Q_proposal, ctm.primary_distn,
                post_root_distn, dwell_times, transitions)
        proposal_ll = -sum(neg_ll_info)

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
                ctm, T_primary_aug, root)
        importance_weight = np.exp(tolerance_ll - proposal_ll)

        # Append the importance weight to the list.
        importance_weights.append(importance_weight)

        imp_cnll = _tmjp.ll_expectation_helper(ctm, T_primary_aug, root)
        imp_cnlls.append(imp_cnll)

    # Get the importance weights normalized to have average value 1.
    w = np.array(importance_weights) / np.mean(importance_weights)

    # define some expectations
    norm_imp_init = w * np.array([x.init for x in imp_cnlls])
    norm_imp_init_prim = w * np.array([x.init_prim for x in imp_cnlls])
    norm_imp_init_tol = w * np.array([x.init_tol for x in imp_cnlls])
    norm_imp_dwell = w * np.array([x.dwell for x in imp_cnlls])
    norm_imp_trans = w * np.array([x.trans for x in imp_cnlls])
    norm_imp_trans_prim = w * np.array([x.trans_prim for x in imp_cnlls])
    norm_imp_trans_tol = w * np.array([x.trans_tol for x in imp_cnlls])

    # Get Rao-Teh samples of primary process trajectories
    # conditional on the primary states at the leaves.
    # The met_ prefix denotes Metropolis-Hastings.
    met_sampled_root_distn = defaultdict(float)
    met_cnlls = []
    naccepted = 0
    nrejected = 0

    # Define the callback.
    def target_log_likelihood_callback(T_aug_in):
        return get_tolerance_process_log_likelihood(ctm, T_aug_in, root)

    # Sample the histories.
    for T_primary_aug, accept_flag in _sampler.gen_mh_histories(
            T, Q_proposal, node_to_allowed_primary_states,
            target_log_likelihood_callback,
            root, root_distn=ctm.primary_distn,
            uniformization_factor=2, nhistories=nsamples):

        if accept_flag:
            naccepted += 1
        else:
            nrejected += 1

        met_cnll = _tmjp.ll_expectation_helper(ctm, T_primary_aug, root)
        met_cnlls.append(met_cnll)

    neg_ll_contribs_init = [x.init for x in plain_cnlls]
    neg_ll_contribs_dwell = [x.dwell for x in plain_cnlls]
    neg_ll_contribs_trans = [x.trans for x in plain_cnlls]
    init_prim = [x.init_prim for x in plain_cnlls]
    init_tol = [x.init_tol for x in plain_cnlls]
    trans_prim = [x.trans_prim for x in plain_cnlls]
    trans_tol = [x.trans_tol for x in plain_cnlls]
    #
    pm_neg_ll_contribs_init = [x.init for x in pm_cnlls]
    pm_neg_ll_contribs_dwell = [x.dwell for x in pm_cnlls]
    pm_neg_ll_contribs_trans = [x.trans for x in pm_cnlls]
    pm_init_prim = [x.init_prim for x in pm_cnlls]
    pm_init_tol = [x.init_tol for x in pm_cnlls]
    pm_trans_prim = [x.trans_prim for x in pm_cnlls]
    pm_trans_tol = [x.trans_tol for x in pm_cnlls]
    #
    met_neg_ll_contribs_init = [x.init for x in met_cnlls]
    met_neg_ll_contribs_dwell = [x.dwell for x in met_cnlls]
    met_neg_ll_contribs_trans = [x.trans for x in met_cnlls]
    met_init_prim = [x.init_prim for x in met_cnlls]
    met_init_tol = [x.init_tol for x in met_cnlls]
    met_trans_prim = [x.trans_prim for x in met_cnlls]
    met_trans_tol = [x.trans_tol for x in met_cnlls]

    print()
    print('--- tmjp experiment ---')
    print('nsamples:', nsamples)
    print()
    print('sampled root distn :', sampled_root_distn)
    print('analytic root distn:', post_root_distn)
    print()
    print('--- init contribution ---')
    print('diff ent  :', val_exact(diff_ent_cnll.init))
    print('    prim  :', val_exact(diff_ent_cnll.init_prim))
    print('    tol   :', val_exact(diff_ent_cnll.init_tol))
    print('neg ll    :', val_err(neg_ll_contribs_init))
    print('    prim  :', val_err(init_prim))
    print('    tol   :', val_err(init_tol))
    print('pm neg ll :', val_err(pm_neg_ll_contribs_init))
    print('    prim  :', val_err(pm_init_prim))
    print('    tol   :', val_err(pm_init_tol))
    print('imp neg ll:', val_err(norm_imp_init))
    print('    prim  :', val_err(norm_imp_init_prim))
    print('    tol   :', val_err(norm_imp_init_tol))
    print('met neg ll:', val_err(met_neg_ll_contribs_init))
    print('    prim  :', val_err(met_init_prim))
    print('    tol   :', val_err(met_init_tol))
    print()
    print('--- dwell contribution ---')
    print('diff ent  :', val_exact(diff_ent_cnll.dwell))
    print('neg ll    :', val_err(neg_ll_contribs_dwell))
    print('pm neg ll :', val_err(pm_neg_ll_contribs_dwell))
    print('imp neg ll:', val_err(norm_imp_dwell))
    print('met neg ll:', val_err(met_neg_ll_contribs_dwell))
    print()
    print('--- trans contribution ---')
    print('diff ent  :', val_exact(diff_ent_cnll.trans))
    print('    prim  :', val_exact(diff_ent_cnll.trans_prim))
    print('    tol   :', val_exact(diff_ent_cnll.trans_tol))
    print('neg ll    :', val_err(neg_ll_contribs_trans))
    print('    prim  :', val_err(trans_prim))
    print('    tol   :', val_err(trans_tol))
    print('pm neg ll :', val_err(pm_neg_ll_contribs_trans))
    print('    prim  :', val_err(pm_trans_prim))
    print('    tol   :', val_err(pm_trans_tol))
    print('imp neg ll:', val_err(norm_imp_trans))
    print('    prim  :', val_err(norm_imp_trans_prim))
    print('    tol   :', val_err(norm_imp_trans_tol))
    print('met neg ll:', val_err(met_neg_ll_contribs_trans))
    print('    prim  :', val_err(met_trans_prim))
    print('    tol   :', val_err(met_trans_tol))
    print()
    print('number of accepted M-H samples:', naccepted)
    print('number of rejected M-H samples:', nrejected)
    print()
    print('importance weights:')
    elements = []
    for weight in importance_weights:
        s = '{:8.4f}'.format(weight).lstrip()
        elements.append(s)
    for line in textwrap.wrap(' '.join(elements), width=79):
        print(line)
    print('mean of weights:', np.mean(importance_weights))
    print('error          :', np.std(importance_weights) / sqrt_nsamp)
    print()
    print('target marginal likelihood  :', target_marginal_likelihood)
    print('proposal marginal likelihood:', proposal_marginal_likelihood)
    print('likelihood ratio            :', (target_marginal_likelihood /
        proposal_marginal_likelihood))


#TODO bit-rotted
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

        primary_info = _mjp.get_history_statistics(T_primary_aug, root=root)
        dwell_times, root_state, transitions = primary_info
        post_root_distn = {root_state : 1}

        neg_ll_info = _mjp.differential_entropy_helper(
                Q_primary, primary_distn,
                post_root_distn, dwell_times, transitions)
        neg_init_prim_ll = neg_ll_info[0]
        neg_dwell_prim_ll = neg_ll_info[1]
        neg_trans_prim_ll = neg_ll_info[2]

        tol_summary = _tmjp.get_tolerance_summary(
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

        d_neg_ll_info = _mjp_dense.differential_entropy_helper(
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


def _tmjp_clever_sample_helper_dense(ctm, T, root, leaf_to_primary_state,
        disease_data=None, nhistories=None):
    """
    A helper function for speed profiling.

    Parameters
    ----------
    ctm : instance of CompoundToleranceModel
        Compound tolerance model.
    T : x
        x
    root : x
        x
    leaf_to_primary_state : x
        x
    disease_data : sequence, optional
        For each tolerance class,
        map each node to a set of allowed tolerance states.
    nhistories : integer, optional
        Sample this many histories.

    """
    # init dense transition matrix stuff
    if set(ctm.primary_to_part) != set(range(ctm.nprimary)):
        raise NotImplementedError
    primary_states = range(ctm.nprimary)
    primary_distn_dense = dict_to_numpy_array(
            ctm.primary_distn, nodelist=primary_states)
    Q_primary_dense = rate_matrix_to_numpy_array(
            ctm.Q_primary, nodelist=primary_states)

    # Construct a map from (tuple(tolerance_states), primary_state) to a
    # compound state.
    compound_reduction_map = {}
    for compound_state in range(ctm.ncompound):
        primary_state = ctm.compound_to_primary[compound_state]
        tolerance_states = ctm.compound_to_tolerances[compound_state]
        expanded_compound_state = (tuple(tolerance_states), primary_state)
        compound_reduction_map[expanded_compound_state] = compound_state
    
    v1_cnlls = []
    d_cnlls = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp.gen_histories_v1(
            ctm, T, root, leaf_to_primary_state,
            disease_data=disease_data, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        # Reconstitute the compound process trajectory,
        # so that we can compute non-Rao-Blackwellized log likelihoods.
        all_trajectories = tol_trajectories + [T_primary_aug]
        T_merged, dummy_events = _graph_transform.add_trajectories(
                T, root, all_trajectories,
                edge_to_event_times=None)

        # Check that T_merged has the same weighted edge length as T.
        assert_allclose(
                T_merged.size(weight='weight'),
                T.size(weight='weight'))

        # Construct the compound trajectory from the merged trajectories.
        T_compound = nx.Graph()
        for na, nb in nx.bfs_edges(T_merged, root):
            edge_obj = T_merged[na][nb]
            weight = edge_obj['weight']
            primary_state = edge_obj['states'][-1]
            tol_states = edge_obj['states'][:-1]
            expanded_compound_state = (tuple(tol_states), primary_state)
            compound_state = compound_reduction_map[expanded_compound_state]

            # Check the round trip for the compound state correspondence.
            assert_equal(ctm.compound_to_primary[compound_state],
                    primary_state)
            assert_equal(ctm.compound_to_tolerances[compound_state],
                    tol_states)

            # Add the edge annotated with the compound state.
            T_compound.add_edge(na, nb, state=compound_state, weight=weight)

        # Check that T_compound has the same weighted edge length as T.
        assert_allclose(
                T_compound.size(weight='weight'),
                T.size(weight='weight'))

        info = _mjp.get_history_statistics(T_compound, root=root)
        dwell_times, root_state, transitions = info

        post_root_distn = {root_state : 1.0}
        v1_cnll = _tmjp.differential_entropy_helper(
                ctm, post_root_distn, dwell_times, transitions)
        v1_cnlls.append(v1_cnll)

        # Use the dense transition matrix Rao-Blackwellization.
        d_cnll = _tmjp_dense.ll_expectation_helper(
                ctm.primary_to_part, ctm.rate_on, ctm.rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root,
                disease_data=disease_data)
        d_cnlls.append(d_cnll)

    return v1_cnlls, d_cnlls


#TODO this function is bit-rotted
def _tmjp_clever_sample_helper(
        T, root, Q_primary, primary_to_part,
        leaf_to_primary_state, rate_on, rate_off,
        primary_distn, nhistories):
    """
    A helper function for speed profiling sparse vs. dense.

    It checks only rao-blackwellized estimates.

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
        ll_info = _tmjp.ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        neg_ll_contribs_init.append(-ll_init)
        neg_ll_contribs_dwell.append(-ll_dwell)
        neg_ll_contribs_trans.append(-ll_trans)

        # use the dense transition matrix rao-blackwellization
        ll_info = _tmjp_dense.ll_expectation_helper(
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
        ctm, T, node_to_allowed_compound_states, root,
        disease_data=None, nhistories=None):
    """
    This sampler is dumb insofar as it uses the compound process directly.

    """
    # Do some Rao-Teh conditional samples,
    # and get the negative expected log likelihood.
    #
    # The idea for the pm_ prefix is that we can use the compound
    # process Rao-Teh to sample the pure primary process without bias
    # if we throw away the tolerance process information.
    # Then with these unbiased primary process history samples,
    # we can compute conditional expectations of statistics of interest
    # by integrating over the possible tolerance trajectories.
    # This allows us to test this integration without worrying that
    # the primary process history samples are biased.

    cnlls = []
    pm_cnlls = []
    for T_aug in _sampler.gen_restricted_histories(
            T, ctm.Q_compound, node_to_allowed_compound_states,
            root=root, root_distn=ctm.compound_distn,
            uniformization_factor=2, nhistories=nhistories):

        # Check that all nodes in T are also in T_aug.
        bad = set(T) - set(T_aug)
        if bad:
            print(root)
            print(list(nx.bfs_edges(T, root)))
            print(list(nx.bfs_edges(T_aug, root)))
            raise Exception('internal error: T_aug missing %s' % bad)

        # Get some stats of the histories.
        info = _mjp.get_history_statistics(T_aug, root=root)
        dwell_times, root_state, transitions = info

        # Summarize non-Rao-Blackwellized log likelihood contributions.
        post_root_distn = {root_state : 1}
        cnll = _tmjp.differential_entropy_helper(
                ctm, post_root_distn, dwell_times, transitions)
        cnlls.append(cnll)

        # Get the primary state augmentation
        # from the compound state augmentation.
        # This uses graph transformations to detect and remove
        # degree-2 vertices whose adjacent states are identical.
        # Only nodes not present in the original tree are removed.
        T_primary_aug = nx.Graph()
        for na, nb in nx.bfs_edges(T_aug, root):
            compound_edge_obj = T_aug[na][nb]
            weight = compound_edge_obj['weight']
            compound_state = compound_edge_obj['state']
            primary_state = ctm.compound_to_primary[compound_state]
            T_primary_aug.add_edge(na, nb, weight=weight, state=primary_state)
        extras = get_redundant_degree_two_nodes(T_primary_aug) - set(T)
        T_primary_aug = remove_redundant_nodes(T_primary_aug, extras)

        # Check that all nodes in T are also in T_primary_aug.
        bad = set(T) - set(T_primary_aug)
        if bad:
            print(root)
            print(list(nx.bfs_edges(T, root)))
            print(list(nx.bfs_edges(T_primary_aug, root)))
            raise Exception('internal error: T_primary_aug missing %s' % bad)

        pm_cnll = _tmjp.ll_expectation_helper(
                ctm, T_primary_aug, root, disease_data=disease_data)
        pm_cnlls.append(pm_cnll)

    return cnlls, pm_cnlls


def get_simulated_data(ctm, sample_disease_data=False):
    """
    """
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
            T, ctm.Q_primary, root, ctm.primary_distn)

    # Get the sampled leaf states from the forward sample.
    leaf_to_primary_state = {}
    for node in T_forward_sample:
        if len(T_forward_sample[node]) == 1:
            nb = get_first_element(T_forward_sample[node])
            edge = T_forward_sample[node][nb]
            primary_state = edge['state']
            leaf_to_primary_state[node] = primary_state

    # Get the simulated disease data.
    reference_leaf = get_first_element(leaf_to_primary_state)
    reference_disease_parts = set()
    if sample_disease_data:

        # Get the primary state and its associated tolerance class
        # for the reference leaf.
        ref_primary_state = leaf_to_primary_state[reference_leaf]
        ref_part = ctm.primary_to_part[ref_primary_state]

        # Construct a tolerance state vector.
        # Require the tolerance of the observed primary state to be 1.
        # Get the sampled disease part set from the tolerance vector.
        tols = [random.choice((0, 1)) for p in range(ctm.nparts)]
        tols[ref_part] = 1
        reference_disease_parts = set(p for p, v in enumerate(tols) if not v)
        ref_tol_set = set(range(ctm.nparts)) - reference_disease_parts

    # Get the state restrictions
    # associated with the sampled leaf states.
    node_to_allowed_compound_states = {}
    node_to_allowed_primary_states = {}
    for node in T:
        if sample_disease_data and (node == reference_leaf):
            primary_state = leaf_to_primary_state[node]
            allowed_primary = {primary_state}
            allowed_compound = set()
            for comp in range(ctm.ncompound):
                prim = ctm.compound_to_primary[comp]
                # if the primary state is not the observed state then skip
                if prim != primary_state:
                    continue
                comp_tols = ctm.compound_to_tolerances[comp]
                comp_tol_set = set(p for p, v in enumerate(tols) if v)
                # the tolerance set must match the reference tolerances exactly
                if comp_tol_set != ref_tol_set:
                    continue
                allowed_compound.add(comp)
        elif node in leaf_to_primary_state:
            primary_state = leaf_to_primary_state[node]
            allowed_primary = {primary_state}
            allowed_compound = set()
            for comp in range(ctm.ncompound):
                prim = ctm.compound_to_primary[comp]
                if prim == primary_state:
                    allowed_compound.add(comp)
        else:
            allowed_primary = set(ctm.primary_distn)
            allowed_compound = set(ctm.compound_distn)
        node_to_allowed_primary_states[node] = allowed_primary
        node_to_allowed_compound_states[node] = allowed_compound

    ret = (T, root, leaf_to_primary_state,
            node_to_allowed_primary_states, node_to_allowed_compound_states,
            reference_leaf, reference_disease_parts)
    return ret


@decorators.slow
def test_sample_tmjp_v1():
    # Compare summaries of samples from the product space
    # of the compound process to summaries of samples that uses
    # gibbs sampling enabled by conditional independence
    # of some components of the compound process.

    # Define the tolerance Markov jump process.
    ctm = _tmjp.get_example_tolerance_process_info()

    # Define the number of samples per repeat.
    nsamples = 1000

    # Simulate some data for testing.
    info = get_simulated_data(ctm, sample_disease_data=False)
    (T, root, leaf_to_primary_state,
            node_to_allowed_primary_states, node_to_allowed_compound_states,
            reference_leaf, reference_disease_parts) = info

    # Compute the conditional expected log likelihood explicitly
    # using some Markov jump process functions.

    # Get some posterior expectations.
    expectation_info = _mjp.get_expected_history_statistics(
            T, node_to_allowed_compound_states,
            root, root_distn=ctm.compound_distn, Q_default=ctm.Q_compound)
    dwell_times, post_root_distn, transitions = expectation_info

    # Compute contributions to differential entropy.
    diff_ent_cnll = _tmjp.differential_entropy_helper(
            ctm, post_root_distn, dwell_times, transitions)

    # Get neg ll contribs using the clever sampler.
    # This calls a separate function for more isolated profiling.
    v1_cnlls, d_cnlls = _tmjp_clever_sample_helper_dense(
            ctm, T, root, leaf_to_primary_state, nhistories=nsamples)

    # Get neg ll contribs using the dumb sampler.
    # This calls a separate function for more isolated profiling.
    cnlls, pm_cnlls = _tmjp_dumb_sample_helper(
            ctm, T, node_to_allowed_compound_states, root,
            disease_data=None, nhistories=nsamples)

    print()
    print('--- tmjp v1 test ---')
    print()
    print('number of sampled histories:', nsamples)
    print()
    print_cnlls(diff_ent_cnll, cnlls, pm_cnlls, v1_cnlls, d_cnlls)



@decorators.slow
def test_sample_tmjp_v1_disease():
    # Compare summaries of samples from the product space
    # of the compound process to summaries of samples that uses
    # gibbs sampling enabled by conditional independence
    # of some components of the compound process.
    # Disease data is used.

    # Define some other properties of the process,
    # in a way that is not object-oriented.
    ctm = _tmjp.get_example_tolerance_process_info()

    # Define the number of samples.
    nsamples = 1000

    # Simulate some data for testing.
    info = get_simulated_data(ctm, sample_disease_data=True)
    (T, root, leaf_to_primary_state,
            node_to_allowed_primary_states, node_to_allowed_compound_states,
            reference_leaf, reference_disease_parts) = info

    # Compute the conditional expected log likelihood explicitly
    # using some Markov jump process functions.

    # Get some posterior expectations.
    expectation_info = _mjp.get_expected_history_statistics(
            T, node_to_allowed_compound_states,
            root, root_distn=ctm.compound_distn, Q_default=ctm.Q_compound)
    dwell_times, post_root_distn, transitions = expectation_info

    # Compute contributions to differential entropy.
    diff_ent_cnll = _tmjp.differential_entropy_helper(
            ctm, post_root_distn, dwell_times, transitions)

    # Construct the disease data from the reference leaf and disease parts.
    disease_data = []
    for part in range(ctm.nparts):
        tmap = dict((n, {0, 1}) for n in T)

        # The tolerance state at a leaf is known for the tolerance class
        # that corresponds to the primary state observed at the leaf.
        for node in T:
            if node in leaf_to_primary_state:
                leaf_prim = leaf_to_primary_state[node]
                leaf_part = ctm.primary_to_part[leaf_prim]
                if leaf_part == part:
                    tmap[node].intersection_update({1})

        # Tolerance classes with observed disease status
        # are not tolerated at the reference leaf.
        if part in reference_disease_parts:
            tmap[reference_leaf].intersection_update({0})
        else:
            tmap[reference_leaf].intersection_update({1})

        disease_data.append(tmap)

    # Get neg ll contribs using the dumb sampler.
    cnlls, pm_cnlls = _tmjp_dumb_sample_helper(
            ctm, T, node_to_allowed_compound_states, root,
            disease_data=disease_data, nhistories=nsamples)

    # Get neg ll contribs using the clever sampler.
    v1_cnlls, d_cnlls = _tmjp_clever_sample_helper_dense(
            ctm, T, root, leaf_to_primary_state,
            disease_data=disease_data, nhistories=nsamples)

    print()
    print('--- tmjp v1 test with simulated disease data ---')
    print()
    print('number of sampled histories:', nsamples)
    print('sampled reference leaf disease tolerance classes:')
    print(reference_disease_parts)
    print()
    print_cnlls(diff_ent_cnll, cnlls, pm_cnlls, v1_cnlls, d_cnlls)


def val_exact(v):
    # helper function for printing
    s = '{:8.5f}'.format(v)
    return s


def val_err(values):
    # helper function for printing
    sqrt_nsamples = np.sqrt(len(values))
    mu = np.mean(values)
    sigma = np.std(values)
    s = '{:8.5f} +/- {:8.5f}'.format(mu, sigma / sqrt_nsamples)
    return s


#TODO clean this up a bit
def print_cnlls(diff_ent_cnll, plain_cnlls, pm_cnlls, v1_cnlls, d_cnlls):
    neg_ll_contribs_init = [x.init for x in plain_cnlls]
    neg_ll_contribs_dwell = [x.dwell for x in plain_cnlls]
    neg_ll_contribs_trans = [x.trans for x in plain_cnlls]
    init_prim = [x.init_prim for x in plain_cnlls]
    init_tol = [x.init_tol for x in plain_cnlls]
    trans_prim = [x.trans_prim for x in plain_cnlls]
    trans_tol = [x.trans_tol for x in plain_cnlls]
    pm_neg_ll_contribs_init = [x.init for x in pm_cnlls]
    pm_neg_ll_contribs_dwell = [x.dwell for x in pm_cnlls]
    pm_neg_ll_contribs_trans = [x.trans for x in pm_cnlls]
    pm_init_prim = [x.init_prim for x in pm_cnlls]
    pm_init_tol = [x.init_tol for x in pm_cnlls]
    pm_trans_prim = [x.trans_prim for x in pm_cnlls]
    pm_trans_tol = [x.trans_tol for x in pm_cnlls]
    v1_neg_ll_contribs_init = [x.init for x in v1_cnlls]
    v1_neg_ll_contribs_dwell = [x.dwell for x in v1_cnlls]
    v1_neg_ll_contribs_trans = [x.trans for x in v1_cnlls]
    v1_init_prim = [x.init_prim for x in v1_cnlls]
    v1_init_tol = [x.init_tol for x in v1_cnlls]
    v1_trans_prim = [x.trans_prim for x in v1_cnlls]
    v1_trans_tol = [x.trans_tol for x in v1_cnlls]
    d_neg_ll_contribs_init = [x.init for x in d_cnlls]
    d_neg_ll_contribs_dwell = [x.dwell for x in d_cnlls]
    d_neg_ll_contribs_trans = [x.trans for x in d_cnlls]
    d_init_prim = [x.init_prim for x in d_cnlls]
    d_init_tol = [x.init_tol for x in d_cnlls]
    d_trans_prim = [x.trans_prim for x in d_cnlls]
    d_trans_tol = [x.trans_tol for x in d_cnlls]
    print('All samplers use Rao-Teh sampling.')
    print('plain: directly from compound distribution no rao blackwellization')
    print('pm:  directly from compound distribution with rao blackwellization')
    print('v1:  Gibbs no rao blackwellization')
    print('d:   Gibbs with rao blackwellization')
    print()
    print('--- init contribution ---')
    print()
    print('diff ent :', val_exact(diff_ent_cnll.init))
    print('    prim :', val_exact(diff_ent_cnll.init_prim))
    print('    tol  :', val_exact(diff_ent_cnll.init_tol))
    print()
    print('neg ll   :', val_err(neg_ll_contribs_init))
    print('    prim :', val_err(init_prim))
    print('    tol  :', val_err(init_tol))
    print()
    print('pm neg ll:', val_err(pm_neg_ll_contribs_init))
    print('    prim :', val_err(pm_init_prim))
    print('    tol  :', val_err(pm_init_tol))
    print()
    print('v1 neg ll:', val_err(v1_neg_ll_contribs_init))
    print('    prim :', val_err(v1_init_prim))
    print('    tol  :', val_err(v1_init_tol))
    print()
    print('d neg ll :', val_err(d_neg_ll_contribs_init))
    print('    prim :', val_err(d_init_prim))
    print('    tol  :', val_err(d_init_tol))
    print()
    print()
    print('--- dwell contribution ---')
    print()
    print('diff ent :', val_exact(diff_ent_cnll.dwell))
    print('neg ll   :', val_err(neg_ll_contribs_dwell))
    print('pm neg ll:', val_err(pm_neg_ll_contribs_dwell))
    print('v1 neg ll:', val_err(v1_neg_ll_contribs_dwell))
    print('d neg ll :', val_err(d_neg_ll_contribs_dwell))
    print()
    print()
    print('--- trans contribution ---')
    print()
    print('diff ent :', val_exact(diff_ent_cnll.trans))
    print('    prim :', val_exact(diff_ent_cnll.trans_prim))
    print('    tol  :', val_exact(diff_ent_cnll.trans_tol))
    print()
    print('neg ll   :', val_err(neg_ll_contribs_trans))
    print('    prim :', val_err(trans_prim))
    print('    tol  :', val_err(trans_tol))
    print()
    print('pm neg ll:', val_err(pm_neg_ll_contribs_trans))
    print('    prim :', val_err(pm_trans_prim))
    print('    tol  :', val_err(pm_trans_tol))
    print()
    print('v1 neg ll:', val_err(v1_neg_ll_contribs_trans))
    print('    prim :', val_err(v1_trans_prim))
    print('    tol  :', val_err(v1_trans_tol))
    print()
    print('d neg ll :', val_err(d_neg_ll_contribs_trans))
    print('    prim :', val_err(d_trans_prim))
    print('    tol  :', val_err(d_trans_tol))


if __name__ == '__main__':
    run_module_suite()

