"""

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import special

import create_mg94
import app_helper

from raoteh.sampler import (
        _density,
        _graph_transform,
        _mjp_dense,
        _tmjp,
        _tmjp_dense,
        _sample_tmjp,
        _sample_tmjp_dense,
        )


def _tmjp_clever_sample_helper_dense(
        ctm, T, root, leaf_to_primary_state,
        disease_data=None, nhistories=None):
    """
    A helper function for speed profiling.

    Parameters
    ----------
    ctm : instance of _tmjp_dense.CompoundToleranceModel
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

    """
    # Construct a map from (tuple(tolerance_states), primary_state) to a
    # compound state.
    compound_reduction_map = {}
    for compound_state in range(ctm.ncompound):
        primary_state = ctm.compound_to_primary[compound_state]
        tolerance_states = ctm.compound_to_tolerances[compound_state]
        expanded_compound_state = (tuple(tolerance_states), primary_state)
        compound_reduction_map[expanded_compound_state] = compound_state
    """
    
    v1_cnlls = []
    d_cnlls = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp_dense.gen_histories_v1(
            ctm, T, root, leaf_to_primary_state,
            disease_data=disease_data, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        """
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

        # Summarize the compound process.
        info = _mjp_dense.get_history_statistics(
                T_compound, ctm.ncompound, root=root)
        dwell_times, root_state, transitions = info

        # Apply Rao-Blackwellization to the compound process summary.
        post_root_distn = np.zeros(ctm.ncompound, dtype=float)
        post_root_distn[root_state] = 1.0
        v1_cnll = _tmjp_dense.differential_entropy_helper(
                ctm, post_root_distn, dwell_times, transitions)
        v1_cnlls.append(v1_cnll)
        """

        # TODO change this call so that it mostly passes only ctm
        # Apply Rao-Blackwellization to the primary process trajectory.
        d_cnll = _tmjp_dense.ll_expectation_helper(
                ctm.primary_to_part, ctm.rate_on, ctm.rate_off,
                ctm.Q_primary, ctm.primary_distn,
                T_primary_aug, root,
                disease_data=disease_data)
        d_cnlls.append(d_cnll)

    return v1_cnlls, d_cnlls



#TODO copypasted from tests/test_sample_tmjp.py
def xxx_tmjp_clever_sample_helper_dense(ctm, T, root, leaf_to_primary_state,
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
    primary_distn_dense = _density.dict_to_numpy_array(
            ctm.primary_distn, nodelist=primary_states)
    Q_primary_dense = _density.rate_matrix_to_numpy_array(
            ctm.Q_primary, nodelist=primary_states)

    """
    # Construct a map from (tuple(tolerance_states), primary_state) to a
    # compound state.
    compound_reduction_map = {}
    for compound_state in range(ctm.ncompound):
        primary_state = ctm.compound_to_primary[compound_state]
        tolerance_states = ctm.compound_to_tolerances[compound_state]
        expanded_compound_state = (tuple(tolerance_states), primary_state)
        compound_reduction_map[expanded_compound_state] = compound_state
    """
    
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
        if not np.allclose(
                T_merged.size(weight='weight'),
                T.size(weight='weight')):
            raise Exception('internal error')

        """
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
        """

        # Use the dense transition matrix Rao-Blackwellization.
        d_cnll = _tmjp_dense.ll_expectation_helper(
                ctm.primary_to_part, ctm.rate_on, ctm.rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root,
                disease_data=disease_data)
        d_cnlls.append(d_cnll)

    return v1_cnlls, d_cnlls


def main():

    # values estimated using codeml
    kappa_mle = 3.17632
    #omega_mle = 0.21925
    omega = 1.0
    T_mle = 0.18883
    C_mle = 0.30126
    A_mle = 0.25039
    G_mle = 0.25952

    # Use blink-on and blink-off rates that approximate omega_mle.
    proportion_on = 0.21925
    proportion_off = 1 - proportion_on
    total_blink_rate = 1.0
    rate_on = total_blink_rate * proportion_on
    rate_off = total_blink_rate * proportion_off

    # read the disease data
    print('reading the disease data...')
    with open('p53RRRR.disease') as fin:
        column_to_disease_residues = app_helper.read_disease_data(fin)

    # read the genetic code
    print('reading the genetic code...')
    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # check that the states are in the right order
    nstates = len(genetic_code)
    if range(nstates) != [s for s, r, c in genetic_code]:
        raise Exception
    states = range(nstates)

    # define the primary rate matrix and distribution and tolerance classes
    info = create_mg94.create_mg94(
            A_mle, C_mle, G_mle, T_mle,
            kappa_mle, omega, genetic_code,
            target_expected_rate=1.0)
    Q_primary, primary_distn, state_to_residue, residue_to_part = info
    primary_to_part = dict(
            (i, residue_to_part[r]) for i, r in state_to_residue.items())

    # Construct the compound tolerance model.
    # Do not initialize attributes that require O(N) where N is
    # the size of the state space of the compound process.
    ctm = _tmjp.CompoundToleranceModel(
            Q_primary, primary_distn, primary_to_part,
            rate_on, rate_off)

    if ctm.nprimary != 61:
        print(ctm.primary_to_part)
        raise Exception(
                'expected 61 non-stop codons '
                'but found ' + str(ctm.nprimary))

    # init dense transition matrix stuff
    nprimary = len(ctm.primary_to_part)
    if set(ctm.primary_to_part) != set(range(ctm.nprimary)):
        raise NotImplementedError
    primary_states = range(ctm.nprimary)
    primary_distn_dense = _density.dict_to_numpy_array(
            ctm.primary_distn, nodelist=primary_states)
    Q_primary_dense = _density.rate_matrix_to_numpy_array(
            ctm.Q_primary, nodelist=primary_states)

    # Get the dense ctm.
    ctm_dense = _tmjp_dense.CompoundToleranceModel(
            Q_primary_dense, primary_distn_dense, ctm.primary_to_part,
            ctm.rate_on, ctm.rate_off)
    
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, ctm.primary_distn)
    print()
    
    # read the tree with branch lengths estimated by paml
    print('reading the newick tree...')
    with open('codeml.estimated.tree') as fin:
        T, root, leaf_name_pairs = app_helper.read_newick(fin)

    # print a summary of the tree
    degree = T.degree()
    print('number of nodes:', len(T))
    print('number of leaves:', degree.values().count(1))
    print('number of branches:', T.size())
    print('total branch length:', T.size(weight='weight'))
    print()

    # read the alignment
    print('reading the alignment...')
    with open('testseq') as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # Estimate expectations, column by column.
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    total_log_likelihood = 0
    print('computing expected log likelihoods per column...')
    for i, codon_column in enumerate(codon_columns):

        # Get the column-specific human disease residues.
        # Convert the human disease data structure
        # into a tolerance class state allowance data structure.
        disease_data = None
        if i in column_to_disease_residues:
            human_disease_residues = column_to_disease_residues[i]
            human_disease_parts = set(
                    residue_to_part[r] for r in human_disease_residues)
            human_node = name_to_leaf['Has']
            disease_data = []
            for part in range(ctm.nparts):
                tmap = dict((n, {0, 1}) for n in T)
                if part in human_disease_parts:
                    tmap[human_node] = {0}
                else:
                    tmap[human_node] = {1}
                disease_data.append(tmap)

        # Define the primary process (codon) allowed state sets
        # using the codon alignment data for the codon site of interest.
        #node_to_allowed_states = dict((node, set(states)) for node in T)
        #for name, codon in zip(names, codon_column):
            #leaf = name_to_leaf[name]
            #codon = codon.upper()
            #state = codon_to_state[codon]
            #node_to_allowed_states[leaf] = set([state])

        leaf_to_primary_state = {}
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            state = codon_to_state[codon]
            leaf_to_primary_state[leaf] = state

        # Use conditional Rao-Teh with Gibbs modification to sample
        # from the distribution of fully augmented trajectories.
        # Use Rao-Blackwellization.
        nhistories = 10
        v1_cnlls, rb_cnlls = _tmjp_clever_sample_helper_dense(
                ctm_dense, T, root, leaf_to_primary_state,
                disease_data=disease_data, nhistories=nhistories)

        contribs = [(x.init, x.dwell, x.trans) for x in rb_cnlls]

        neg_log_likelihoods = [sum(x) for x in contribs]

        mean_log_likelihood = -np.mean(neg_log_likelihoods)

        # Add to the total log likelihood.
        total_log_likelihood += mean_log_likelihood

        # Report the column-specific mean log likelihood contribution.
        print(
                'column', i + 1, 'of', len(codon_columns),
                'll', mean_log_likelihood)
    
    # print the total log likelihood
    print('total mean log likelihood:', total_log_likelihood)


if __name__ == '__main__':
    main()

