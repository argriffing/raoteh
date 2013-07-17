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
        _tmjp_util,
        _mjp_dense,
        _tmjp,
        _tmjp_dense,
        _sample_tmjp,
        _sample_tmjp_dense,
        )



#TODO copypasted and modified
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

    cnlls = []
    tolerance_summaries = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp_dense.gen_histories_v1(
            ctm, T, root, leaf_to_primary_state,
            disease_data=disease_data, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        # Summarize primary process.
        primary_summary = _mjp_dense.get_history_statistics(
                T_primary_aug, ctm.nprimary, root=root)

        # Summarize tolerance process.
        tolerance_summary = _tmjp_dense.get_tolerance_summary(
                ctm.primary_to_part, ctm.rate_on, ctm.rate_off,
                ctm.Q_primary, T_primary_aug, root,
                disease_data=disease_data)

        # Get the total tree length.
        total_tree_length = T_primary_aug.size(weight='weight')

        # Expand primary expectations.
        prim_dwell_times, prim_root_state, prim_transitions = primary_summary

        # Convert the root state to a root distribution.
        post_root_distn = np.zeros(ctm.nprimary, dtype=float)
        post_root_distn[prim_root_state] = 1

        # Expand tolerance expectations.
        (
                expected_initial_on, expected_initial_off,
                expected_dwell_on, expected_dwell_off,
                expected_nabsorptions,
                expected_ngains, expected_nlosses) = tolerance_summary

        # Record the tolerance summary.
        tolerance_summaries.append(tolerance_summary)

        # Convert expectations into log likelihood contributions.

        tol_info = _tmjp_dense.get_tolerance_ll_contribs(
                ctm.rate_on, ctm.rate_off,
                total_tree_length, *tolerance_summary)
        init_tol_ll, dwell_prim_ll, dwell_tol_ll, trans_tol_ll = tol_info

        neg_ll_info = _mjp_dense.differential_entropy_helper(
                ctm.Q_primary, ctm.primary_distn,
                post_root_distn, prim_dwell_times, prim_transitions)
        neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

        cnll = _tmjp_util.CompoundNegLL(
                neg_init_prim_ll, -init_tol_ll,
                -dwell_prim_ll, -dwell_tol_ll,
                neg_trans_prim_ll, -trans_tol_ll)

        cnlls.append(cnll)

    return cnlls, tolerance_summaries


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

    print('primary (codon) process simulation values:')
    print('kappa:', kappa_mle)
    print('omega:', omega)
    print('P(T):', T_mle)
    print('P(C):', C_mle)
    print('P(A):', A_mle)
    print('P(G):', G_mle)
    print()

    print('blinking process simulation values:')
    print('proportion tolerance on:', proportion_on)
    print('proportion tolerance off:', proportion_off)
    print('blink rate on:', rate_on)
    print('blink rate off:', rate_off)
    print()

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
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    root = name_to_leaf['Has']

    # print a summary of the tree
    degree = T.degree()
    print('number of nodes:', len(T))
    print('number of leaves:', degree.values().count(1))
    print('number of branches:', T.size())
    print('total branch length:', T.size(weight='weight'))
    print('rooted at the human taxon')
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
    nhistories = 10
    col_tol_summary_means = []
    print('number of sampled histories per column:', nhistories)
    print('sample averages per column...')
    print()
    for i, codon_column in enumerate(codon_columns):
        
        #if i > 9:
            #break

        # Get the column-specific human disease residues.
        # Convert the human disease data structure
        # into a tolerance class state allowance data structure.
        # If data does not exist for a column,
        # then do *not* treat it as missing --
        # instead, treat it as an observation that every
        # amino acid is tolerated in humans.
        disease_data = None
        if i in column_to_disease_residues:
            human_disease_residues = column_to_disease_residues[i]
        else:
            human_disease_residues = set()

        # Convert human disease data to a more general tolerance data format.
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

        leaf_to_primary_state = {}
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            state = codon_to_state[codon]
            leaf_to_primary_state[leaf] = state

        # Use conditional Rao-Teh with Gibbs modification to sample
        # from the distribution of fully augmented trajectories.
        # Use Rao-Blackwellization.
        rb_cnlls, col_tol_summaries = _tmjp_clever_sample_helper_dense(
                ctm_dense, T, root, leaf_to_primary_state,
                disease_data=disease_data, nhistories=nhistories)

        col_tol_summaries_array = np.array(col_tol_summaries, dtype=float)
        col_tol_summary_mean = col_tol_summaries_array.mean(axis=0)
        col_tol_summary_means.append(col_tol_summary_mean)

        contribs = [(x.init, x.dwell, x.trans) for x in rb_cnlls]

        neg_log_likelihoods = [sum(x) for x in contribs]

        mean_log_likelihood = -np.mean(neg_log_likelihoods)

        # Add to the total log likelihood.
        total_log_likelihood += mean_log_likelihood

        # Unpack tolerance summary.
        (
                expected_initial_on, expected_initial_off,
                expected_dwell_on, expected_dwell_off,
                expected_nabsorptions,
                expected_ngains, expected_nlosses) = col_tol_summary_mean

        # Report the column-specific mean log likelihood contribution.
        print('column', i + 1, 'of', len(codon_columns))
        print('ll of full augmentation:', mean_log_likelihood)
        print('initial on:', expected_initial_on)
        print('initial off:', expected_initial_off)
        print('dwell on:', expected_dwell_on)
        print('dwell off:', expected_dwell_off)
        print('untolerated mutations:', expected_nabsorptions)
        print('tolerance gains:', expected_ngains)
        print('tolerance losses:', expected_nlosses)
        print()


    col_tol_grand_summary = np.array(col_tol_summary_means).mean(axis=0)
    (
            expected_initial_on, expected_initial_off,
            expected_dwell_on, expected_dwell_off,
            expected_nabsorptions,
            expected_ngains, expected_nlosses) = col_tol_grand_summary
    print()
    print('estimate of total log likelihood:', total_log_likelihood)
    print()
    print('sample averages over all histories over all columns:')
    print('initial on:', expected_initial_on)
    print('initial off:', expected_initial_off)
    print('dwell on:', expected_dwell_on)
    print('dwell off:', expected_dwell_off)
    print('untolerated mutations:', expected_nabsorptions)
    print('tolerance gains:', expected_ngains)
    print('tolerance losses:', expected_nlosses)
    print()


if __name__ == '__main__':
    main()

