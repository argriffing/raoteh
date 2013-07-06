"""

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import special

import create_mg94
import app_helper

from raoteh.sampler import (
        _tmjp_dense,
        _sample_tmjp,
        _mjp_dense,
        _density,
        )


#TODO add disease data conditioning
#TODO copypasted from tests/test_sample_tmjp.py
#TODO this should go into the _tmjp_dense module.
def _compound_ll_expectation_helper_dense(
        primary_to_part, rate_on, rate_off,
        Q_primary, primary_distn, T_primary_aug, root,
        disease_data=None):
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
    disease_data : x
        x

    Returns
    -------
    init_ll : float
        x
    dwell_ll : float
        x
    trans_ll : float
        x

    """
    _density.check_square_dense(Q_primary)
    nprimary = Q_primary.shape[0]
    if primary_distn.shape[0] != nprimary:
        raise ValueError('inconsistency in the number of primary states')

    total_tree_length = T_primary_aug.size(weight='weight')
    primary_info = _mjp_dense.get_history_statistics(
            T_primary_aug, nprimary, root=root)
    dwell_times, root_state, transitions = primary_info
    post_root_distn = np.zeros(nprimary, dtype=float)
    post_root_distn[root_state] = 1

    neg_ll_info = _mjp_dense.differential_entropy_helper(
            Q_primary, primary_distn,
            post_root_distn, dwell_times, transitions)
    neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

    tol_summary = _tmjp_dense.get_tolerance_summary(
            primary_to_part, rate_on, rate_off,
            Q_primary, T_primary_aug, root,
            disease_data=disease_data)

    tol_info = _tmjp_dense.get_tolerance_ll_contribs(
            rate_on, rate_off, total_tree_length, *tol_summary)
    init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

    init_ll = -neg_init_prim_ll + init_tol_ll
    dwell_ll = dwell_tol_ll
    trans_ll = -neg_trans_prim_ll + trans_tol_ll

    return init_ll, dwell_ll, trans_ll


#TODO copypasted from tests/test_sample_tmjp.py
#TODO some modification for disease data
def _tmjp_clever_sample_helper_dense(
        T, root, Q_primary, primary_to_part,
        leaf_to_primary_state, rate_on, rate_off, primary_distn,
        disease_data=None, nhistories=None):
    """
    The args are the same as for _sample_tmjp.gen_histories_v1().
    """
    # init dense transition matrix stuff
    nprimary = len(primary_to_part)
    if set(primary_to_part) != set(range(nprimary)):
        raise NotImplementedError
    primary_states = range(nprimary)
    primary_distn_dense = _density.dict_to_numpy_array(
            primary_distn, nodelist=primary_states)
    Q_primary_dense = _density.rate_matrix_to_numpy_array(
            Q_primary, nodelist=primary_states)

    # init dense transition matrix process summary lists
    d_neg_ll_contribs_init = []
    d_neg_ll_contribs_dwell = []
    d_neg_ll_contribs_trans = []

    # sample histories and summarize them using rao-blackwellization
    for history_info in _sample_tmjp.gen_histories_v1(
            T, root, Q_primary, primary_to_part,
            leaf_to_primary_state, rate_on, rate_off, primary_distn,
            disease_data=disease_data, nhistories=nhistories):
        T_primary_aug, tol_trajectories = history_info

        # use the dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper_dense(
                primary_to_part, rate_on, rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root,
                disease_data=disease_data)
        ll_init, ll_dwell, ll_trans = ll_info
        d_neg_ll_contribs_init.append(-ll_init)
        d_neg_ll_contribs_dwell.append(-ll_dwell)
        d_neg_ll_contribs_trans.append(-ll_trans)

    d_neg_ll_contribs = (
            d_neg_ll_contribs_init,
            d_neg_ll_contribs_dwell,
            d_neg_ll_contribs_trans)
    
    return d_neg_ll_contribs


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
    nparts = len(set(primary_to_part.values()))
    nprimary = len(primary_to_part)
    if nprimary != 61:
        print(primary_to_part)
        raise Exception(
                'expected 61 non-stop codons '
                'but found ' + str(nprimary))
    
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, primary_distn)
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
    #TODO use disease constraints
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    total_log_likelihood = 0
    primary_distn_dense = _density.dict_to_numpy_array(
            primary_distn, nodelist=states)
    Q_dense = _density.rate_matrix_to_numpy_array(
            Q_primary, nodelist=states)
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
            for part in range(nparts):
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
        nhistories = 10
        info = _tmjp_clever_sample_helper_dense(
                T, root, Q_primary, primary_to_part,
                leaf_to_primary_state, rate_on, rate_off, primary_distn,
                disease_data=disease_data, nhistories=nhistories)
        d_neg_ll_contribs_init = info[0]
        d_neg_ll_contribs_dwell = info[1]
        d_neg_ll_contribs_trans = info[2]
        neg_log_likelihoods = [sum(contribs) for contribs in zip(*info)]
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

