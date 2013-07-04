"""
Test functions that sample tolerance Markov jump trajectories on a tree.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import special

import create_mg94
import app_helper

from raoteh.sampler import (
        _mjp_dense,
        _density,
        )


def main():

    # values estimated using PAML
    kappa_mle = 3.17632
    omega_mle = 0.21925
    T_mle = 0.18883
    C_mle = 0.30126
    A_mle = 0.25039
    G_mle = 0.25952

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
            kappa_mle, omega_mle, genetic_code,
            target_expected_rate=1.0)
    Q, primary_distn, state_to_residue, residue_to_part = info
    primary_to_part = dict(
            (i, residue_to_part[r]) for i, r in state_to_residue.items())
    
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

    # compute the log likelihood, column by column
    # using _mjp_dense (the dense Markov jump process module).
    print('preparing to compute log likelihood...')
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    total_log_likelihood = 0
    primary_distn_dense = _density.dict_to_numpy_array(
            primary_distn, nodelist=states)
    Q_dense = _density.rate_matrix_to_numpy_array(
            Q, nodelist=states)
    print('computing log likelihood...')
    for i, codon_column in enumerate(codon_columns):
        node_to_allowed_states = dict((node, set(states)) for node in T)
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            state = codon_to_state[codon]
            node_to_allowed_states[leaf] = set([state])
        likelihood = _mjp_dense.get_likelihood(
                T, node_to_allowed_states, root, nstates,
                root_distn=primary_distn_dense, Q_default=Q_dense)
        log_likelihood = np.log(likelihood)
        total_log_likelihood += log_likelihood
        print('column', i + 1, 'of', len(codon_columns), 'll', log_likelihood)
    
    # print the total log likelihood
    print('total log likelihood:', total_log_likelihood)


if __name__ == '__main__':
    main()

