"""
Attempt to reproduce log likelihood of the background/reference model.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import networkx as nx
import numpy as np
from scipy import special

import create_mg94
import app_helper

from raoteh.sampler import (
        _util,
        _density,
        _mjp_dense,
        )


def main():

    # FINAL LIKELIHOOD IS -10279.111666
    # FINAL ESTIMATE: rho12 =    0.32771
    # FINAL ESTIMATE: for frequency of purines is    0.50781
    # FINAL ESTIMATE: for freq. of A among purines is    0.49542
    # FINAL ESTIMATE: for freq. of T among pyrimidines is    0.39194
    # FINAL ESTIMATE: kappa =    3.38998
    # FINAL ESTIMATE: omega =    0.40198

    # Jeff has estimated these parameters
    # using the background/reference switching process.
    kappa = 3.38998
    omega = 0.40198
    AG = 0.50781
    CT = 1 - AG
    A = AG * 0.49542
    G = AG - A
    T = CT * 0.39194
    C = CT - T
    rho = 0.32771

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
    Q, primary_distn, primary_to_part = create_mg94.create_mg94(
            A, C, G, T,
            kappa, omega, genetic_code,
            target_expected_rate=1.0)
    
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, primary_distn)
    print()
    
    # read the tree with branch lengths estimated by paml
    print('reading the newick tree...')
    with open('liwen.estimated.tree') as fin:
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
    print()

    # read the alignment
    print('reading the alignment...')
    with open('testseq') as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # compute the log likelihood, column by column
    # using _mjp_dense (the dense Markov jump process module).
    print('preparing to compute log likelihood...')
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    print('computing log likelihood...')
    total_log_likelihood = 0
    for i, codon_column in enumerate(codon_columns):

        # define the column-specific disease states and the benign states
        disease_residues = column_to_disease_residues.get(codon_column, set())
        disease_states = set(
                s for s, r, c in genetic_code if r in disease_residues)
        benign_states = set(range(nstates)) - disease_states

        # Define the compound process state space.
        ncompound = 2 * nstates
        compound_states = range(ncompound)

        # Initialize the column-specific compound rate matrix.
        Q_compound = nx.DiGraph()

        # Add block-diagonal entries of the background process.
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            Q_compound.add_edge(nstates + sa, nstates + sb, weight=weight)

        # Add block-diagonal entries of the reference process.
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            if sb in benign_states:
                Q_compound.add_edge(sa, sb, weight=weight)

        # Add off-block-diagonal entries directed from the reference
        # to the background process.
        for s in range(nstates):
            Q_compound.add_edge(s, nstates + s, weight=rho)

        # Define the column-specific initial state distribution.
        compound_weights = {}
        for s in range(ncompound):
            if (s in primary_distn) and (s in benign_states):
                compound_weights[s] = primary_distn[s]
        compound_distn = _util.get_normalized_dict_distn(compound_weights)

        # Convert to dense representations.
        Q_compound_dense = _density.rate_matrix_to_numpy_array(
                Q_compound, nodelist=compound_states)
        compound_distn_dense = _density.dict_to_numpy_array(
                compound_distn, nodelist=compound_states)

        # Define the map from node to allowed compound states.
        node_to_allowed_states = dict((n, set(compound_states)) for n in T)
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            codon_state = codon_to_state[codon]
            node_to_allowed_states[leaf] = {codon_state, nstates + codon_state}

        # Get the likelihood for this column.
        likelihood = _mjp_dense.get_likelihood(
                T, node_to_allowed_states, root, ncompound,
                root_distn=compound_distn_dense,
                Q_default=Q_compound_dense)
        log_likelihood = np.log(likelihood)
        total_log_likelihood += log_likelihood
        print('column', i + 1, 'of', len(codon_columns), 'll', log_likelihood)
    
    # print the total log likelihood
    print('total log likelihood:', total_log_likelihood)


if __name__ == '__main__':
    main()

