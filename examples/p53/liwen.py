"""
Attempt to reproduce log likelihood of the default/reference switching model.

The input disease data file should be a 3-column tsv file with a header.
The first column should give the 1-indexed codon position.
The second column should give the amino acid.
The third column should give the BENIGN/LETHAL/UNKNOWN status.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import functools
import argparse

import networkx as nx
import numpy as np
import scipy.linalg
from scipy import special

import create_mg94
import app_helper
import qtop

from raoteh.sampler import (
        _util,
        _density,
        _mjp_dense,
        _mcy_dense,
        )

from raoteh.sampler._util import (
        StructuralZeroProb,
        )

BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'


def get_expm_augmented_tree(T, root, P_callback=None):
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        weight = edge['weight']
        #Q = edge.get('Q', Q_default)
        #_density.check_square_dense(Q)
        #P = custom_expm(Q, weight)
        P = P_callback(weight)
        T_aug.add_edge(na, nb, weight=weight, P=P)
    return T_aug

def get_likelihood(tree, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    if root not in tree:
        raise ValueError('the specified root is not in the tree')
    T_aug = get_expm_augmented_tree(tree, root, P_callback=P_callback)
    return _mcy_dense.get_likelihood(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states,
            root_distn=root_distn, P_default=None)


def get_jeff_params():

    # FINAL LIKELIHOOD IS -10279.111666
    # FINAL ESTIMATE: rho12 =    0.32771
    # FINAL ESTIMATE: for frequency of purines is    0.50781
    # FINAL ESTIMATE: for freq. of A among purines is    0.49542
    # FINAL ESTIMATE: for freq. of T among pyrimidines is    0.39194
    # FINAL ESTIMATE: kappa =    3.38998
    # FINAL ESTIMATE: omega =    0.40198

    # Jeff has reported these somewhere
    kappa = 3.38998
    omega = 0.40198
    AG = 0.50781
    CT = 1 - AG
    A = AG * 0.49542
    G = AG - A
    T = CT * 0.39194
    C = CT - T
    rho = 0.32771

    # Not sure what branch lengths to use for Jeff's estimate...
    tree_filename = 'liwen.estimated.tree'
    print('reading the newick tree...')
    with open(tree_filename) as fin:
        tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)


def get_codeml_estimated_params():

    # Use the estimates that I got from codeml,
    # for the default process.
    # Rho is from Jeff's estimate because codeml
    # doesn't do the switching process.
    kappa = 3.17632
    omega = 0.21925
    T = 0.18883
    C = 0.30126
    A = 0.25039
    G = 0.25952
    rho = 0.32771
    tree_filename = 'codeml.estimated.tree'
    print('reading the newick tree...')
    with open(tree_filename) as fin:
        tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)


def get_liwen_toy_params():

    # Parameters reported by Liwen in her email
    kappa = 1.0
    omega = 1.0
    A = 0.25
    C = 0.25
    G = 0.25
    T = 0.25
    rho = 1.15

    # Branch lengths are all the same,
    # and they are in units of expected number of codon changes
    # if the process were the default process (as opposed to reference).
    branch_length = 0.1
    tree_filename = 'liwen.estimated.tree'

    # Read the tree, and change the branch lengths
    # so that they are equal to the universal toy branch length.
    print('reading the newick tree...')
    with open(tree_filename) as fin:
        tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    for na, nb in tree.edges():
        tree[na][nb]['weight'] = branch_length

    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)

def read_interpreted_disease_data(fin):
    """
    Read some filtered disease data.

    The interpretation filters the p53 disease data
    by assigning a disease state to each of the 20 amino acids,
    for each codon position in the reference (human) p53 sequence.
    The possible disease states are BENIGN, LETHAL, or UNKNOWN.

    """
    interpreted_disease_data = []
    lines = fin.readlines()[1:]
    for line in lines:
        if not line.strip():
            continue
        codon_pos, aa_residue, status = line.split()
        codon_pos = int(codon_pos)
        row = (codon_pos, aa_residue, status)
        interpreted_disease_data.append(row)
    return interpreted_disease_data



def main(args):

    # Pick some parameters.
    info = get_liwen_toy_params()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = info
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)

    # Read the genetic code.
    print('reading the genetic code...')
    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # Check that the states are in the correct order.
    nstates = len(genetic_code)
    if range(nstates) != [s for s, r, c in genetic_code]:
        raise Exception
    states = range(nstates)

    # Define the default process codon rate matrix
    # and distribution and tolerance classes.
    info = create_mg94.create_mg94(
            A, C, G, T,
            kappa, omega, genetic_code,
            target_expected_rate=1.0)
    Q, primary_distn, state_to_residue, residue_to_part = info
    primary_to_part = dict(
            (i, residue_to_part[r]) for i, r in state_to_residue.items())

    # Define the dense default process codon rate matrix and distribution.
    Q_dense = _density.rate_matrix_to_numpy_array(
            Q, nodelist=states)
    primary_distn_dense = _density.dict_to_numpy_array(
            primary_distn, nodelist=states)

    # Report the genetic code.
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
    print()
    print('codon distn:')
    app_helper.print_codon_distn(codon_to_state, primary_distn)
    print()

    # Define an SD decomposition of the default process rate matrix.
    D1 = primary_distn_dense
    S1 = np.dot(Q_dense, np.diag(qtop.pseudo_reciprocal(D1)))

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    root = name_to_leaf['Has']

    # print a summary of the tree
    degree = tree.degree()
    print('number of nodes:', len(tree))
    print('number of leaves:', degree.values().count(1))
    print('number of branches:', tree.size())
    print('total branch length:', tree.size(weight='weight'))
    print()

    # Read the alignment.
    print('reading the alignment...')
    with open('testseq') as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # Read the interpreted disease data.
    with open(args.disease) as fin:
        interpreted_disease_data = read_interpreted_disease_data(fin)
    pos_to_benign_residues = defaultdict(set)
    pos_to_lethal_residues = defaultdict(set)
    for pos, residue, status in interpreted_disease_data:
        if status == BENIGN:
            pos_to_benign_residues[pos].add(residue)
        elif status == LETHAL:
            pos_to_lethal_residues[pos].add(residue)
        elif status == UNKNOWN:
            raise NotImplementedError(
                    'unknown amino acid status in the reference process '
                    'requires integrating over too many things')
        else:
            raise Exception('invalid disease status: ' + str(status))
    pos_to_benign_residues = dict(pos_to_benign_residues)
    pos_to_lethal_residues = dict(pos_to_lethal_residues)

    # compute the log likelihood, column by column
    # using _mjp_dense (the dense Markov jump process module).
    print('preparing to compute log likelihood...')
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    print('computing log likelihood...')
    total_ll_default = 0
    total_ll_reference = 0
    total_ll_compound = 0
    for i, codon_column in enumerate(codon_columns):

        # Define the column-specific disease states and the benign states.
        pos = i + 1
        benign_residues = pos_to_benign_residues.get(pos, set())
        lethal_residues = pos_to_lethal_residues.get(pos, set())
        benign_states = set()
        lethal_states = set()
        for s, r, c in genetic_code:
            if r in benign_residues:
                benign_states.add(s)
            elif r in lethal_residues:
                lethal_states.add(s)
            else:
                print(benign_residues)
                print(lethal_residues)
                raise Exception(
                        'each amino acid should be considered either '
                        'benign or lethal in this model, '
                        'but residue %s at position %s '
                        'was found to be neither' % (r, pos))

        # Define the reference process.

        # Define the reference process rate matrix.
        Q_reference = nx.DiGraph()
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            if sa in benign_states and sb in benign_states:
                Q_reference.add_edge(sa, sb, weight=weight)

        # Define the column-specific initial state distribution.
        reference_weights = {}
        for s in range(nstates):
            if (s in primary_distn) and (s in benign_states):
                reference_weights[s] = primary_distn[s]
        reference_distn = _util.get_normalized_dict_distn(reference_weights)

        # Convert to dense representations of the reference process.
        Q_reference_dense = _density.rate_matrix_to_numpy_array(
                Q_reference, nodelist=states)
        reference_distn_dense = _density.dict_to_numpy_array(
                reference_distn, nodelist=states)

        # Define an SD decomposition of the reference process.
        L = np.array(
                [rho if s in benign_states else 0 for s in range(nstates)],
                dtype=float)
        D0 = reference_distn_dense
        S0 = qtop.dot_square_diag(Q_reference_dense, qtop.pseudo_reciprocal(D0))

        # Define the decompositions.
        # Define the callbacks that converts branch length to prob matrix.
        sylvester_decomp = qtop.decompose_sylvester_v2(S0, S1, D0, D1, L)
        A0, B0, A1, B1, L, lam0, lam1, XQ = sylvester_decomp
        P_cb_sylvester = functools.partial(qtop.getp_sylvester_v2,
                D0, A0, B0, A1, B1, L, lam0, lam1, XQ)
        A0, lam0, B0 = qtop.decompose_spectral_v2(S0, D0)
        P_cb_reference = functools.partial(
                qtop.getp_spectral_v2, D0, A0, lam0, B0)
        A1, lam1, B1 = qtop.decompose_spectral_v2(S1, D1)
        P_cb_default = functools.partial(
                qtop.getp_spectral_v2, D1, A1, lam1, B1)

        # Define the compound process.

        # Define the compound process state space.
        ncompound = 2 * nstates
        compound_states = range(ncompound)

        # Initialize the column-specific compound rate matrix.
        Q_compound = nx.DiGraph()
        
        # Add block-diagonal entries of the default process component
        # of the compound process.
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            Q_compound.add_edge(nstates + sa, nstates + sb, weight=weight)

        # Add block-diagonal entries of the reference process component
        # of the compound process.
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            if sb in benign_states:
                Q_compound.add_edge(sa, sb, weight=weight)

        # Add off-block-diagonal entries directed from the reference
        # to the default process.
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

        # End compound process definition.

        # Define the map from node to allowed compound states.
        node_to_allowed_states = dict((n, set(compound_states)) for n in tree)
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            codon_state = codon_to_state[codon]
            node_to_allowed_states[leaf] = {codon_state, nstates + codon_state}

        # Get the log likelihood for the reference process.
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=reference_distn_dense,
                    P_callback=P_cb_reference)
            ll_reference = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_reference = -np.inf

        # Get the log likelihood for the default process.
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=primary_distn_dense,
                    P_callback=P_cb_default)
            ll_default = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_default = -np.inf

        # Get the log likelihood for the compound process.
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, ncompound,
                    root_distn=compound_distn_dense,
                    P_callback=P_cb_sylvester)
            ll_compound = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_compound = -np.inf

        total_ll_reference += ll_reference
        total_ll_default += ll_default
        total_ll_compound += ll_compound

        print(
                'column', i + 1, 'of', len(codon_columns),
                'll_reference:', ll_reference,
                'll_default:', ll_default,
                'll_compound:', ll_compound,
                'lethal_residues:', lethal_residues)
    
    # print the total log likelihoods
    print('total reference log likelihood:', total_ll_reference)
    print('total default log likelihood:', total_ll_default)
    print('total compound log likelihood:', total_ll_compound)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    main(parser.parse_args())

