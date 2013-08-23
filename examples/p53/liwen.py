"""
Attempt to reproduce log likelihood of the default/reference switching model.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import functools

import networkx as nx
import numpy as np
import scipy.linalg
from scipy import special

import create_mg94
import app_helper

from raoteh.sampler import (
        _util,
        _density,
        _mjp_dense,
        )

from raoteh.sampler._util import (
        StructuralZeroProb,
        )


def ndot(*args):
    M = args[0]
    for B in args[1:]:
        M = np.dot(M, B)
    return M

def build_block_2x2(M):
    return np.vstack([np.hstack(M[0]), np.hstack(M[1])])

def get_original(
        S0, S1, D0, D1, L,
        ):
    """
    Return the original matrix given its block form.
    """
    Q_original = build_block_2x2([
        [ndot(S0, np.diag(D0)) - np.diag(L), np.diag(L)],
        [np.zeros_like(np.diag(L)), ndot(S1, np.diag(D1))],
        ])
    return Q_original

def get_reconstructed(
        D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    """
    Return the reconstructed matrix given a spectral form.
    """
    R11 = ndot(
            np.diag(np.reciprocal(np.sqrt(D0))),
            U0,
            np.diag(lam0),
            U0.T,
            np.diag(np.reciprocal(D0)),
            )
    R22 = ndot(
            np.diag(np.reciprocal(np.sqrt(D1))),
            U1,
            np.diag(lam1),
            U1.T,
            np.diag(np.reciprocal(D1)),
            )
    Q_reconstructed = build_block_2x2([
        [R11, ndot(R11, XQ) - ndot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed


def check_decomp(
        S0, S1, D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    Q_original = get_original(S0, S1, D0, D1, L)
    Q_reconstructed = get_reconstructed(
            D0, D1, L,
            U0, U1, lam0, lam1, XQ)
    testing.assert_array_almost_equal(Q_original, Q_reconstructed)

def check_decomp_expm(
        S0, S1, D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    t = 0.123
    Q_original = get_original(S0, S1, D0, D1, L)
    Q_original_expm = scipy.linalg.expm(t * Q_original)
    Q_spectral_expm = get_reconstructed(
            D0, D1, L,
            U0,
            U1,
            np.exp(t * lam0),
            np.exp(t * lam1),
            XQ,
            )
    testing.assert_array_almost_equal(
            Q_original_expm,
            Q_spectral_expm,
            )


def foo(S0, S1, D0, D1, L, t):
    #FIXME: this code uses slow ways to multiply by diagonal matrices

    # load the input ndarrays
    #S0 = np.loadtxt(args.S0_in)
    #S1 = np.loadtxt(args.S1_in)
    #D0 = np.loadtxt(args.D0_in)
    #D1 = np.loadtxt(args.D1_in)
    #L = np.loadtxt(args.L_in)

    # compute the first symmetric eigendecomposition
    D0_sqrt = np.sqrt(D0)
    H0 = ndot(np.diag(D0_sqrt), S0, np.diag(D0_sqrt)) - np.diag(L)
    lam0, U0 = scipy.linalg.eigh(H0)

    # compute the second symmetric eigendecomposition
    D1_sqrt = np.sqrt(D1)
    H1 = ndot(np.diag(D1_sqrt), S1, np.diag(D1_sqrt))
    lam1, U1 = scipy.linalg.eigh(H1)

    # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
    A = ndot(S0, np.diag(D0)) - np.diag(L)
    B = -ndot(S1, np.diag(D1))
    Q = np.diag(L)
    XQ = scipy.linalg.solve_sylvester(A, B, Q)

    # check some stuff if debugging
    if args.debug:
        check_decomp(
                S0, S1, D0, D1, L,
                U0, U1, lam0, lam1, XQ)
        check_decomp_expm(
                S0, S1, D0, D1, L,
                U0, U1, lam0, lam1, XQ)

    # write the output ndarrays
    #fmt = '%.17g'
    #np.savetxt(args.U0_out, U0, fmt)
    #np.savetxt(args.U1_out, U1, fmt)
    #np.savetxt(args.lam0_out, lam0, fmt)
    #np.savetxt(args.lam1_out, lam1, fmt)
    #np.savetxt(args.XQ_out, XQ, fmt)

    Q_spectral_expm = get_reconstructed(
            D0, D1, L,
            U0,
            U1,
            np.exp(t * lam0),
            np.exp(t * lam1),
            XQ,
            )


def bar():
    """
    """
    pass



def get_likelihood(T, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Edges of this tree are annotated with weights and possibly with
        edge-specific Q rate matrices.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    nstates : integer
        Number of states.
    root_distn : 2d ndarray, optional
        Distribution over states at the root.
    Q_default : 2d ndarray, optional
        A rate matrix.

    Returns
    -------
    likelihood : float
        A marginal likelihood.

    Notes
    -----
    This function is meaningful even when the root_distn is not technically
    a distribution in the sense of summing to 1.
    If the root distribution is not specified,
    then it is treated as all ones; this is different than
    treating it as a uniform distribution on some set of states.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = get_expm_augmented_tree(T, root, Q_default=Q_default)

    # Return the Markov chain likelihood.
    return _mcy_dense.get_likelihood(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states,
            root_distn=root_distn, P_default=None)


def get_probability_matrix_sylvester(D0, D1, L, U0, U1, lam0, lam1, XQ, t):
    """
    This is intended to be called through functools.partial().

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    P = get_reconstructed(
            D0, D1, L, U0, np.exp(t * lam0), np.exp(t * lam1), XQ)
    return P

def get_probability_matrix_spectral(d, w, U, t):
    """
    This is intended to be called through functools.partial().

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    pass


def pseudo_reciprocal_1d(arr):
    return np.array([1/x if x else 0 for x in arr], dtype=float)


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


def main():

    # Pick some parameters.
    info = get_liwen_toy_params()
    kappa, omega, A, C, T, G, rho, tree, root, leaf_name_pairs = info
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)

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

    """
    # Define a decomposition of the default process rate matrix.
    S1 = ndot(Q_dense, np.diag(pseudo_reciprocal(primary_distn_dense)))
    D1 = primary_distn_dense
    M1 = ndot(
            np.diag(np.sqrt(D1)),
            S1,
            np.diag(np.sqrt(D1)),
            )
    lam1, U1 = scipy.linalg.eigh(M1)
    """

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    root = name_to_leaf['Has']

    # print a summary of the tree
    degree = tree.degree()
    print('number of nodes:', len(tree))
    print('number of leaves:', degree.values().count(1))
    print('number of branches:', tree.size())
    print('total branch length:', tree.size(weight='weight'))
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
    total_ll_default = 0
    total_ll_reference = 0
    total_ll_compound = 0
    for i, codon_column in enumerate(codon_columns):

        # Define the column-specific disease states and the benign states.
        disease_residues = column_to_disease_residues.get(i, set())
        disease_states = set(
                s for s, r, c in genetic_code if r in disease_residues)
        benign_states = set(range(nstates)) - disease_states

        # Define the reference process.

        # Define the reference process rate matrix.
        Q_reference = nx.DiGraph()
        for sa, sb in Q.edges():
            weight = Q[sa][sb]['weight']
            if sb in benign_states:
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

        """
        # Define a decomposition of the reference process.
        L = np.array(
                [rho if s in benign_states else 0 for s in range(nstates)],
                dtype=float)
        S0 = ndot(Q_dense, np.diag(pseudo_reciprocal(primary_distn_dense)))
        D0 = reference_distn_dense
        M0 = ndot(
                np.diag(np.sqrt(D0)),
                S0,
                np.diag(np.sqrt(D0)),
                )
        lam0, U0 = scipy.linalg.eigh(M0 - np.diag(L))

        # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
        A = ndot(S0, np.diag(D0)) - np.diag(L)
        B = -ndot(S1, np.diag(D1))
        Q = np.diag(L)
        XQ = scipy.linalg.solve_sylvester(A, B, Q)

        # Define the callback that converts branch length to prob matrix.
        P_callback = functools.partial(
                sylvester_get_probability_matrix,
                D0, D1, L, U0, U1, lam0, lam1, XQ)
        """

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
            likelihood = _mjp_dense.get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=reference_distn_dense,
                    Q_default=Q_reference_dense)
            ll_reference = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_reference = -np.inf

        # Get the log likelihood for the default process.
        try:
            likelihood = _mjp_dense.get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=primary_distn_dense,
                    Q_default=Q_dense)
            ll_default = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_default = -np.inf

        # Get the log likelihood for the compound process.
        try:
            likelihood = _mjp_dense.get_likelihood(
                    tree, node_to_allowed_states, root, ncompound,
                    root_distn=compound_distn_dense,
                    Q_default=Q_compound_dense)
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
                'disease_residues:', disease_residues)
    
    # print the total log likelihoods
    print('total reference log likelihood:', total_ll_reference)
    print('total default log likelihood:', total_ll_default)
    print('total compound log likelihood:', total_ll_compound)


if __name__ == '__main__':
    main()

