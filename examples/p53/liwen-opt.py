"""
Attempt to reproduce log likelihood of the default/reference switching model.

The input disease data file should be a 3-column tsv file with a header.
The first column should give the 1-indexed codon position.
The second column should give the amino acid.
The third column should give the BENIGN/LETHAL/UNKNOWN status.

"""
from __future__ import division, print_function, absolute_import

from StringIO import StringIO
from collections import defaultdict
import functools
import argparse

import networkx as nx
import numpy as np
import scipy.linalg
from scipy import special
import pyfelscore

import create_mg94
import app_helper
import qtop
import jeffopt

from raoteh.sampler import (
        _util,
        _density,
        _mjp_dense,
        _mcy_dense,
        _mc0_dense,
        )

from raoteh.sampler._util import (
        StructuralZeroProb,
        )

BENIGN = 'BENIGN'
LETHAL = 'LETHAL'
UNKNOWN = 'UNKNOWN'



#TODO why am I getting negative inf for dt=0.01
#     for the initial parameter values when I use the optimization script,
#     but I get a finite log likelihood
#     when I use the script dedicated to log likelihood calculation?


def pack_params(
        blen_props, root_depth, 
        rho12, A, C, G, T, kappa, omega):
    """
    A helper function for maximum likelihood search.

    The first two params (blen_props and root_depth) are already
    in a packed format.

    """
    if len(blen_props) != 23:
        raise Exception

    AG = A + G
    CT = C + T
    kappa_d10 = kappa / 10
    omega_d10 = omega / 10
    v = np.empty(30)

    v[:23] = blen_props
    v[23] = root_depth
    v[24] = rho12
    v[25] = AG
    v[26] = A / AG
    v[27] = T / CT
    v[28] = kappa_d10
    v[29] = omega_d10

    return v


def unpack_params(v, tree, original_root):
    """
    A helper function for maximum likelihood search.

    The tree will be annotated with branch lengths (networkx 'weights').

    """
    if len(v) != 30:
        raise ValueError('expected a 30-dimensional search space, '
                'including parameter values and branch length proportions')

    # Unpack the parameters.
    blen_props = v[:23]
    root_depth = v[23]
    rho12 = v[24]
    prop_AG_of_all = v[25]
    prop_A_of_AG = v[26]
    prop_T_of_CT = v[27]
    kappa_d10 = v[28]
    omega_d10 = v[29]

    kappa = kappa_d10 * 10
    omega = omega_d10 * 10

    # Further unpacking of mutational process nucleotide proportions.
    prop_CT_of_all = 1 - prop_AG_of_all
    prop_G_of_AG = 1 - prop_A_of_AG
    prop_C_of_CT = 1 - prop_T_of_CT
    A = prop_A_of_AG * prop_AG_of_all
    C = prop_C_of_CT * prop_CT_of_all
    G = prop_G_of_AG * prop_AG_of_all
    T = prop_T_of_CT * prop_CT_of_all

    # Define the ordering of the nodes that correspond to the
    # branch length proportions.
    selected_nodes = set()
    node_to_degree = tree.degree()
    for na, nb in nx.bfs_edges(tree, original_root):
        if node_to_degree[nb] > 1:
            selected_nodes.add(nb)
    ordered_selected_nodes = sorted(selected_nodes)
    selected_node_to_prop = dict(
            (nb, p) for nb, p in zip(ordered_selected_nodes, blen_props))

    # Unpack distances from root,
    # and set branch lengths as networkx edge weights.
    node_to_dist_from_root = {original_root : 0}
    for na, nb in nx.bfs_edges(tree, original_root):
        dist_from_root_to_na = node_to_dist_from_root[na]
        if nb in selected_nodes:
            p = selected_node_to_prop[nb]
        else:
            p = 1
        weight = p * (root_depth - dist_from_root_to_na)
        tree[na][nb]['weight'] = weight
        node_to_dist_from_root[nb] = dist_from_root_to_na + weight

    # Return unpacked parameter values.
    # The parameter values related to branch lengths have already
    # been set as weights on the tree.
    return rho12, A, C, G, T, kappa, omega


def getp_approx(Q, t):
    """
    Approximate the transition matrix over a small time interval.
    """
    ident = np.eye(*Q.shape)
    return ident + Q*t

def getp_bigt_approx(Q, dt, t):
    """
    Approximate exp(x) = exp(x/n)^n ~= (1 + x/n)^n.
    """
    n = max(1, int(np.ceil(t / dt)))
    psmall = getp_approx(Q, t/n)
    return np.linalg.matrix_power(psmall, n)

def get_expm_augmented_tree(T, root, P_callback=None):
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        weight = edge['weight']
        P = P_callback(weight)
        T_aug.add_edge(na, nb, weight=weight, P=P)
    return T_aug

def get_node_to_distn(tree, node_to_allowed_states, root, nstates,
        root_distn=None, P_callback=None):
    if root not in tree:
        raise ValueError('the specified root is not in the tree')
    T_aug = get_expm_augmented_tree(tree, root, P_callback=P_callback)
    node_to_pmap = _mcy_dense.get_node_to_pmap(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states)
    node_to_distn = _mc0_dense.get_node_to_distn_esd(
            T_aug, root, node_to_pmap, nstates,
            root_distn=root_distn)
    return node_to_distn

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


def get_jeff_params_b():
    # FINAL ESTIMATE: rho12 =    0.61343
    # FINAL ESTIMATE: for frequency of purines is    0.50759
    # FINAL ESTIMATE: for freq. of A among purines is    0.49489
    # FINAL ESTIMATE: for freq. of T among pyrimidines is    0.39140
    # FINAL ESTIMATE: kappa =    3.35879
    # FINAL ESTIMATE: omega =    0.37953

    kappa = 3.35879
    omega = 0.37953
    AG = 0.50759
    CT = 1 - AG
    A = AG * 0.49489
    G = AG - A
    T = CT * 0.39140
    C = CT - T
    rho = 0.61343

    tree_string = """((((((Has:  0.00750,Ptr:  0.00750):  0.06478,Ppy:  0.07228):  0.05425,(((Mmu:  0.00256,Mfu:  0.00256):  0.00000,Mfa:  0.00256):  0.03202,Cae:  0.03458):  0.09195):  0.20038,(Mim:  0.32691,Tgl:  0.32691):  0.00000):  0.22356,((((((Mum:  0.18000,Rno:  0.18000):  0.15514,Mun:  0.33514):  0.01932,(Cgr:  0.10721,Mau:  0.10721):  0.24724):  0.04431,Sju:  0.39876):  0.10020,(Cpo:  0.41556,Mmo:  0.41556):  0.08341):  0.02586,(Ocu:  0.41458,Opr:  0.41458):  0.11024):  0.02565):  0.00000,(Sar:  0.45268,((Fca:  0.28000,Cfa:  0.28000):  0.13266,((Bta:  0.09038,Oar:  0.09038):  0.15438,Dle:  0.24476):  0.16790):  0.04002):  0.09779);"""
    fin = StringIO(tree_string)
    tree, root, leaf_name_pairs = app_helper.read_newick(fin)
    return (kappa, omega, A, C, T, G, rho,
            tree, root, leaf_name_pairs)

def get_jeff_params_c():
    # with kappa = 2.0,
    # omega = 0.5,
    # freq of A = 0.12,
    # freq of G = 0.18,
    # freq of C = 0.35,
    # freq of T = 0.35
    # and (for switching model) switching parameter = 0.5 ?

    kappa = 2.0
    omega = 0.5
    A = 0.12
    G = 0.18
    C = 0.35
    T = 0.35
    rho = 0.5

    tree_string = """((((((Has:  0.0156250000,Ptr:  0.0156250000):  0.0156250000,Ppy:  0.0312500000):  0.0312500000,(((Mmu:  0.0078125000,Mfu:  0.0078125000):  0.0078125000,Mfa:  0.0156250000):  0.0156250000,Cae:  0.0312500000):  0.0312500000):  0.0625000000,(Mim:  0.0625000000,Tgl:  0.0625000000):  0.0625000000):  0.1250000000,((((((Mum:  0.0039062500,Rno:  0.0039062500):  0.0039062500,Mun:  0.0078125000):  0.0078125000,(Cgr:  0.0078125000,Mau:  0.0078125000):  0.0078125000):  0.0156250000,Sju:  0.0312500000):  0.0312500000,(Cpo:  0.0312500000,Mmo:  0.0312500000):  0.0312500000):  0.0625000000,(Ocu:  0.0625000000,Opr:  0.0625000000):  0.0625000000):  0.1250000000):  0.2500000000,(Sar:  0.2500000000,((Fca:  0.0625000000,Cfa:  0.0625000000):  0.0625000000,((Bta:  0.0312500000,Oar:  0.0312500000):  0.0312500000,Dle:  0.0625000000):  0.0625000000):  0.1250000000):  0.2500000000);"""
    fin = StringIO(tree_string)
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



def get_codon_site_inferences(
        tree, node_to_allowed_states, root, original_root,
        nstates, ncompound,
        default_root_distn, reference_root_distn, compound_root_distn,
        P_cb_default, P_cb_reference, P_cb_compound,
        ):
    """
    Return posterior info for a single codon site under various models.

    This info includes log likelihoods and also the posterior
    probability that the original root of the tree is in the reference process
    (as opposed to the default process).

    """
    ll_default = None
    ll_reference = None
    ll_compound = None
    p_reference = None

    # Get the log likelihood for the default process.
    if P_cb_default is not None:
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=default_root_distn,
                    P_callback=P_cb_default)
            ll_default = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_default = -np.inf

    # Get the log likelihood for the reference process.
    if P_cb_reference is not None:
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, nstates,
                    root_distn=reference_root_distn,
                    P_callback=P_cb_reference)
            ll_reference = np.log(likelihood)
        except StructuralZeroProb as e:
            ll_reference = -np.inf

    # Get the log likelihood for the compound process.
    # Also get the distributions at each node,
    # and reduce this to the probability that the original root
    # is in the reference process.
    if P_cb_compound is not None:
        try:
            likelihood = get_likelihood(
                    tree, node_to_allowed_states, root, ncompound,
                    root_distn=compound_root_distn,
                    P_callback=P_cb_compound)
            ll_compound = np.log(likelihood)
            #node_to_distn = get_node_to_distn(
                    #tree, node_to_allowed_states, root, ncompound,
                    #root_distn=compound_root_distn,
                    #P_callback=P_cb_compound)
            #p_reference = node_to_distn[original_root][:nstates].sum()
        except StructuralZeroProb as e:
            ll_compound = -np.inf

    return ll_default, ll_reference, ll_compound, p_reference



def main(args):

    # Use a function to read the tree plus other things we will not use.
    info = get_jeff_params_c()
    kappa, omega, A, C, T, G, rho, tree, original_root, leaf_name_pairs = info
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)

    # Define initial parameter guesses.
    blen_props = np.array([0.5]*23)
    root_depth = 0.5
    rho12 = 0.5
    A = 0.12
    C = 0.35
    G = 0.18
    T = 0.35
    kappa = 2.0
    omega = 0.5
    v_initial = pack_params(
            blen_props, root_depth,
            rho12, A, C, G, T, kappa, omega)

    # Read the genetic code.
    print('reading the genetic code...')
    with open('universal.code.txt') as fin:
        genetic_code = app_helper.read_genetic_code(fin)

    # Report the genetic code.
    print('genetic code:')
    for triple in genetic_code:
        print(triple)
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

    # This numerical optimization search is designed for robustness.
    f = functools.partial(get_log_likelihood,
            genetic_code, tree, original_root, name_to_leaf,
            name_codons_list,
            pos_to_benign_residues, pos_to_lethal_residues,
            args.dt,
            )
    v_opt, value_opt = jeffopt.fmax_jeff(f, v_initial)

    # Unpack the optimal parameter values.
    rho12, A, C, G, T, kappa, omega = unpack_params(
            v_opt, tree, original_root)

    # Report the optimal parameter values.
    print('optimal value:', value_opt)
    print('packed optimal parameter array:', v_opt)
    print('optimal parameter values:')
    print('  rho12:', rho12)
    print('  A:', A)
    print('  C:', C)
    print('  G:', G)
    print('  T:', T)
    print('  kappa:', kappa)
    print('  omega:', omega)


def get_log_likelihood(
        genetic_code, tree, original_root, name_to_leaf,
        name_codons_list,
        pos_to_benign_residues, pos_to_lethal_residues,
        dt,
        v_packed):

    #
    print('computing log likelihood for packed parameter values')
    print(v_packed)

    # Preprocessing.
    codon_to_state = dict((c, s) for s, r, c in genetic_code)

    # Check that the states are in the correct order.
    nstates = len(genetic_code)
    if range(nstates) != [s for s, r, c in genetic_code]:
        raise Exception
    states = range(nstates)

    # Unpack the parameter values.
    rho12, A, C, G, T, kappa, omega = unpack_params(
            v_packed, tree, original_root)

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

    # Define an SD decomposition of the default process rate matrix.
    D1 = primary_distn_dense
    S1 = np.dot(Q_dense, np.diag(qtop.pseudo_reciprocal(D1)))

    # Change the root to 'Has' which is typoed from 'H'omo 'sa'piens.
    root = name_to_leaf['Has']

    # compute the log likelihood, column by column
    # using _mjp_dense (the dense Markov jump process module).
    #print('preparing to compute log likelihood...')
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*codon_sequences)
    # Get the row index of the homo sapiens name.
    reference_codon_row_index = names.index('Has')
    #total_ll_default_cont = 0
    #total_ll_reference_cont = 0
    #total_ll_compound_cont = 0
    #total_ll_default_disc = 0
    #total_ll_reference_disc = 0
    total_ll_compound = 0
    #cond_adj_total = 0
    for i, codon_column in enumerate(codon_columns):

        print('column', i+1)

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
            Q_compound.add_edge(s, nstates + s, weight=rho12)

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

        # Define the t -> P callbacks for the true process
        # or for a time-discretized process, depending on cmdline flags.

        P_cb_default = None
        P_cb_reference = None
        P_cb_compound = None

        if dt is None:

            # Define the diagonal associated with switching processes.
            L = np.zeros(nstates, dtype=float)
            for s in range(nstates):
                if s in benign_states:
                    L[s] = rho12

            # Define an SD decomposition of the reference process.
            D0 = reference_distn_dense
            S0 = qtop.dot_square_diag(
                    Q_reference_dense, qtop.pseudo_reciprocal(D0))

            # Define the decompositions.
            # Define the callbacks that converts branch length to prob matrix.
            sylvester_decomp = qtop.decompose_sylvester_v2(S0, S1, D0, D1, L)
            A0, B0, A1, B1, L, lam0, lam1, XQ = sylvester_decomp
            P_cb_compound = functools.partial(qtop.getp_sylvester_v2,
                    D0, A0, B0, A1, B1, L, lam0, lam1, XQ)
            #A0, lam0, B0 = qtop.decompose_spectral_v2(S0, D0)
            #P_cb_reference = functools.partial(
                    #qtop.getp_spectral_v2, D0, A0, lam0, B0)
            #A1, lam1, B1 = qtop.decompose_spectral_v2(S1, D1)
            #P_cb_default = functools.partial(
                    #qtop.getp_spectral_v2, D1, A1, lam1, B1)

        else:

            # Define a discretization.
            f = getp_bigt_approx

            # Define the callbacks.
            P_cb_compound = functools.partial(f, Q_compound_dense, dt)
            #P_cb_reference = functools.partial(f, Q_reference_dense, dt)
            #P_cb_default = functools.partial(f, Q_dense, dt)

        # Define the map from node to allowed compound states.
        node_to_allowed_states = dict((n, set(compound_states)) for n in tree)
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            codon_state = codon_to_state[codon]
            node_to_allowed_states[leaf] = {codon_state, nstates + codon_state}

        site_info = get_codon_site_inferences(
                tree, node_to_allowed_states, root, original_root,
                nstates, ncompound,
                primary_distn_dense,
                reference_distn_dense,
                compound_distn_dense,
                P_cb_default, P_cb_reference, P_cb_compound,
                )

        ll_default, ll_reference, ll_compound, p_reference = site_info

        total_ll_compound += ll_compound

    print('log likelihood:', total_ll_compound)

    return total_ll_compound



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--disease', required=True,
            help='csv file with filtered disease data')
    parser.add_argument('--dt', type=float,
            help='discretize the tree with this maximum branchlet length')
    main(parser.parse_args())

