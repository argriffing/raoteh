"""
Test functions that sample tolerance Markov jump trajectories on a tree.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx
from scipy import special


from raoteh.sampler import (
        _mjp,
        _mjp_dense,
        _tmjp,
        _tmjp_dense,
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

from raoteh.sampler._mjp import (
        get_history_statistics,
        get_expected_history_statistics,
        )

from raoteh.sampler._tmjp import (
        get_tolerance_process_log_likelihood,
        get_tolerance_summary,
        get_primary_proposal_rate_matrix,
        get_example_tolerance_process_info,
        )


#TODO copypasted from test_tmjp_
#TODO this should go into the _tmjp_dense module.
def _compound_ll_expectation_helper_dense(
        primary_to_part, rate_on, rate_off,
        Q_primary, primary_distn, T_primary_aug, root):
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

    Returns
    -------
    init_ll : float
        x
    dwell_ll : float
        x
    trans_ll : float
        x

    """
    check_square_dense(Q_primary)
    nprimary = Q_primary.shape[0]
    if primary_distn.shape[0] != nprimary:
        raise ValueError('inconsistency in the number of primary states')

    total_tree_length = T_primary_aug.size(weight='weight')
    primary_info = _mjp_dense.get_history_statistics(
            T_primary_aug, nprimary, root=root)
    dwell_times, root_state, transitions = primary_info
    post_root_distn = np.zeros(nprimary, dtype=float)
    post_root_distn[root_state] = 1

    neg_ll_info = _differential_entropy_helper_dense(
            Q_primary, primary_distn,
            post_root_distn, dwell_times, transitions)
    neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

    tol_summary = _tmjp_dense.get_tolerance_summary(
            primary_to_part, rate_on, rate_off,
            Q_primary, T_primary_aug, root)

    tol_info = _tmjp_dense.get_tolerance_ll_contribs(
            rate_on, rate_off, total_tree_length, *tol_summary)
    init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

    init_ll = -neg_init_prim_ll + init_tol_ll
    dwell_ll = dwell_tol_ll
    trans_ll = -neg_trans_prim_ll + trans_tol_ll

    return init_ll, dwell_ll, trans_ll


#TODO this should be a log likelihood helper function in the _mjp_dense module.
def _differential_entropy_helper_dense(
        Q, prior_root_distn,
        post_root_distn, post_dwell_times, post_transitions,
        ):
    """
    Use posterior expectations to help compute differential entropy.

    Parameters
    ----------
    Q : 2d ndarray
        Rate matrix.
    prior_root_distn : 1d ndarray
        Prior distribution at the root.
        If Q is a time-reversible rate matrix,
        then the prior root distribution
        could be the stationary distribution associated with Q.
    post_root_distn : 1d ndarray
        Posterior state distribution at the root.
    post_dwell_times : 1d ndarray
        Posterior expected dwell time for each state.
    post_transitions : 2d ndarray
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
    check_square_dense(Q)
    check_square_dense(post_transitions)
    nstates = Q.shape[0]

    # Get the total rates.
    total_rates = _mjp_dense.get_total_rates(Q)

    # Initial state distribution contribution to differential entropy.
    diff_ent_init = -special.xlogy(post_root_distn, prior_root_distn).sum()

    # Dwell time contribution to differential entropy.
    diff_ent_dwell = post_dwell_times.dot(total_rates)

    # Transition contribution to differential entropy.
    diff_ent_trans = -special.xlogy(post_transitions, Q).sum()

    # Return the contributions to differential entropy.
    return diff_ent_init, diff_ent_dwell, diff_ent_trans


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

        primary_info = get_history_statistics(T_primary_aug, root=root)
        dwell_times, root_state, transitions = primary_info
        post_root_distn = {root_state : 1}

        neg_ll_info = _differential_entropy_helper_sparse(
                Q_primary, primary_distn,
                post_root_distn, dwell_times, transitions)
        neg_init_prim_ll = neg_ll_info[0]
        neg_dwell_prim_ll = neg_ll_info[1]
        neg_trans_prim_ll = neg_ll_info[2]

        tol_summary = get_tolerance_summary(
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

        d_neg_ll_info = _differential_entropy_helper_dense(
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


def _tmjp_clever_sample_helper_dense(
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

        # use the dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper_dense(
                primary_to_part, rate_on, rate_off,
                Q_primary_dense, primary_distn_dense,
                T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        d_neg_ll_contribs_init.append(-ll_init)
        d_neg_ll_contribs_dwell.append(-ll_dwell)
        d_neg_ll_contribs_trans.append(-ll_trans)

    d_neg_ll_contribs = (
            d_neg_ll_contribs_init,
            d_neg_ll_contribs_dwell,
            d_neg_ll_contribs_trans)
    
    return d_neg_ll_contribs


def _tmjp_clever_sample_helper(
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

        # use the non-dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper(
                primary_to_part, rate_on, rate_off,
                Q_primary, primary_distn, T_primary_aug, root)
        ll_init, ll_dwell, ll_trans = ll_info
        neg_ll_contribs_init.append(-ll_init)
        neg_ll_contribs_dwell.append(-ll_dwell)
        neg_ll_contribs_trans.append(-ll_trans)

        # use the dense transition matrix rao-blackwellization
        ll_info = _compound_ll_expectation_helper_dense(
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

        # Get the total tree length.
        total_tree_length = T.size(weight='weight')

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

        # Compute contributions to differential entropy.
        diff_ent_info = _differential_entropy_helper_sparse(
            Q_compound, compound_distn,
            post_root_distn, dwell_times, transitions)
        diff_ent_init, diff_ent_dwell, diff_ent_trans = diff_ent_info

        # Get neg ll contribs using the clever sampler.
        # This calls a separate function for more isolated profiling.
        d_info = _tmjp_clever_sample_helper_dense(
                T, root, Q_primary, primary_to_part,
                leaf_to_primary_state, rate_on, rate_off,
                primary_distn, nsamples)
        #v1_neg_ll_contribs_init = v1_info[0]
        #v1_neg_ll_contribs_dwell = v1_info[1]
        #v1_neg_ll_contribs_trans = v1_info[2]
        d_neg_ll_contribs_init = d_info[0]
        d_neg_ll_contribs_dwell = d_info[1]
        d_neg_ll_contribs_trans = d_info[2]

        # Get neg ll contribs using the dumb sampler.
        # This calls a separate function for more isolated profiling.
        neg_ll_info, pm_neg_ll_info = _tmjp_dumb_sample_helper(
                T, primary_to_part, compound_to_primary,
                Q_compound, compound_distn,
                Q_primary, primary_distn,
                node_to_allowed_compound_states,
                root, rate_on, rate_off,
                nsamples)
        neg_ll_contribs_init = neg_ll_info[0]
        neg_ll_contribs_dwell = neg_ll_info[1]
        neg_ll_contribs_trans = neg_ll_info[2]
        pm_neg_ll_contribs_init = pm_neg_ll_info[0]
        pm_neg_ll_contribs_dwell = pm_neg_ll_info[1]
        pm_neg_ll_contribs_trans = pm_neg_ll_info[2]

    print()
    print('--- tmjp v1 test ---')
    print('nsamples:', nsamples)
    print()
    print('diff ent init :', diff_ent_init)
    print('neg ll init   :', np.mean(neg_ll_contribs_init))
    print('error         :', np.std(neg_ll_contribs_init) / sqrt_nsamp)
    print('pm neg ll init:', np.mean(pm_neg_ll_contribs_init))
    print('error         :', np.std(pm_neg_ll_contribs_init) / sqrt_nsamp)
    #print('v1 neg ll init:', np.mean(v1_neg_ll_contribs_init))
    #print('error         :', np.std(v1_neg_ll_contribs_init) / sqrt_nsamp)
    print('d neg ll init :', np.mean(d_neg_ll_contribs_init))
    print('error         :', np.std(d_neg_ll_contribs_init) / sqrt_nsamp)
    print()
    print('diff ent dwell:', diff_ent_dwell)
    print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
    print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamp)
    print('pm neg ll dwel:', np.mean(pm_neg_ll_contribs_dwell))
    print('error         :', np.std(pm_neg_ll_contribs_dwell) / sqrt_nsamp)
    #print('v1 neg ll dwel:', np.mean(v1_neg_ll_contribs_dwell))
    #print('error         :', np.std(v1_neg_ll_contribs_dwell) / sqrt_nsamp)
    print('d neg ll dwell:', np.mean(d_neg_ll_contribs_dwell))
    print('error         :', np.std(d_neg_ll_contribs_dwell) / sqrt_nsamp)
    print()
    print('diff ent trans:', diff_ent_trans)
    print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
    print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamp)
    print('pm neg ll tran:', np.mean(pm_neg_ll_contribs_trans))
    print('error         :', np.std(pm_neg_ll_contribs_trans) / sqrt_nsamp)
    #print('v1 neg ll tran:', np.mean(v1_neg_ll_contribs_trans))
    #print('error         :', np.std(v1_neg_ll_contribs_trans) / sqrt_nsamp)
    print('d neg ll trans:', np.mean(d_neg_ll_contribs_trans))
    print('error         :', np.std(d_neg_ll_contribs_trans) / sqrt_nsamp)



#TODO backport into cmedb
def gen_paragraphs(lines):
    para = []
    for line in lines:
        line = line.strip()
        if not line:
            if para:
                yield para
                para = []
        else:
            para.append(line)
    if para:
        yield para


#TODO backport into cmedb
def read_phylip(fin):
    """
    Yield (taxon name, codons) pairs.
    @param fin: file open for reading
    """

    # Get the paragraphs in the most inefficient way possible.
    # Ignore the first line which is also the first paragraph.
    paras = list(gen_paragraphs(fin))[1:]
    if len(paras) != 25:
        raise Exception('expected p53 alignment of 25 taxa')

    # Each paragraph defines a p53 coding sequence of some taxon.
    # The first line gives the taxon name.
    # The rest of the lines are codons.
    for para in paras:
        taxon_name = para[0]
        codons = ' '.join(para[1:]).split()
        if len(codons) != 393:
            raise Exception('expected 393 codons')
        yield taxon_name, codons


#TODO backport this into cmedb
def read_newick(fin):
    """

    Returns
    -------
    T : undirected weighted networkx tree
        Tree with edge weights.
    root_index : integer
        The root node.
    leaf_name_pairs : sequence
        Sequence of (node, name) pairs.

    """
    # use dendropy to read this newick file
    t = dendropy.Tree(stream=fin, schema='newick')
    leaves = t.leaf_nodes()
    nodes = list(t.postorder_node_iter())
    non_leaves = [n for n in nodes if n not in leaves]
    ordered_nodes = leaves + non_leaves
    root_index = len(ordered_nodes) - 1

    # node index lookup
    node_id_to_index = dict((id(n), i) for i, n in enumerate(ordered_nodes))

    # build the networkx tree
    T = nx.Graph()
    edges = list(t.postorder_edge_iter())
    for i, edge in enumerate(edges):
        if edge.head_node and edge.tail_node:
            na = node_id_to_index[id(edge.head_node)]
            nb = node_id_to_index[id(edge.tail_node)]
            T.add_edge(na, nb, weight=edge.length)

    # get a list of (leaf, name) pairs for the table
    leaf_name_pairs = [(i, str(n.taxon)) for i, n in enumerate(leaves)]

    return T, root_index, leaf_name_pairs


# TODO for now, this just tests the PAML likelihood
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
    genetic_code = []
    with open('universal.code.txt') as fin:
        for line in fin:
            line = line.strip()
            if line:
                state, residue, codon = line.split()
                state = int(state)
                residue = residue.upper()
                codon = codon.upper()
                if codon != 'STOP':
                    triple = (state, residue, codon)
                    genetic_code.append(triple)

    # define the primary rate matrix and distribution and tolerance classes
    Q, primary_distn, primary_to_part = create_mg94.create_mg94(
            A_mle, C_mle, G_mle, T_mle,
            kappa_mle, omega_mle, genetic_code,
            target_expected_rate=1.0)
    
    # read the tree with branch lengths estimated by paml
    print('reading the newick tree...')
    with open('codeml.estimated.tree') as fin:
        T, root, leaf_name_pairs = read_newick(fin)

    # read the alignment
    print('reading the alignment...')
    with open('testseq') as fin:
        name_codons_list = list(read_phylip(fin))

    # compute the log likelihood, column by column
    # using _mjp (the sparse Markov jump process module).
    print('preparing to compute log likelihood...')
    nstates = len(genetic_code)
    states = range(nstates)
    codon_to_state = dict((c, s) for s, r, c in genetic_code)
    name_to_leaf = dict((name, leaf) for leaf, name in leaf_name_pairs)
    names, codon_sequences = zip(*name_codons_list)
    codon_columns = zip(*sequences)
    total_log_likelihood = 0
    print('computing log likelihood...')
    for codon_column in codon_columns:
        node_to_allowed_states = dict((node, set(states)) for node in T)
        for name, codon in zip(names, codon_column):
            leaf = name_to_leaf[name]
            codon = codon.upper()
            state = codon_to_state[codon]
            node_to_allowed_states[leaf] = set([state])
        likelihood = _mcy.get_likelihood(
                T, node_to_allowed_states, root,
                root_distn=primary_distn, Q_default=Q)
        log_likelihood = np.log(likelihood)
        total_log_likelihood += log_likelihood
    
    # print the total log likelihood
    print('total log likelihood:', total_log_likelihood)


if __name__ == '__main__':
    main()

