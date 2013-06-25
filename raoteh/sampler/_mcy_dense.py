"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
Transition probability matrices may be provided per edge,
and a default transition matrix may be provided which will be used
if an edge-specific transition matrix is not available.
The name of this module
is derived from "m"arkov "c"hain observation type "y",
where the observation type y refers to one of the
ways to deal with partial observations of the process state.
Type y uses a map from nodes in the network
to sets of states which are allowed according to observations.
Nodes missing from the map will be assumed to be unrestricted.
Nodes in the map which do not correspond to nodes on the tree
will be silently ignored.

The word 'dense' in the function name refers to transition matrix density;
this module uses dense numpy transition matrices
for simpler interfacing with cython, whereas the non-dense
module uses sparse transition matrices based on networkx.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

import pyfelscore

from raoteh.sampler import _mc0, _util



__all__ = []


def _check_P(P):
    if P is None:
        raise ValueError('the transition matrix for this edge is None')
    try:
        if len(P.shape) != 2:
            raise ValueError('expected len(P.shape) == 2')
    except AttributeError as e:
        try:
            nnodes = P.number_of_nodes()
            raise ValueError('expected an ndarray but found a networkx graph')
        except AttributeError as e:
            raise ValueError('expected an ndarray')
    if P.shape[0] != P.shape[1]:
        raise ValueError('expected the array to be square')


#TODO move this function to a less specific module
def _digraph_to_bool_csr(G, ordered_nodes):
    """
    This is a helper function for converting between networkx and cython.

    The output consists of two out of the three arrays of the csr interface.
    The third csr array (data) is not needed
    because we only care about the boolean sparsity structure.

    Parameters
    ----------
    G : networkx directed graph
        The unweighted graph to convert into csr form.
    ordered_nodes : sequence of nodes
        Nodes listed in a requested order.

    Returns
    -------
    csr_indices : ndarray of indices
        Part of the csr interface.
    csr_indptr : ndarray of pointers
        Part of the csr interface.

    """
    node_to_index = dict((n, i) for i, n in enumerate(ordered_nodes))
    csr_indices = []
    csr_indptr = [0]
    node_count = 0
    for na_index, na in enumerate(ordered_nodes):
        if na in G:
            for nb in G[na]:
                nb_index = node_to_index[nb]
                csr_indices.append(nb_index)
                node_count += 1
        csr_indptr.append(node_count)
    csr_indices = np.array(csr_indices, dtype=int)
    csr_indptr = np.array(csr_indptr, dtype=int)
    return csr_indices, csr_indptr


def _get_esd_transitions(G, preorder_nodes, nstates, P_default=None):
    """
    Construct the edge-specific transition matrix as an ndim-3 numpy array.
    """
    nnodes = len(preorder_nodes)
    node_to_index = dict((n, i) for i, n in enumerate(preorder_nodes))
    esd_transitions = np.zeros((nnodes, nstates, nstates), dtype=float)
    for na_index, na in enumerate(preorder_nodes):
        if na in G:
            for nb in G[na]:
                nb_index = node_to_index[nb]
                edge_object = G[na][nb]
                P = edge_object.get('P', P_default)
                if P is None:
                    raise ValueError('expected either a default '
                            'transition matrix '
                            'or a transition matrix on the edge '
                            'from node {0} to node {1}'.format(na, nb))
                esd_transitions[nb_index] = P
    return esd_transitions


def _define_state_mask(node_to_allowed_states, preorder_nodes, nstates):
    """
    Define the state mask.
    """
    nnodes = len(preorder_nodes)
    all_states = set(range(nstates))
    state_mask = np.ones((nnodes, nstates), dtype=int)
    if node_to_allowed_states is not None:
        for na_index, na in enumerate(preorder_nodes):
            for sa in all_states - node_to_allowed_states[na]:
                state_mask[na_index, sa] = 0
    return state_mask

    
# TODO is this obsolete
def _state_mask_to_dict(state_mask, preorder_nodes, nstates):
    """
    Convert the updated state mask into a node_to_set dict.
    """
    node_to_set = {}
    for na_index, na in enumerate(preorder_nodes):
        allowed_states = set()
        for sa_index, sa in enumerate(sorted_states):
            if state_mask[na_index, sa_index]:
                allowed_states.add(sa)
        node_to_set[na] = allowed_states
    return node_to_set


def _esd_get_node_to_pmap(T, root,
        node_to_allowed_states=None, P_default=None):
    # Construct the bfs tree, preserving transition matrices on the edges.
    T_bfs = nx.DiGraph()
    for na, nb in nx.bfs_edges(T, root):
        T_bfs.add_edge(na, nb)
        edge_object = T[na][nb]
        P = edge_object.get('P', None)
        if P is not None:
            T_bfs[na][nb]['P'] = P

    # Get the ordered list of nodes in preorder.
    preorder_nodes = list(nx.dfs_preorder_nodes(T, root))

    # Put the tree into sparse boolean csr form.
    tree_csr_indices, tree_csr_indptr = _digraph_to_bool_csr(
            T_bfs, preorder_nodes)

    # Define the state mask.
    state_mask = _define_state_mask(
            node_to_allowed_states, preorder_nodes, nstates)

    # Construct the edge-specific transition matrix as an ndim-3 numpy array.
    esd_transitions = _get_esd_transitions(
            T_bfs, preorder_nodes, sorted_states, P_default=P_default)

    # Backward pass to update the state mask.
    pyfelscore.mcy_esd_get_node_to_pset(
            tree_csr_indices,
            tree_csr_indptr,
            esd_transitions,
            state_mask)

    # Forward pass to update the state mask.
    pyfelscore.esd_get_node_to_set(
            tree_csr_indices,
            tree_csr_indptr,
            esd_transitions,
            state_mask)

    # Check for agreement on the set of possible states for each node.
    if node_to_set is not None:
        my_node_to_set = _state_mask_to_dict(
                state_mask, preorder_nodes, sorted_states)
        if my_node_to_set != node_to_set:
            msg = 'internal error %s %s' % (my_node_to_set, node_to_set)
            raise Exception(msg)

    # Backward pass to get partial probabilities.
    nnodes = len(preorder_nodes)
    nstates = len(sorted_states)
    subtree_probability = np.empty((nnodes, nstates), dtype=float)
    pyfelscore.mcy_esd_get_node_to_pmap(
            tree_csr_indices,
            tree_csr_indptr,
            esd_transitions,
            state_mask,
            subtree_probability)

    # Convert the subtree probability ndarray to node_to_pmap.
    node_to_index = dict((n, i) for i, n in enumerate(preorder_nodes))
    state_to_index = dict((s, i) for i, s in enumerate(sorted_states))
    node_to_pmap = {}
    for na_index, na in enumerate(preorder_nodes):
        allowed_states = set()
        for sa_index, sa in enumerate(sorted_states):
            if state_mask[na_index, sa_index]:
                allowed_states.add(sa)
        pmap = {}
        for sa_index, sa in enumerate(sorted_states):
            if sa in allowed_states:
                pmap[sa] = subtree_probability[na_index, sa_index]
        node_to_pmap[na] = pmap

    # Return the state mask and the node_to_pmap dict.
    return state_mask, node_to_pmap


def get_node_to_pmap(T, root,
        node_to_allowed_states=None, P_default=None, node_to_set=None):
    """
    For each node, construct the map from state to subtree likelihood.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
        If the map is None then the sets of allowed states are assumed
        to be unrestricted by observations.
        Similarly, if a node is missing from this map
        then its set of allowed states is assumed to be unrestricted.
        Entries of this map that correspond to nodes not in the tree
        will be silently ignored.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.
    node_to_set : dict, optional
        Maps nodes to possible states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    if len(T) == 1 and P_default is not None:
        _util._check_root(T, root)
        allowed_states = set(P_default)
        if node_to_allowed_states is not None:
            allowed_states &= node_to_allowed_states[root]
        root_pmap = dict((s, 1.0) for s in allowed_states)
        node_to_pmap = {root : root_pmap}
    else:
        node_to_pmap = _esd_get_node_to_pmap(T, root,
                node_to_allowed_states=node_to_allowed_states,
                P_default=P_default,
                node_to_set=node_to_set)
    return node_to_pmap


#TODO use this for unit-testing
def unaccelerated_get_node_to_pmap(T, root,
        node_to_allowed_states=None, P_default=None, node_to_set=None):
    """
    For each node, construct the map from state to subtree likelihood.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
        If the map is None then the sets of allowed states are assumed
        to be unrestricted by observations.
        Similarly, if a node is missing from this map
        then its set of allowed states is assumed to be unrestricted.
        Entries of this map that correspond to nodes not in the tree
        will be silently ignored.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.
    node_to_set : dict, optional
        Maps nodes to possible states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    # Get the possible states for each node,
    # after accounting for the rooted tree shape
    # and the edge-specific transition matrix sparsity patterns
    # and the observed states.
    if node_to_set is None:
        node_to_set = get_node_to_set(T, root,
                node_to_allowed_states=node_to_allowed_states,
                P_default=P_default)

    # Bookkeeping.
    successors = nx.dfs_successors(T, root)

    # For each node, get a sparse map from state to subtree likelihood.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):

        # Build the pmap.
        pmap = {}
        for node_state in node_to_set[node]:

            # Add the subtree likelihood to the node state pmap.
            cprob = 1.0
            for n in successors.get(node, []):
                P = T[node][n].get('P', P_default)
                nprob = 0.0
                allowed_states = set(P[node_state]) & set(node_to_pmap[n])
                if not allowed_states:
                    raise ValueError('internal error')
                for s in allowed_states:
                    a = P[node_state][s]['weight']
                    b = node_to_pmap[n][s]
                    nprob += a * b
                cprob *= nprob
            pmap[node_state] = cprob

        # Add the map from state to subtree likelihood.
        node_to_pmap[node] = pmap

    # Return the map from node to the map from state to subtree likelihood.
    return node_to_pmap


def get_likelihood(T, root,
        node_to_allowed_states=None, root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
        If the map is None then the sets of allowed states are assumed
        to be unrestricted by observations.
        Similarly, if a node is missing from this map
        then its set of allowed states is assumed to be unrestricted.
        Entries of this map that correspond to nodes not in the tree
        will be silently ignored.
    root_distn : dict, optional
        A sparse finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.
    P_default : directed weighted networkx graph, optional
        If an edge is not annotated with a transition matrix P,
        then this default transition matrix will be used.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    # If the tree has no edges then treat this as a special case.
    # Otherwise, get likelihoods conditional on the root state
    # and eturn the likelihood.
    if len(T) == 1:
        if _util.get_first_element(T) != root:
            raise Exception('the tree has only a single node, '
                    'but this node is not the root')
        allowed_states = node_to_allowed_states[root]
        if not allowed_states:
            raise _util.StructuralZeroProb('the tree has only a single node, '
                    'and no state is allowed for the root')
        if root_distn is None:
            return 1
        else:
            nonzero_prob_states = allowed_states & set(root_distn)
            if not nonzero_prob_states:
                raise _util.StructuralZeroProb('the tree has only '
                        'a single node, and every state with positive '
                        'prior probability at the root is disallowed '
                        'by a node state constraint')
            return sum(root_distn[s] for s in nonzero_prob_states)
    else:
        node_to_pmap = get_node_to_pmap(T, root,
                node_to_allowed_states=node_to_allowed_states,
                P_default=P_default)
        root_pmap = node_to_pmap[root]
        return _mc0.get_likelihood(root_pmap, root_distn=root_distn)

