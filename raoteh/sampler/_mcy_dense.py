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

from raoteh.sampler._density import (
        check_square_dense,
        digraph_to_bool_csr,
        get_esd_transitions,
        )

__all__ = []



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


def _esd_get_node_to_pmap(T, root, nstates,
        node_to_allowed_states=None, P_default=None):
    """
    
    Returns
    -------
    node_to_pmap : dict
        Map from node to 1d ndarray.
        The ndarray associated with each node gives the subtree likelihood
        conditional on each state of the given node.

    """
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
    tree_csr_indices, tree_csr_indptr = digraph_to_bool_csr(
            T_bfs, preorder_nodes)

    # Define the state mask.
    state_mask = _define_state_mask(
            node_to_allowed_states, preorder_nodes, nstates)

    # Construct the edge-specific transition matrix as an ndim-3 numpy array.
    esd_transitions = get_esd_transitions(
            T_bfs, preorder_nodes, nstates, P_default=P_default)

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

    # Backward pass to get partial probabilities.
    nnodes = len(preorder_nodes)
    subtree_probability = np.empty((nnodes, nstates), dtype=float)
    pyfelscore.mcy_esd_get_node_to_pmap(
            tree_csr_indices,
            tree_csr_indptr,
            esd_transitions,
            state_mask,
            subtree_probability)

    # Convert the subtree probability ndarray to node_to_pmap.
    node_to_index = dict((n, i) for i, n in enumerate(preorder_nodes))
    node_to_pmap = {}
    for na_index, na in enumerate(preorder_nodes):
        allowed_states = set()
        for sa in range(nstates):
            if state_mask[na_index, sa]:
                allowed_states.add(sa)
        pmap = np.zeros(nstates, dtype=float)
        for sa in range(nstates):
            if sa in allowed_states:
                pmap[sa] = subtree_probability[na_index, sa]
        node_to_pmap[na] = pmap

    # Return the state mask and the node_to_pmap dict.
    return state_mask, node_to_pmap


def get_node_to_pmap(T, root, nstates,
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
    nstates : integer
        Number of states.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
        If the map is None then the sets of allowed states are assumed
        to be unrestricted by observations.
        Similarly, if a node is missing from this map
        then its set of allowed states is assumed to be unrestricted.
        Entries of this map that correspond to nodes not in the tree
        will be silently ignored.
    P_default : 2d ndarray, optional
        Transition matrix to be used for edges
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
        allowed_states = set(range(nstates))
        if node_to_allowed_states is not None:
            allowed_states &= node_to_allowed_states[root]
        root_pmap = np.array(
                [1 if s in allowed_states else 0 for s in range(nstates)],
                dtype=float)
        node_to_pmap = {root : root_pmap}
    else:
        if node_to_set is not None:
            best_state_restriction = node_to_set
        else:
            best_state_restriction = node_to_allowed_states
        state_mask, node_to_pmap = _esd_get_node_to_pmap(
                T, root, nstates,
                node_to_allowed_states=best_state_restriction,
                P_default=P_default)
    return node_to_pmap


#TODO use this for unit-testing
#TODO this has not been updated for dense matrices
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

