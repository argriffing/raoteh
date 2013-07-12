"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
Transition probability matrices may be provided per edge,
and a default transition matrix may be provided which will be used
if an edge-specific transition matrix is not available.
The name of this module
is derived from "m"arkov "c"hain observation type "x",
where the observation type x refers to the least abstract of three
ways to deal with partial observations of the process state.
Type x uses a sparse map from nodes in the network
to their corresponding observed state if any.
Nodes missing from the map are assumed to have completely unobserved state,
while nodes in the map are assumed to have completely observed state.

Here are some more notes regarding the argument node_to_state.
Nodes in this map are assumed to have completely known state.
Nodes not in this map are assumed to have completely missing state.
If this map is not provided,
all states information will be assumed to be completely missing.
Entries of this dict that correspond to nodes not in the tree
will be silently ignored.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

import pyfelscore

from raoteh.sampler import _util, _mc0, _mc0_dense

from raoteh.sampler._density import (
        check_square_dense,
        digraph_to_bool_csr,
        get_esd_transitions,
        )


__all__ = []


#TODO under construction


def _define_state_mask(preorder_nodes, nstates, node_to_state=None):
    """
    Define the state mask.

    Parameters
    ----------
    preorder_nodes : x
        x
    nstates : integer
        Number of states.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.

    Returns
    -------
    state_mask : 2d ndarray
        For each (node, state) indicate whether the given
        state is allowed for the given node.

    """
    nnodes = len(preorder_nodes)
    if node_to_state is None:
        state_mask = np.ones((nnodes, nstates), dtype=int)
    else:
        state_mask = np.empty((nnodes, nstates), dtype=int)
        for na_index, na in enumerate(preorder_nodes):
            if na in node_to_state:
                sa = node_to_state[na]
                state_mask[na_index] = 0
                state_mask[na_index, sa] = 1
            else:
                state_mask[na_index] = 1
    return state_mask


#TODO this is copypasted from _mcy_dense
def _esd_get_node_to_pmap(T, root, nstates,
        node_to_state=None, P_default=None):
    """

    Parameters
    ----------
    T : x
        x
    root : x
        x
    nstates : x
        x
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
    P_default : x, optional
        x
    
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
    state_mask = _define_state_mask(preorder_nodes, nstates, node_to_state)

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


#TODO also copypasted from _mcy_dense
def get_node_to_pmap(T, root, nstates,
        node_to_state=None, P_default=None, node_to_set=None):
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
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
    P_default : 2d ndarray, optional
        Transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.
    node_to_set : dict, optional
        Precomputed map from nodes to possible states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    if node_to_set is not None:
        raise NotImplementedError

    if len(T) == 1:
        _util._check_root(T, root)
        if (node_to_state is not None) and (root in node_to_state):
            root_state = node_to_state[root]
            root_pmap = np.zeros(nstates, dtype=float)
            root_pmap[root_state] = 1
        else:
            root_pmap = np.ones(nstates, dtype=float)
        node_to_pmap = {root : root_pmap}
    else:
        state_mask, node_to_pmap = _esd_get_node_to_pmap(
                T, root, nstates,
                node_to_state=node_to_state, P_default=P_default)
    return node_to_pmap


#TODO also copypasted from _mcy_dense
def get_likelihood(T, root, nstates,
        node_to_state=None, root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    nstates : integer
        The number of states.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
        If the map is None then the sets of allowed states are assumed
        to be unrestricted by observations.
        Similarly, if a node is missing from this map
        then its set of allowed states is assumed to be unrestricted.
        Entries of this map that correspond to nodes not in the tree
        will be silently ignored.
    root_distn : 1d ndarray, optional
        A dense finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.
    P_default : 2d ndarray, optional
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
        _util._check_root(T, root)
        if (node_to_state is not None) and (root in node_to_state):
            root_state = node_to_state[root]
            allowed_states = {root_state}
        else:
            allowed_states = set(range(nstates))
        if not allowed_states:
            raise _util.StructuralZeroProb('the tree has only a single node, '
                    'and no state is allowed for the root')
        if root_distn is None:
            return 1
        else:
            pos_prob_states = set(s for s in allowed_states if root_distn[s])
            if not pos_prob_states:
                raise _util.StructuralZeroProb('the tree has only '
                        'a single node, and every state with positive '
                        'prior probability at the root is disallowed '
                        'by a node state constraint')
            return sum(root_distn[s] for s in pos_prob_states)
    else:
        node_to_pmap = get_node_to_pmap(T, root, nstates,
                node_to_state=node_to_state,
                P_default=P_default)
        root_pmap = node_to_pmap[root]
        return _mc0_dense.get_likelihood(root_pmap, root_distn=root_distn)



#TODO obsolete function
def xxx_get_node_to_pset(T, root, nstates,
        node_to_state=None, P_default=None):
    """
    For each node, get the set of states that give positive subtree likelihood.

    This function is analogous to get_node_to_pmap.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    nstates : integer
        Number of states.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.

    Returns
    -------
    node_to_pset : dict
        A map from a node to the set of states with positive subtree likelihood.

    """
    if len(set(T)) == 1:
        if root not in T:
            raise ValueError('unrecognized root')
        if (node_to_state is not None) and (root in node_to_state):
            root_state = node_to_state[root]
            root_pset = {root_state}
        else:
            all_states = set(P_default)
            root_pset = all_states
        return {root : root_pset}

    # Bookkeeping.
    successors = nx.dfs_successors(T, root)
    predecessors = nx.dfs_predecessors(T, root)

    # Compute the map from node to set.
    node_to_pset = {}
    for nb in nx.dfs_postorder_nodes(T, root):

        # If a parent node is available, get a set of states
        # involved in the transition matrix associated with the parent edge.
        # A more complicated implementation would use only the sink
        # states of that transition matrix.
        na_set = None
        if nb in predecessors:
            na = predecessors[nb]
            P = T[na][nb].get('P', P_default)
            na_set = set(P)

        # If the state of the current state is known,
        # define the set containing only that state.
        nb_set = None
        if nb in node_to_state:
            nb_set = {node_to_state[nb]}

        # If a child node is available, get the set of states
        # that have transition to child states
        # for which the child subtree likelihoods are positive.
        nc_set = None
        if nb in successors:
            for nc in successors[nb]:
                allowed_set = set()
                P = T[nb][nc].get('P', P_default)
                for sb, sc in P.edges():
                    if sc in node_to_pset[nc]:
                        allowed_set.add(sb)
                if nc_set is None:
                    nc_set = allowed_set
                else:
                    nc_set.intersection_update(allowed_set)

        # Take the intersection of informative constraints due to
        # possible parent transitions,
        # possible direct constraints on the node state,
        # and possible child node state constraints.
        pset = None
        for constraint_set in (na_set, nb_set, nc_set):
            if constraint_set is not None:
                if pset is None:
                    pset = constraint_set
                else:
                    pset.intersection_update(constraint_set)

        # This value should not be None unless there has been some problem.
        if pset is None:
            raise ValueError('internal error')

        # Define the pset for the node.
        node_to_pset[nb] = pset

    # Return the node_to_pset map.
    return node_to_pset


#TODO obsolete function
def xxx_get_node_to_pmap(T, root,
        node_to_state=None, P_default=None, node_to_set=None):
    """
    For each node, construct the map from state to subtree likelihood.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
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
        node_to_pset = get_node_to_pset(T, root,
                node_to_state=node_to_state, P_default=P_default)
        node_to_set = _mc0.get_node_to_set(T, root,
                node_to_pset, P_default=P_default)

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


#TODO obsolete function
def xxx_get_likelihood(T, root,
        node_to_state=None, root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
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
    # Get likelihoods conditional on the root state.
    node_to_pmap = get_node_to_pmap(T, root,
            node_to_state=node_to_state, P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Return the likelihood.
    return _mc0.get_likelihood(root_pmap, root_distn=root_distn)

