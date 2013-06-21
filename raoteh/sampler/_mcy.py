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

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

import pyfelscore

from raoteh.sampler import _mc0


__all__ = []


def _digraph_to_csr(G, ordered_nodes):
    """

    Parameters
    ----------
    G : networkx directed graph
        The unweighted graph to convert into csr form.
    ordered_nodes : sequence of nodes
        Nodes listed in a requested order.

    Returns
    -------
    csr_indices : sequence of indices
        Part of the csr interface.
    csr_indptr : sequence of pointers
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
    return csr_indices, csr_indptr


def _get_node_to_set_same_transition_matrix(T, root, P,
        node_to_allowed_states=None):
    pass



def _get_node_to_pset_same_transition_matrix(T, root, P,
        node_to_allowed_states=None):
    T_bfs = nx.bfs_tree(T, root)
    preorder_nodes = list(nx.dfs_preorder_nodes(T, root))
    sorted_states = sorted(P)
    nnodes = len(preorder_nodes)
    nstates = len(sorted_states)
    n_to_canon = dict((n, i) for i, n in enumerate(preorder_nodes))
    s_to_canon = dict((s, i) for i, s in enumerate(sorted_states))

    # Put the tree into sparse boolean csr form.
    tree_csr_indices, tree_csr_indptr = _digraph_to_csr(T_bfs, preorder_nodes)

    # Put the transition matrix into sparse boolean csr form.
    trans_csr_indices, trans_csr_indptr = _digraph_to_csr(P, sorted_states)

    # Define the state mask.
    state_mask = np.ones((nnodes, nstates), dtype=int)
    if node_to_allowed_states is not None:
        for na_canon, na in enumerate(preorder_nodes):
            if na in node_to_allowed_states:
                allowed_states = node_to_allowed_states[na]
                for sa_canon, sa in enumerate(sorted_states):
                    if sa not in allowed_states:
                        state_mask[na_canon, sa_canon] = 0

    # Update the state mask.
    pyfelscore.mcy_get_node_to_pset(
            np.array(tree_csr_indices, dtype=int),
            np.array(tree_csr_indptr, dtype=int),
            np.array(trans_csr_indices, dtype=int),
            np.array(trans_csr_indptr, dtype=int),
            state_mask)
    
    # Convert the updated state mask into a node_to_pset dict.
    node_to_pset = {}
    for na_canon, na in enumerate(preorder_nodes):
        allowed_states = set()
        for sa_canon, sa in enumerate(sorted_states):
            if state_mask[na_canon, sa_canon]:
                allowed_states.add(sa)
        node_to_pset[na] = allowed_states

    # Return the node_to_pset dict.
    return node_to_pset


def get_node_to_pset(T, root,
        node_to_allowed_states=None, P_default=None):
    """
    For each node, get the set of states that give positive subtree likelihood.

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

    Returns
    -------
    node_to_pset : dict
        A map from a node to the set of states with positive subtree likelihood.

    """
    # Use a specialized function if P_default is available.
    has_custom_P = any('P' in T[na][nb] for na, nb in T.edges())
    if (P_default is not None) and (not has_custom_P):
        node_to_pset = _get_node_to_pset_same_transition_matrix(
                T, root, P_default,
                node_to_allowed_states=node_to_allowed_states)
        return node_to_pset

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

        # Use the set of allowed states for the current node,
        # if it is known.
        nb_set = None
        if nb in node_to_allowed_states:
            nb_set = set(node_to_allowed_states[nb])

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

        # If the pset is still None,
        # then as a last attempt to get a node set,
        # try using the states in P_default if it is available.
        if pset is None:
            if P_default is not None:
                pset = set(P_default)

        # This value should not be None unless there has been some problem.
        if pset is None:
            raise ValueError('internal error')

        # Define the pset for the node.
        node_to_pset[nb] = pset

    # Return the node_to_pset map.
    return node_to_pset


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
    # Get the possible states for each node,
    # after accounting for the rooted tree shape
    # and the edge-specific transition matrix sparsity patterns
    # and the observed states.
    if node_to_set is None:
        node_to_pset = get_node_to_pset(T, root,
                node_to_allowed_states=node_to_allowed_states,
                P_default=P_default)
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
    # Get likelihoods conditional on the root state.
    node_to_pmap = get_node_to_pmap(T, root,
            node_to_allowed_states=node_to_allowed_states,
            P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Return the likelihood.
    return _mc0.get_likelihood(root_pmap, root_distn=root_distn)

