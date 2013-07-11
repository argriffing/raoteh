"""
Sample Markov chain trajectories on trees.

This module allows intermediately complicated observations.
This is a thin wrapper that uses _mcy_dense to get the node_to_pmap from
the observation or constraint data and uses _sample_mc0 to sample
the joint states using the node_to_pmap.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np

from raoteh.sampler import (
        _util, _graph_transform,
        _mc0_dense, mcy_dense, _sample_mc0_dense,
        )


__all__ = []

def resample_states(T, root, nstates,
        node_to_allowed_states=None, root_distn=None, P_default=None):
    """
    This function applies to a tree for which nodes will be assigned states.

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
    nstates : integer
        Number of states.
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
    node_to_sampled_state : dict
        A map from each node of T to its state.
        If the state was not defined by the node_to_state argument,
        then the state will have been sampled.

    """
    # Get the map from each node to a dense array
    # defining the subtree likelihood for each state.
    node_to_pmap = _mcy_dense.get_node_to_pmap(T, root, nstates,
            node_to_allowed_states=node_to_allowed_states,
            P_default=P_default)

    # Use the generic sampler.
    return _sample_mc0_dense.resample_states(T, root, node_to_pmap, nstates,
            root_distn=root_distn, P_default=P_default)


def resample_edge_states(T, root, P, event_nodes, nstates,
        node_to_allowed_states=None, root_distn=None):
    """
    This function applies to a tree for which edges will be assigned states.

    After the edge states have been sampled,
    some of the nodes will possibly have become redundant.
    If this is the case, it is the responsibility of the caller
    to remove these redundant nodes.
    This function should not modify the original tree.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.
    P : 2d ndarray
        A transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.
    event_nodes : set of integers
        States of edges adjacent to these nodes are allowed to not be the same
        as each other.  For nodes that are not event nodes,
        the adjacent edge states must all be the same as each other.
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
    root_distn : 1d ndarray, optional
        A dense finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.

    Returns
    -------
    T_out : weighted undirected networkx graph
        A copy of the original tree, with each edge annoted with a state.

    """
    # Get the set of non event nodes.
    non_event_nodes = set(T) - set(event_nodes)

    # Require that every node with known state is a non-event node.
    bad = set(node_to_allowed_states) - set(non_event_nodes)
    if bad:
        raise ValueError('found nodes with restricted states which are not '
                'non-event nodes: ' + str(sorted(bad)))

    # Check the root.
    if root is None:
        raise ValueError('the root must be provided')
    if root not in T:
        raise ValueError('the root must be a node in the tree')
    if root in event_nodes:
        raise ValueError('the root cannot be an event node')

    # Construct the chunk tree using the provided root.
    info = _graph_transform.get_chunk_tree_type_b(T, root, event_nodes)
    chunk_tree, edge_to_chunk_node, event_node_to_chunk_edge = info

    # Propagate the state restrictions of nodes
    # to state restrictions of their containing chunk nodes.
    if node_to_allowed_states is None:
        chunk_node_to_allowed_states = None
    else:
        chunk_node_to_allowed_states = {}
        for edge in nx.bfs_edges(T, root):
            na, nb = edge
            cnode = edge_to_chunk_node[edge]
            for n in set(node_to_allowed_states) & {na, nb}:

                # A node in the chunk has a restriction on its allowed states.
                # If the chunk node currently has no restriction,
                # then set the chunk node state restriction
                # to the restriction at this node.
                allowed_states = node_to_allowed_states[n]
                if cnode not in chunk_node_to_allowed_states:
                    chunk_node_to_allowed_states[cnode] = set(allowed_states)
                else:
                    cnode_allowed_states = chunk_node_to_allowed_states[cnode]
                    cnode_allowed_states.intersection_update(allowed_states)

    # Check if any chunk node has an empty set of allowed states.
    if chunk_node_to_allowed_states is not None:
        if not all(chunk_node_to_allowed_states.values()):
            raise _util.StructuralZeroProb('a region has no allowed states')

    # Sample the states.
    chunk_node_to_sampled_state = resample_states(chunk_tree, root, nstates,
            node_to_allowed_states=chunk_node_to_allowed_states,
            root_distn=root_distn, P_default=P)

    # Construct a copy of the original tree, map states onto its edges,
    # and return the tree.
    T_out = T.copy()
    for edge in nx.bfs_edges(T, root):
        na, nb = edge
        chunk_node = edge_to_chunk_node[edge]
        T_out[na][nb]['state'] = chunk_node_to_sampled_state[chunk_node]
    return T_out

