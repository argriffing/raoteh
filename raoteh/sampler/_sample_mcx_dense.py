"""
Sample Markov chain trajectories on trees.

This module uses a particularly simple observation type.
This is a thin wrapper that uses _mcx_dense to get the node_to_pmap from
the observation or constraint data and uses _sample_mc0_dense to sample
the joint states using the node_to_pmap.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from raoteh.sampler import (
        _util, _graph_transform,
        _mcx_dense, _sample_mc0_dense,
        )


__all__ = []


def get_feasible_history(T, node_to_state, nstates,
        root=None, root_distn=None, P_default=None):
    """
    Find an arbitrary feasible history.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    node_to_state : dict
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    nstates : integer
        Number of states.
    root : integer, optional
        Root of the tree.
    root_distn : 1d ndarray, optional
        Array giving a probability or general weight per root state.
    P_default : 2d ndarray, optional
        A dense transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.

    Returns
    -------
    feasible_history : weighted undirected networkx graph
        A feasible history as a networkx graph.
        The format is similar to that of the input tree,
        except for a couple of differences.
        Additional degree-two vertices have been added at the points
        at which the state has changed along a branch.
        Each edge is annotated not only by the 'weight'
        that defines its length, but also by the 'state'
        which is constant along each edge.

    Notes
    -----
    The returned history is not sampled according to any particularly
    meaningful distribution.
    It is up to the caller to remove redundant self-transitions.

    """
    # Check that nodes and states are integers.
    for node, state in node_to_state.items():
        fail = False
        try:
            if int(node) != node or int(state) != state:
                fail = True
        except TypeError:
            fail = True
        if fail:
            raise ValueError(
                    'expected nodes and states in the node_to_state map '
                    'to be integers but instead found node_to_state: ' + str(
                        node_to_state))

    # If the root has not been specified,
    # pick a root with known state if any exist,
    # and pick an arbitrary one otherwise.
    if root is None:
        if node_to_state:
            root = get_first_element(node_to_state)
        else:
            root = get_first_element(T)

    # Bookkeeping.
    non_event_nodes = set(T)

    # Repeatedly split edges until no structural error is raised.
    events_per_edge = 0
    k = None
    while True:

        # If the number of events per edge is already as large
        # as the number of states, then no feasible solution exists.
        # For this conclusion to be valid,
        # self-transitions (as in uniformization) must be allowed,
        # otherwise strange things can happen because of periodicity.
        if events_per_edge > nstates:
            raise Exception('failed to find a feasible history')

        # Increment some stuff and bisect edges if appropriate.
        if k is None:
            k = 0
        else:
            T = _graph_transform.get_edge_bisected_graph(T)
            events_per_edge += 2**k
            k += 1

        # Get the event nodes.
        event_nodes = set(T) - non_event_nodes

        # Try to sample edge states.
        try:
            return resample_edge_states(T, root, event_nodes, nstates,
                    node_to_state=node_to_state,
                    root_distn=root_distn, P_default=P_default)
        except _util.StructuralZeroProb as e:
            pass


def resample_states(T, root, nstates,
        node_to_state=None, root_distn=None, P_default=None):
    """
    This function applies to a tree for which nodes will be assigned states.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
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
    # Get the map from each node to a sparse map
    # from each feasible state to the subtree likelihood.
    node_to_pmap = _mcx_dense.get_node_to_pmap(T, root, nstates,
            node_to_state=node_to_state, P_default=P_default)

    # Use the generic sampler.
    return _sample_mc0_dense.resample_states(T, root, node_to_pmap, nstates,
            root_distn=root_distn, P_default=P_default)


def resample_edge_states(T, root, event_nodes, nstates,
        node_to_state=None, root_distn=None, P_default=None):
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
    event_nodes : set of integers
        States of edges adjacent to these nodes are allowed to not be the same
        as each other.  For nodes that are not event nodes,
        the adjacent edge states must all be the same as each other.
    nstates : integer
        Number of states.
    node_to_state : dict, optional
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    root_distn : 1d ndarray, optional
        A dense finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.
    P_default : 2d ndarray, optional
        A dense transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.

    Returns
    -------
    T_out : weighted undirected networkx graph
        A copy of the original tree, with each edge annoted with a state.

    """
    # Get the set of non event nodes.
    non_event_nodes = set(T) - set(event_nodes)

    # Require that every node with known state is a non-event node.
    bad = set(node_to_state) - set(non_event_nodes)
    if bad:
        raise ValueError('found nodes with known state which are not '
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

    # If some of the original nodes have a known state,
    # then this state will propagate to the containing chunk node.
    # If a chunk node contains nodes of the original tree which have
    # conflicting known states, then a StructuralZeroProb exception is raised.
    if node_to_state is None:
        chunk_node_to_state = None
    else:
        chunk_node_to_state = {}
        for edge in nx.bfs_edges(T, root):
            na, nb = edge
            chunk_node = edge_to_chunk_node[edge]
            for n in set(node_to_state) & {na, nb}:
                state = node_to_state[n]
                chunk_state = chunk_node_to_state.get(chunk_node, state)
                if state != chunk_state:
                    raise _util.StructuralZeroProb(
                            'found conflicting state assignments '
                            'for a tree region delimited by event nodes')
                chunk_node_to_state[chunk_node] = state

    # Sample the states.
    chunk_node_to_sampled_state = resample_states(chunk_tree, root, nstates,
            node_to_state=chunk_node_to_state,
            root_distn=root_distn, P_default=P_default)

    # Construct a copy of the original tree, map states onto its edges,
    # and return the tree.
    T_out = T.copy()
    for edge in nx.bfs_edges(T, root):
        na, nb = edge
        chunk_node = edge_to_chunk_node[edge]
        T_out[na][nb]['state'] = chunk_node_to_sampled_state[chunk_node]
    return T_out

