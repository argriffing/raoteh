"""
Sample Markov chain trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mc
from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)


__all__ = []


def resample_states(T, P, node_to_state, root=None, root_distn=None):
    """
    This function applies to a tree for which nodes will be assigned states.

    Parameters
    ----------
    T : unweighted undirected acyclic networkx graph
        States do not change within chunks represented by nodes in this tree.
    P : weighted directed networkx graph
        A sparse transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.
    node_to_state : dict
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    root : integer, optional
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.

    Returns
    -------
    node_to_sampled_state : dict
        A map from each node of T to its state.
        If the state was not defined by the node_to_state argument,
        then the state will have been sampled.

    Notes
    -----
    Nodes other than tips of the tree may have known states.
    If no root is provided,
    then an arbitrary node with a known state will be chosen as the root.

    """

    # Check for lack of any known state.
    if not node_to_state:
        raise NotImplementedError(
                'unconditional forward simulation is not implemented')

    # If the root has not been provided, then pick one with a known state.
    if root is None:
        root = get_first_element(node_to_state)

    # Bookkeeping structure related to tree traversal.
    predecessors = nx.dfs_predecessors(T, root)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = _mc.construct_node_to_pmap(T, P, node_to_state, root)

    # Sample the node states, beginning at the root.
    node_to_sampled_state = {}

    # Treat the root separately.
    # If only one state is possible at the root, then we do not have to sample.
    # Otherwise consult the map from root states to probabilities.
    root_pmap = node_to_pmap[root]
    if not root_pmap:
        raise StructuralZeroProb('no state is feasible at the root')
    elif len(root_pmap) == 1:
        root_state = get_first_element(root_pmap)
        if not root_pmap[root_state]:
            raise NumericalZeroProb('numerical problem at the root')
    else:
        if root_distn is None:
            raise ValueError(
                    'expected a prior distribution over the '
                    '%d possible states at the root' % len(root_pmap))
        posterior_distn = _mc.get_zero_step_posterior_distn(
                root_distn, root_pmap)
        states, probs = zip(*posterior_distn.items())
        root_state = np.random.choice(states, p=np.array(probs))
    node_to_sampled_state[root] = root_state

    # Sample the states at the rest of the nodes.
    for node in nx.dfs_preorder_nodes(T, root):

        # The root has already been sampled.
        if node == root:
            continue

        # Get the parent node and its state.
        parent_node = predecessors[node]
        parent_state = node_to_sampled_state[parent_node]
        
        # Check that the parent state has transitions.
        if parent_state not in P:
            raise StructuralZeroProb(
                    'no transition from the parent state is possible')

        # Sample the state of a non-root node.
        # A state is possible if it is reachable in one step from the
        # parent state which has already been sampled
        # and if it gives a subtree probability that is not structurally zero.
        sinks = set(P[parent_state])
        prior_distn = dict((s, P[parent_state][s]['weight']) for s in sinks)
        posterior_distn = _mc.get_zero_step_posterior_distn(
                prior_distn, node_to_pmap[node])
        states, probs = zip(*posterior_distn.items())
        sampled_state = np.random.choice(states, p=np.array(probs))
        node_to_sampled_state[node] = sampled_state

    # Return the map of sampled states.
    return node_to_sampled_state


def resample_edge_states(T, P, node_to_state, event_nodes,
        root=None, root_distn=None):
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
    P : weighted directed networkx graph
        A sparse transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.
    node_to_state : dict
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    event_nodes : set of integers
        States of edges adjacent to these nodes are allowed to not be the same
        as each other.  For nodes that are not event nodes,
        the adjacent edge states must all be the same as each other.
    root : integer, optional
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.

    Returns
    -------
    T_out : weighted undirected networkx graph
        A copy of the original tree, with each edge annoted with a state.

    """

    # Bookkeeping.
    non_event_nodes = set(T) - event_nodes

    # Input validation.
    if set(event_nodes) & set(node_to_state):
        raise ValueError('event nodes cannot have known states')
    if root is not None:
        if root not in non_event_nodes:
            raise ValueError('the root must be a non-event node')

    # Construct the chunk tree using the provided root.
    chunk_tree, non_event_map, event_map = _graph_transform.get_chunk_tree(
            T, event_nodes, root=root)

    # Try to map known states onto chunk tree nodes.
    # This may fail if multiple nodes with different known states
    # map onto the same chunk tree node,
    # which may happen if some edges have no uniformized events.
    chunk_node_to_known_state = {}
    for node, state in node_to_state.items():
        chunk_node = non_event_map[node]
        if chunk_node in chunk_node_to_known_state:
            if chunk_node_to_known_state[chunk_node] != state:
                raise StructuralZeroProb('chunk state collision')
        else:
            chunk_node_to_known_state[chunk_node] = state

    # Try to sample states on the chunk tree.
    # This may fail if not enough uniformized events have been placed
    # on the edges.
    if root is not None:
        chunk_root = non_event_map[root]
        chunk_root_distn = root_distn
    else:
        chunk_root = None
        chunk_root_distn = None
    chunk_node_to_sampled_state = resample_states(
            chunk_tree, P, chunk_node_to_known_state,
            root=chunk_root, root_distn=chunk_root_distn)

    # Copy the original tree before adding states to the edges.
    T = T.copy()

    # Map states onto edges of the tree.
    for a, b in T.edges():
        if a in node_to_state:
            T[a][b]['state'] = node_to_state[a]
        elif b in node_to_state:
            T[a][b]['state'] = node_to_state[b]
        else:
            if a in non_event_map:
                chunk_node = non_event_map[a]
                state = chunk_node_to_sampled_state[chunk_node]
            elif b in non_event_map:
                chunk_node = non_event_map[b]
                state = chunk_node_to_sampled_state[chunk_node]
            else:
                a_chunks = set(event_map[a])
                b_chunks = set(event_map[b])
                chunk_nodes = list(a_chunks & b_chunks)
                if len(chunk_nodes) != 1:
                    raise Exception('internal error')
                chunk_node = chunk_nodes[0]
                state = chunk_node_to_sampled_state[chunk_node]
            T[a][b]['state'] = state

    # Return the new tree.
    return T

