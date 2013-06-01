"""
Sample Markov chain trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mc

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        construct_node_to_pmap,
        )

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_normalized_dict_distn,
        dict_random_choice,
        )

from raoteh.sampler._mc import get_zero_step_posterior_distn


__all__ = []


def get_test_transition_matrix():
    # This returns a sparse transition matrix for testing.
    # It uses an hack for the node indices, because I want to keep integers.
    # This transition graph looks kind of like the following ascii art.
    #
    # 41 --- 42 --- 43 --- 44
    #  |      |      |      |
    # 31 --- 32 --- 33 --- 34
    #  |      |      |      |
    # 21 --- 22 --- 23 --- 24
    #  |      |      |      |
    # 11 --- 12 --- 13 --- 14
    #
    P = nx.DiGraph()
    weighted_edges = []
    for i in (1, 2, 3, 4):
        for j in (1, 2, 3, 4):
            source = i*10 + j
            sinks = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = i + di
                    nj = j + dj
                    if not (di and dj):
                        if (1 <= ni <= 4) and (1 <= nj <= 4):
                            sink = ni*10 + nj
                            sinks.append(sink)
            nsinks = len(sinks)
            weight = 1 / float(nsinks)
            for sink in sinks:
                weighted_edges.append((source, sink, weight))
    P.add_weighted_edges_from(weighted_edges)
    return P


# XXX this is a special case of resample_restricted_states.
def resample_states(T, P, node_to_state, root=None, root_distn=None):
    """
    This function applies to a tree for which nodes will be assigned states.

    It is somewhat obsolete.

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

    See also
    --------
    resample_restricted_states

    """
    # If the root has not been provided, then pick one with a known state.
    # If no state is known then pick an arbitrary root.
    if root is None:
        if node_to_state:
            root = get_first_element(node_to_state)
        else:
            root = get_first_element(T)

    # Construct the set of all possible states.
    all_states = set(P)
    if root_distn is not None:
        all_states.update(root_distn)

    # Construct the map from each node to its set of allowed states.
    node_to_allowed_states = {}
    for node in T:
        if node in node_to_state:
            node_to_allowed_states[node] = {node_to_state[node]}
        else:
            node_to_allowed_states[node] = all_states

    # Sample the state at each node.
    return resample_restricted_states(T, node_to_allowed_states,
            root, root_distn, P_default=P)


def resample_restricted_states(T, node_to_allowed_states,
        root, prior_root_distn=None, P_default=None):
    """
    This function applies to a tree for which nodes will be assigned states.

    Parameters
    ----------
    T : unweighted undirected acyclic networkx graph
        States do not change within chunks represented by nodes in this tree.
        Edges may be annotated with a sparse transition matrix P
        as a weighted directed networkx graph.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    prior_root_distn : dict, optional
        Map from root state to prior probability.
        If the distribution is not provided,
        the it will be assumed to be uninformative.
    P_default : weighted directed networkx graph, optional
        Edges that are not explicitly annotated with a transition matrix P
        will use this default transition matrix instead.
        The edge weights are transition probabilities.

    Returns
    -------
    node_to_sampled_state : dict
        A map from each node of T to its sampled state.

    See also
    --------
    resample_states

    """
    # Check for failure to annotate the nodes.
    bad = set(T) - set(node_to_allowed_states)
    if bad:
        raise ValueError('the following nodes in the tree '
                'are not annotated with allowed states: ' + str(sorted(bad)))

    # Check for annotated non-nodes.
    bad = set(node_to_allowed_states) - set(T)
    if bad:
        raise ValueError('the following nodes are annotated '
                'with allowed states, but these nodes do not appear '
                'in the tree: ' + str(sorted(bad)))

    # Check that the root is in the tree.
    if root not in T:
        raise ValueError('the specified root ' + str(root) + 'is not '
                'a node in the tree')

    # Check for blatant infeasibility.
    bad = set(n for n, s in node_to_allowed_states.items() if not s)
    if bad:
        raise StructuralZeroProb('the following nodes have no allowed states '
                'according to their annotation: ' + str(sorted(bad)))

    # Check for blatant root infeasibility
    # caused partially by sparsity of the prior root distribution.
    if prior_root_distn is not None:
        if not prior_root_distn:
            raise StructuralZeroProb('no root state is feasible '
                    'because no state has a positive prior probability')
        if not set(prior_root_distn) & node_to_allowed_states[root]:
            raise StructuralZeroProb('no root state is feasible '
                    'because the states with positive prior probability '
                    'are disallowed by the annotation')

    # Bookkeeping structure related to tree traversal.
    predecessors = nx.dfs_predecessors(T, root)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = construct_node_to_restricted_pmap(
            T, root, node_to_allowed_states, P_default)

    # Sample the node states, beginning at the root.
    node_to_sampled_state = {}

    # Sample the states.
    for node in nx.dfs_preorder_nodes(T, root):

        # Get the precomputed pmap associated with the node.
        # This is a sparse map from state to subtree likelihood.
        pmap = node_to_pmap[node]

        # Check for infeasibility caused by pmap sparsity.
        if not pmap:
            if node == root:
                raise StructuralZeroProb('no root state is feasible because '
                        'the subtree has zero likelihood for each state')
            else:
                raise StructuralZeroProb('no state is feasible '
                        'for node %s because the subtree has zero likelihood '
                        'for each state' % node)

        # Define the posterior distribution over states at the node.
        if node == root:
            if prior_root_distn is None:
                dpost = get_normalized_dict_distn(pmap)
            else:
                if not set(prior_root_distn) & set(pmap):
                    raise StructuralZeroProb('no root state is feasible '
                            'because the subtree has zero probability '
                            'for each root state with positive prior '
                            'probability')
                dpost = get_zero_step_posterior_distn(
                        prior_root_distn, pmap)
        else:

            # Get the parent node and its state.
            parent_node = predecessors[node]
            parent_state = node_to_sampled_state[parent_node]

            # Get the transition probability matrix.
            P = T[parent_node][node].get('P', P_default)
            if P is None:
                raise ValueError(
                        'no transition probability matrix is available')
            
            # Check that the parent state has transitions.
            if parent_state not in P:
                raise StructuralZeroProb(
                        'no transition from the parent state is possible')

            # Get the distribution of a non-root node.
            sinks = set(P[parent_state]) & node_to_allowed_states[node]
            if not sinks:
                raise StructuralZeroProb(
                        'all allowed transitions from the parent node, '
                        'given its sampled state, lead to disallowed '
                        'states at the current node')
            prior_distn = dict((s, P[parent_state][s]['weight']) for s in sinks)
            if not set(prior_distn) & set(pmap):
                raise StructuralZeroProb(
                        'all transitions from the parent node, '
                        'in its currently sampled state, to an allowed '
                        'state in the current node, lead to only subtrees '
                        'with likelihood zero')
            dpost = get_zero_step_posterior_distn(prior_distn, pmap)

        # Sample the state from the posterior distribution.
        node_to_sampled_state[node] = dict_random_choice(dpost)

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

