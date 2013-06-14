"""
Sample Markov chain trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler import _sample_mcx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element,
        get_normalized_dict_distn,
        get_unnormalized_dict_distn,
        dict_random_choice,
        )

from raoteh.sampler._graph_transform import (
        get_chunk_tree,
        )

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        )


__all__ = []


#TODO this is obsolete; use _mcy.resample_states instead
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
            prior_distn = prior_root_distn
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

        # Sample the state from the posterior distribution.
        dpost_unnormal = get_unnormalized_dict_distn(pmap, prior_distn)
        node_to_sampled_state[node] = dict_random_choice(dpost_unnormal)

    # Return the map of sampled states.
    return node_to_sampled_state


#TODO use a function in _sample_mcx instead
def xxx_resample_edge_states(T, P, node_to_state, event_nodes,
        root=None, root_distn=None):
    """
    This function applies to a tree for which edges will be assigned states.

    It is kind of obsolete.
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
    # Get the set of non event nodes.
    non_event_nodes = set(T) - set(event_nodes)

    # Require that every node with known state is a non-event node.
    bad = set(node_to_state) - set(non_event_nodes)
    if bad:
        raise ValueError('found nodes with known state which are not '
                'non-event nodes: ' + str(sorted(bad)))

    # If a root has not been provided,
    # then use a root that has known state if possible.
    # Otherwise use a random non-event node in the tree.
    if root is None:
        if node_to_state:
            root = get_first_element(node_to_state)
        else:
            root = get_first_element(non_event_nodes)

    # Construct a full set of states.
    all_states = set(P)
    if root_distn is not None:
        all_states.update(set(root_distn))

    # Get the map from each node to its set of allowed states.
    node_to_allowed_states = {}
    for node in non_event_nodes:
        if node in node_to_state:
            allowed = {node_to_state[node]}
        else:
            allowed = all_states
        node_to_allowed_states[node] = allowed

    # Return the new tree annotated with sampled edge states.
    return resample_restricted_edge_states(
            T, P, node_to_allowed_states, event_nodes,
            root=root, prior_root_distn=root_distn)


def resample_restricted_edge_states(T, P, node_to_allowed_states, event_nodes,
        root, prior_root_distn=None):
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
    P : weighted directed networkx graph, optional
        A sparse transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.
    node_to_allowed_states: dict
        A map from each non-event node to a set of allowed states.
    event_nodes : set of integers
        States of edges adjacent to these nodes are allowed to not be the same
        as each other.  For nodes that are not event nodes,
        the adjacent edge states must all be the same as each other.
    root : integer
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

    # Check for state restrictions of nodes that are not even in the tree.
    bad = set(node_to_allowed_states) - set(T)
    if bad:
        raise ValueError('some of the nodes which have been annotated '
                'with state restrictions '
                'are not even in the tree: ' + str(sorted(bad)))

    # Check the state restrictions.
    bad = set(event_nodes) & set(node_to_allowed_states)
    if bad:
        raise ValueError('the following event nodes have been annotated with '
                'state restrictions: ' + str(sorted(bad)))

    # Check the root.
    if root is None:
        raise ValueError('a root node must be specified')
    if root not in non_event_nodes:
        raise ValueError('the root must be a non-event node')
    if root not in T:
        raise ValueError('the root must be a node in the tree')

    # Check for failure to annotate the non-event nodes.
    bad = set(non_event_nodes) - set(node_to_allowed_states)
    if bad:
        raise ValueError('the following non-event nodes '
                'are not annotated with allowed states: ' + str(sorted(bad)))

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

    # Construct the chunk tree using the provided root.
    chunk_tree, non_event_map, event_map = get_chunk_tree(
            T, event_nodes, root=root)

    # Construct a full set of states.
    all_states = set(P)
    if prior_root_distn is not None:
        all_states.update(set(prior_root_distn))

    # Compute the allowed state set of each chunk tree node.
    chunk_node_to_allowed_states = dict(
            (c, set(all_states)) for c in chunk_tree)
    for node, allowed_states in node_to_allowed_states.items():
        chunk_node = non_event_map[node]
        chunk_node_to_allowed_states[chunk_node].intersection_update(
                allowed_states)

    # Check that the mapping has not failed.
    # This may fail if multiple nodes with different known states
    # map onto the same chunk tree node,
    # which may happen if some edges have no uniformized events.
    if not all(chunk_node_to_allowed_states.values()):
        raise StructuralZeroProb('found an empty intersection '
                'of sets of allowed states for some non-event nodes '
                'which are not separated by an event node')

    # Try to sample states on the chunk tree.
    # This may fail if not enough uniformized events have been placed
    # on the edges.
    chunk_root = non_event_map[root]
    chunk_root_distn = prior_root_distn
    chunk_node_to_sampled_state = resample_restricted_states(
            chunk_tree, chunk_node_to_allowed_states,
            root=chunk_root, prior_root_distn=chunk_root_distn, P_default=P)

    # Copy the original tree before adding states to the edges.
    T = T.copy()

    # Map states onto edges of the tree.
    for a, b in T.edges():
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

