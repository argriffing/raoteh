"""
Sample Markov chain trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler import _sample_mcx, _sample_mcy

from raoteh.sampler._util import StructuralZeroProb

from raoteh.sampler._graph_transform import get_chunk_tree


__all__ = []



#TODO move this into the _sample_mcy module
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
    chunk_node_to_sampled_state = _sample_mcy.resample_states(
            chunk_tree, chunk_root,
            node_to_allowed_states=chunk_node_to_allowed_states,
            root_distn=chunk_root_distn, P_default=P)

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

