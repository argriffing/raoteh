"""
Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform


__all__ = []


class SamplingError(Exception):
    pass

class StructuralZeroProb(SamplingError):
    pass

class NumericalZeroProb(SamplingError):
    pass



def get_first_element(elements):
    for x in elements:
        return x


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

    # If the root has not been provided, then pick one with a known state.
    if root is None:
        root = get_first_element(node_to_state)

    # Bookkeeping structures related to tree traversal.
    successors = nx.dfs_successors(T, root)
    predecessors = nx.dfs_predecessors(T, root)

    # A bookkeeping structure related to state sampling.
    P_for_sampling = {}
    for source in P:
        sinks = []
        probs = []
        for sink in P[source]:
            sinks.append(sink)
            probs.append(P[source][sink]['weight'])
        P_for_sampling[source] = (sinks, probs)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):
        if node in node_to_state:
            node_state = node_to_state[node]
            node_to_pmap[node] = {node_state : 1.0}
        else:
            pmap = {}
            for node_state in P:
                nprobs = []
                for n in successors[node]:

                    # Get the list of possible child node states.
                    # These are limited by sparseness of the matrix of
                    # transitions from the parent state,
                    # and also by the possibility
                    # that the state of the child node is already known.
                    valid_states = set(P[node_state]) & set(node_to_pmap[n])
                    if valid_states:
                        nprob = 0.0
                        for s in valid_states:
                            a = P[node_state][s]['weight']
                            b = node_to_pmap[n][s]
                            nprob += a * b
                        nprobs.append(nprob)
                    else:
                        nprobs = None
                        break
                if nprobs is not None:
                    cprob = np.product(nprobs)
                    pmap[node_state] = cprob
            node_to_pmap[node] = pmap

    # Sample the node states, beginning at the root.
    node_to_sampled_state = {}

    # Treat the root separately.
    # If only one state is possible at the root, then we do not have to sample.
    # Otherwise consult the map from root states to probabilities.
    if len(node_to_pmap[root]) == 1:
        root_state = get_first_element(node_to_pmap[root])
        if not node_to_pmap[root][root_state]:
            raise NumericalZeroProb(
                    'the only feasible state at the root '
                    'gives a subtree probability of zero')
    else:
        if root_distn is None:
            raise ValueError('expected a distribution over states at the root')
        prior_distn = root_distn
        states = list(set(prior_distn) & set(node_to_pmap[root]))
        if not states:
            raise StructuralZeroProb('no root is feasible')
        weights = []
        for s in states:
            weights.append(prior_distn[s] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise NumericalZeroProb('numerical problem at the root')
        probs = np.array(weights, dtype=float) / weight_sum
        sampled_state = np.random.choice(states, p=probs)
        root_state = sampled_state
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
        states = list(set(P[parent_state]) & set(node_to_pmap[node]))
        if not states:
            raise StructuralZeroProb('found a non-root infeasibility')
        weights = []
        for s in states:
            weights.append(P[parent_state][s]['weight'] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise NumericalZeroProb('numerical problem at a non-root node')
        probs = np.array(weights, dtype=float) / weight_sum
        sampled_state = np.random.choice(states, p=probs)
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


def get_feasible_history(T, P, node_to_state, root=None, root_distn=None):
    """
    Find an arbitrary feasible history.

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
    root : integer, optional
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.

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
        if events_per_edge > len(P):
            raise StructuralZeroProb('failed to find a feasible history')

        # Increment some stuff and bisect edges if appropriate.
        if k is None:
            k = 0
        else:
            T = get_edge_bisected_graph(T)
            events_per_edge += 2**k
            k += 1

        # Get the event nodes.
        event_nodes = set(T) - non_event_nodes

        # Try to sample edge states.
        try:
            return resample_edge_states(
                    T, P, node_to_state, event_nodes,
                    root=root, root_distn=root_distn)
        except StructuralZeroProb as e:
            pass

