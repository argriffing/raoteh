"""
Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mjp
from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)


__all__ = []


def get_forward_sample(T, Q, root, root_distn):
    """
    Use simple unconditional forward sampling to get a history sample.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree.
    Q : directed weighted networkx graph
        A sparse rate matrix.
    root : integer, optional
        Root node.
    root_distn : dict
        Map from root state to probability.

    Returns
    -------
    T_out : weighted undirected acyclic networkx graph
        Weighted tree with extra degree-2 nodes
        and with edges annotated with weights and with sampled states.

    """
    # Summarize the rate matrix.
    total_rates = _mjp.get_total_rates(Q)
    P = _mjp.get_conditional_transition_matrix(Q, total_rates)

    # Initialize some stuff.
    next_node = max(T) + 1

    # Sample the root state.
    states, probs = zip(*root_distn.items())
    root_state = np.random.choice(states, p=probs)

    # The node_to_state is for internal bookkeeping.
    node_to_state = {root : root_state}

    # Build the list of weighted edges.
    annotated_edges = []
    for a, b in nx.bfs_edges(T, root):
        state = node_to_state[a]
        weight = T[a][b]['weight']
        prev_node = a
        total_dwell = 0.0
        if state in total_rates:
            rate = total_rates[state]
            while True:
                dwell = random.expovariate(rate)
                if total_dwell + dwell > weight:
                    break
                total_dwell += dwell
                mid_node = next_node
                next_node += 1
                annotated_edge = (prev_node, mid_node, state, dwell)
                annotated_edges.append(annotated_edge)
                prev_node = mid_node

                # Sample the state.
                states = [s for s in P[state]]
                probs = [P[state][s]['weight'] for s in states]
                state = np.random.choice(states, p=probs)
                node_to_state[prev_node] = state

        # Add the last segment of the branch.
        node_to_state[b] = state
        annotated_edges.append((prev_node, b, state, weight - total_dwell))

    # Return the history sample as an augmented tree.
    T_out = nx.Graph()
    for a, b, state, weight in annotated_edges:
        T_out.add_edge(a, b, state=state, weight=weight)
    return T_out


def gen_histories(T, Q, node_to_state, uniformization_factor=2,
        root=None, root_distn=None):
    """
    Use the Rao-Teh method to sample histories on trees.

    Edges of the yielded trees will be augmented
    with weights and states.
    The weighted size of each yielded tree should be the same
    as the weighted size of the input tree.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree.
    Q : directed weighted networkx graph
        A sparse rate matrix.
    node_to_state : dict
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    uniformization_factor : float, optional
        A value greater than 1.
    root : integer, optional
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.

    """

    # Validate some input.
    if uniformization_factor <= 1:
        raise ValueError('the uniformization factor must be greater than 1')
    if not Q:
        raise ValueError('the rate matrix is empty')
    for a, b in Q.edges():
        if a == b:
            raise ValueError('the rate matrix should have no loops')
    
    # Get the total rate away from each state.
    total_rates = {}
    for a in Q:
        if Q[a]:
            rate_out = 0.0
            for b in Q[a]:
                rate_out += Q[a][b]['weight']
            total_rates[a] = rate_out

    # Initialize omega as the uniformization rate.
    omega = uniformization_factor * max(total_rates.values())

    # Construct a uniformized transition matrix from the rate matrix
    # and the uniformization rate.
    P = nx.DiGraph()
    for a in Q:
        if Q[a]:
            weight = 1.0 - total_rates[a] / omega
            P.add_edge(a, a, weight=weight)
            for b in Q[a]:
                weight = Q[a][b]['weight'] / omega
                P.add_edge(a, b, weight=weight)

    # Define the uniformized poisson rates for Rao-Teh resampling.
    poisson_rates = dict((a, omega - q) for a, q in total_rates.items())

    # Define the initial set of nodes.
    initial_nodes = set(T)

    # Construct an initial feasible history,
    # possibly with redundant event nodes.
    T = get_feasible_history(T, P, node_to_state,
            root=root, root_distn=root_distn)

    # Generate histories using Rao-Teh sampling.
    while True:

        # Identify and remove the non-original redundant nodes.
        all_rnodes = _graph_transform.get_redundant_degree_two_nodes(T)
        expendable_rnodes = all_rnodes - initial_nodes
        T = _graph_transform.remove_redundant_nodes(T, expendable_rnodes)

        # Yield the sampled history on the tree.
        yield T

        # Resample poisson events.
        T = resample_poisson(T, poisson_rates)

        # Resample edge states.
        event_nodes = set(T) - initial_nodes
        T = resample_edge_states(T, P, node_to_state, event_nodes,
                root=root, root_distn=root_distn)


def resample_poisson(T, state_to_rate, root=None):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree whose edges are annotated with states.
    state_to_rate : dict
        Map the state to the expected number of poisson events
        per edge weight.
    root : integer, optional
        Root of the tree.

    Returns
    -------
    T_out : weighted undirected acyclic networkx graph
        Weighted tree without state annotation.

    """

    # If no root was specified then pick one arbitrarily.
    if root is None:
        root = get_first_element(T)

    # Define the next node.
    next_node = max(T) + 1

    # Build the list of weighted edges.
    weighted_edges = []
    for a, b in nx.bfs_edges(T, root):
        weight = T[a][b]['weight']
        state = T[a][b]['state']
        rate = state_to_rate[state]
        prev_node = a
        total_dwell = 0.0
        while True:
            dwell = random.expovariate(rate)
            if total_dwell + dwell > weight:
                break
            total_dwell += dwell
            mid_node = next_node
            next_node += 1
            weighted_edges.append((prev_node, mid_node, dwell))
            prev_node = mid_node
        weighted_edges.append((prev_node, b, weight - total_dwell))

    # Return the resampled tree with poisson events on the edges.
    T_out = nx.Graph()
    T_out.add_weighted_edges_from(weighted_edges)
    return T_out


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
    node_to_pmap = _mjp.construct_node_to_pmap(T, P, node_to_state, root)

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
        posterior_distn = _mjp.get_zero_step_posterior_distn(
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
        posterior_distn = _mjp.get_zero_step_posterior_distn(
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
            return resample_edge_states(
                    T, P, node_to_state, event_nodes,
                    root=root, root_distn=root_distn)
        except StructuralZeroProb as e:
            pass

