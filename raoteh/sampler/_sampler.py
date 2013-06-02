"""
Forward samples and Rao-Teh samples of MJP trajectories on trees.

MJP means Markov jump process.
Rao-Teh sampling is a kind of endpoint-conditioned continuous-time
discrete-state trajectory sampling which we have generalized
from sampling trajectories on paths to sampling trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mjp

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._sample_mc import (
        resample_edge_states,
        resample_restricted_edge_states,
        )


__all__ = []


class _FastRandomChoice:
    # This is a helper class to speed up np.random.choice.

    def __init__(self, P, bufsize=1000):
        self._P = P
        self._bufsize = bufsize
        self._state_to_sink_samples = {}
        self._state_to_nconsumed = {}

    def sample(self, sa):
        nconsumed = self._state_to_nconsumed.get(sa, None)
        if (nconsumed is None) or (nconsumed >= self._bufsize):
            sbs = [sb for sb in self._P[sa]]
            probs = [self._P[sa][sb]['weight'] for sb in sbs]
            if not np.allclose(np.sum(probs), 1):
                raise Exception('internal error')
            self._state_to_sink_samples[sa] = np.random.choice(
                    sbs, size=self._bufsize, p=probs)
            self._state_to_nconsumed[sa] = 0
            nconsumed = 0
        sampled_state = self._state_to_sink_samples[sa][nconsumed]
        self._state_to_nconsumed[sa] += 1
        return sampled_state


def gen_forward_samples(T, Q, root, root_distn, nsamples=None):
    """
    Use simple unconditional forward sampling to generate history samples.

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
    nsamples : integer, optional
        Yield this many forward samples.

    Yields
    ------
    T_out : weighted undirected acyclic networkx graph
        Weighted tree with extra degree-2 nodes
        and with edges annotated with weights and with sampled states.

    """
    # Summarize the rate matrix.
    total_rates = _mjp.get_total_rates(Q)
    P = _mjp.get_conditional_transition_matrix(Q, total_rates)
    state_sampler = _FastRandomChoice(P)

    # Unpack the distribution over root states.
    root_states, root_probs = zip(*root_distn.items())

    # If nsamples is known then sample all root states.
    all_root_states = None
    if nsamples is not None:
        all_root_states = np.random.choice(
                root_states, size=nsamples, p=root_probs)

    for i in itertools.count():

        # Check if we have already made enough samples.
        if i == nsamples:
            return

        # Initialize some stuff.
        next_node = max(T) + 1

        # Sample the root state.
        if all_root_states is not None:
            root_state = all_root_states[i]
        else:
            root_state = np.random.choice(root_states, p=root_probs)

        # The node_to_state is for internal bookkeeping.
        node_to_state = {root : root_state}

        # Build the list of weighted edges.
        annotated_edges = []
        for a, b in nx.bfs_edges(T, root):
            state = node_to_state[a]
            weight = T[a][b]['weight']
            prev_node = a
            total_dwell = 0.0
            while True:
                if state in total_rates:
                    rate = total_rates[state]
                else:
                    break
                dwell = np.random.exponential(scale = 1/rate)
                if total_dwell + dwell > weight:
                    break
                total_dwell += dwell
                mid_node = next_node
                next_node += 1
                annotated_edge = (prev_node, mid_node, state, dwell)
                annotated_edges.append(annotated_edge)
                prev_node = mid_node

                # Sample the state.
                #states = [s for s in P[state]]
                #probs = [P[state][s]['weight'] for s in states]
                #state = np.random.choice(states, p=probs)
                state = state_sampler.sample(state)
                node_to_state[prev_node] = state

            # Add the last segment of the branch.
            node_to_state[b] = state
            annotated_edges.append((prev_node, b, state, weight - total_dwell))

        # Yield the history sample as an augmented tree.
        T_out = nx.Graph()
        for a, b, state, weight in annotated_edges:
            T_out.add_edge(a, b, state=state, weight=weight)
        yield T_out


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
        while True:
            if state in total_rates:
                rate = total_rates[state]
            else:
                break
            dwell = np.random.exponential(scale = 1/rate)
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
    total_rates = _mjp.get_total_rates(Q)

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
            dwell = np.random.exponential(scale = 1/rate)
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


def get_feasible_history(T, P, node_to_state, root=None, root_distn=None):
    """
    Find an arbitrary feasible history.

    This is being replaced by get_restricted_feasible_history.

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
    # If the root has not been specified,
    # pick a root with known state if any exist,
    # and pick an arbitrary one otherwise.
    if root is None:
        if node_to_state:
            root = get_first_element(node_to_state)
        else:
            root = get_first_element(T)

    # Get the set of all states.
    all_states = set(P)
    if root_distn is not None:
        all_states.update(set(root_distn))

    # Convert the state restriction format.
    node_to_allowed_states = {}
    for node in T:
        if node in node_to_state:
            allowed = {node_to_state[node]}
        else:
            allowed = all_states
        node_to_allowed_states[node] = allowed

    # Get the restricted feasible history.
    return get_restricted_feasible_history(
        T, P, node_to_allowed_states, root=root, root_distn=root_distn)


def get_restricted_feasible_history(
        T, P, node_to_allowed_states, root, root_distn=None):
    """
    Find an arbitrary feasible history.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    P : weighted directed networkx graph
        A sparse transition matrix assumed to be identical for all edges.
        The weights are transition probabilities.
    node_to_allowed_states : dict
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
            return resample_restricted_edge_states(
                    T, P, node_to_allowed_states, event_nodes,
                    root=root, prior_root_distn=root_distn)
        except StructuralZeroProb as e:
            pass

