"""
Forward samples and Rao-Teh samples of MJP trajectories on trees.

MJP means Markov jump process.
Rao-Teh sampling is a kind of endpoint-conditioned continuous-time
discrete-state trajectory sampling which we have generalized
from sampling trajectories on paths to sampling trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import itertools
import random

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mjp, _sample_mcy

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._mjp import (
        get_total_rates,
        get_trajectory_log_likelihood,
        )

from raoteh.sampler._sample_mjp import (
        get_uniformized_transition_matrix,
        resample_poisson,
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
    total_rates = get_total_rates(Q)
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
    total_rates = get_total_rates(Q)
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


def gen_histories(T, Q, node_to_state,
        root=None, root_distn=None,
        uniformization_factor=2, nhistories=None):
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
    root : integer, optional
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.
    uniformization_factor : float, optional
        A value greater than 1.
    nhistories : integer, optional
        Sample this many histories.
        If None, then sample an unlimited number of histories.

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
    all_states = set(Q)
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

    # Sample the histories.
    for history in gen_restricted_histories(T, Q, node_to_allowed_states,
            root, root_distn=root_distn,
            uniformization_factor=uniformization_factor,
            nhistories=nhistories):
        yield history


def gen_restricted_histories(T, Q, node_to_allowed_states,
        root, root_distn=None, uniformization_factor=2, nhistories=None):
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
    node_to_allowed_states : dict
        A map from each node to a set of allowed states.
    root : integer
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.
    uniformization_factor : float, optional
        A value greater than 1.
    nhistories : integer, optional
        Sample this many histories.
        If None, then sample an unlimited number of histories.

    """
    # Check for state restrictions of nodes that are not even in the tree.
    bad = set(node_to_allowed_states) - set(T)
    if bad:
        raise ValueError('some of the nodes which have been annotated '
                'with state restrictions '
                'are not even in the tree: ' + str(sorted(bad)))

    # Validate some more input.
    if uniformization_factor <= 1:
        raise ValueError('the uniformization factor must be greater than 1')
    if not Q:
        raise ValueError('the rate matrix is empty')
    for a, b in Q.edges():
        if a == b:
            raise ValueError('the rate matrix should have no loops')
    
    # Get the total rate away from each state.
    total_rates = get_total_rates(Q)

    # Initialize omega as the uniformization rate.
    omega = uniformization_factor * max(total_rates.values())

    # Get the uniformized transition matrix.
    P = get_uniformized_transition_matrix(Q, uniformization_factor)

    # Define the uniformized poisson rates for Rao-Teh resampling.
    poisson_rates = dict((a, omega - q) for a, q in total_rates.items())

    # Define the initial set of nodes.
    initial_nodes = set(T)

    # Construct an initial feasible history,
    # possibly with redundant event nodes.
    T = get_restricted_feasible_history(T, P, node_to_allowed_states,
            root=root, root_distn=root_distn)

    # Generate histories using Rao-Teh sampling.
    for i in itertools.count():

        # Identify and remove the non-original redundant nodes.
        all_rnodes = _graph_transform.get_redundant_degree_two_nodes(T)
        expendable_rnodes = all_rnodes - initial_nodes
        T = _graph_transform.remove_redundant_nodes(T, expendable_rnodes)

        # Yield the sampled history on the tree.
        yield T

        # If we have sampled enough histories, then return.
        if nhistories is not None:
            nsampled = i + 1
            if nsampled >= nhistories:
                return

        # Resample poisson events.
        T = resample_poisson(T, poisson_rates)

        # Resample edge states.
        event_nodes = set(T) - initial_nodes
        T = _sample_mcy.resample_edge_states(
                T, root, P, event_nodes,
                node_to_allowed_states=node_to_allowed_states,
                root_distn=root_distn)


def gen_mh_histories(T, Q, node_to_allowed_states,
        target_log_likelihood_callback,
        root, root_distn=None, uniformization_factor=2, nhistories=None):
    """
    Rao-Teh with Metropolis-Hastings to sample histories on trees.

    Yield pairs of (tree, indicator) where the indicator communicates
    whether or not the sampled value is new.
    If the proposal was accepted or if it is the initial sample
    then this indicator is True.
    If the proposal was rejected then this indicator is False.
    In both cases the caller should treat the yielded tree as a
    Metropolis-Hastings sample from the target distribution.
    Edges of the yielded trees will be augmented with weights and states.
    The weighted size of each yielded tree should be the same
    as the weighted size of the input tree.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree.
    Q : directed weighted networkx graph
        A sparse rate matrix.
    node_to_allowed_states : dict
        A map from each node to a set of allowed states.
    target_log_likelihood_callback : callable
        This function gets the log likelihood of the given proposal,
        under the target distribution.
    root : integer
        Root of the tree.
    root_distn : dict, optional
        Map from root state to probability.
    uniformization_factor : float, optional
        A value greater than 1.
    nhistories : integer, optional
        Sample this many histories.
        If None, then sample an unlimited number of histories.

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
    total_rates = get_total_rates(Q)

    # Initialize omega as the uniformization rate.
    omega = uniformization_factor * max(total_rates.values())

    # Get the uniformized transition matrix.
    P = get_uniformized_transition_matrix(Q, uniformization_factor)

    # Define the uniformized poisson rates for Rao-Teh resampling.
    poisson_rates = dict((a, omega - q) for a, q in total_rates.items())

    # Define the initial set of nodes.
    initial_nodes = set(T)

    # Define the biased log likelihood callback function.
    # This is required for determining ratios of proposal densities
    # for the Metropolis-Hastings acceptance or rejection.
    def biased_log_likelihood_callback(T_aug):
        return get_trajectory_log_likelihood(T_aug, root, root_distn, Q)

    # No previous sample.
    T_prev = None
    ll_biased_prev = None
    ll_target_prev = None

    # Construct an initial feasible history,
    # possibly with redundant event nodes.
    T = get_restricted_feasible_history(T, P, node_to_allowed_states,
            root=root, root_distn=root_distn)

    # Generate histories using Rao-Teh proposals for Metropolis-Hastings.
    # This could possibly be abstracted into a more general framework.
    for i in itertools.count():

        # Identify and remove the non-original redundant nodes.
        all_rnodes = _graph_transform.get_redundant_degree_two_nodes(T)
        expendable_rnodes = all_rnodes - initial_nodes
        T = _graph_transform.remove_redundant_nodes(T, expendable_rnodes)

        # If this is the first sample, then accept it without
        # computing anything related to Metropolis-Hastings.
        # Otherwise, compute a Metropolis-Hastings ratio
        # and determine whether to accept or reject the proposed tree.
        # This requires the log likelihoods of both the previous
        # and the proposed samples, computed under both the
        # sampling and the target distributions.
        if T_prev is None:
            accept_flag = True
        else:

            # Compute the log likelihood
            # according to the biased distribution used for sampling.
            if ll_biased_prev is None:
                ll_biased_prev = biased_log_likelihood_callback(T_prev)
            ll_biased_curr = biased_log_likelihood_callback(T)

            # Compute the log likelihood
            # according to the target distribution.
            if ll_target_prev is None:
                ll_target_prev = target_log_likelihood_callback(T_prev)
            ll_target_curr = target_log_likelihood_callback(T)

            # Compute the log Metropolis-Hastings ratio.
            log_mh_ratio = sum((
                ll_target_curr,
                -ll_target_prev,
                -ll_biased_curr,
                ll_biased_prev))

            # If the M-H ratio is favorable then always accept.
            # If the M-H ratio is unfavoarable
            # then accept with some probability.
            if log_mh_ratio > 0:
                accept_flag = True
            else:
                if random.random() < np.exp(log_mh_ratio):
                    accept_flag = True
                else:
                    accept_flag = False

            # If accepted, store the log likelihoods for the next iteration.
            if accept_flag:
                ll_biased_prev = ll_biased_curr
                ll_target_prev = ll_target_curr

        # If the proposal was rejected then revert to the previous sample.
        if not accept_flag:
            T = T_prev

        # Yield a sample
        yield T, accept_flag

        # Store the tree for the next iteration.
        T_prev = T

        # If we have sampled enough histories, then return.
        if nhistories is not None:
            nsampled = i + 1
            if nsampled >= nhistories:
                return

        # Resample poisson events.
        T = resample_poisson(T, poisson_rates)

        # Resample edge states.
        event_nodes = set(T) - initial_nodes
        T = _sample_mcy.resample_edge_states(
                T, root, P, event_nodes,
                node_to_allowed_states=node_to_allowed_states,
                root_distn=root_distn)


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
        T, P, node_to_allowed_states, root, root_distn=root_distn)


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
    # Check for state restrictions of nodes that are not even in the tree.
    bad = set(node_to_allowed_states) - set(T)
    if bad:
        raise ValueError('some of the nodes which have been annotated '
                'with state restrictions '
                'are not even in the tree: ' + str(sorted(bad)))

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
            return _sample_mcy.resample_edge_states(
                    T, root, P, event_nodes,
                    node_to_allowed_states=node_to_allowed_states,
                    root_distn=root_distn)
        except StructuralZeroProb as e:
            pass

