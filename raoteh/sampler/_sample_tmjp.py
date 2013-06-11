"""
Rao-Teh samples of tolerance MJP trajectories on trees.

This should use concepts related to inference of parameters
of continuous time Bayesian networks (CTBN),
but it will not be so general as to allow any arbitrary network.

"""
from __future__ import division, print_function, absolute_import

import itertools
import random

import numpy as np
import networkx as nx

from raoteh.sampler import _graph_transform, _mjp

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._mjp import (
        get_total_rates,
        get_trajectory_log_likelihood,
        )

from raoteh.sampler._sample_mc import (
        resample_edge_states,
        resample_restricted_edge_states,
        )

from raoteh.sampler._sampler import (
        get_uniformized_transition_matrix,
        )


__all__ = []


# TODO unmodified from _sampler.gen_restricted_histories
def gen_histories(T, Q, node_to_allowed_states,
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
        T = resample_restricted_edge_states(
                T, P, node_to_allowed_states, event_nodes,
                root=root, prior_root_distn=root_distn)


#TODO unmodified from _sampler.resample_poisson
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


#TODO unmodified from _sampler.get_restricted_feasible_history
def get_feasible_history(
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
            return resample_restricted_edge_states(
                    T, P, node_to_allowed_states, event_nodes,
                    root=root, prior_root_distn=root_distn)
        except StructuralZeroProb as e:
            pass

