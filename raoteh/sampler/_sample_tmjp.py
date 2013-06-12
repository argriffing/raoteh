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

from raoteh.sampler import _graph_transform, _sampler, _mjp, _tmjp

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



# TODO under construction, modified from _sampler.gen_restricted_histories
def gen_histories(T, root, Q_primary, primary_to_part,
        node_to_primary_state, rate_on, rate_off,
        primary_distn, uniformization_factor=2, nhistories=None):
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
    root : integer
        Root of the tree.
    Q_primary : directed weighted networkx graph
        A matrix of transition rates among states in the primary process.
    primary_to_part : dict
        A map from a primary state to its tolerance class.
    node_to_primary_state : dict
        A map from a node to a primary process state observation.
        If a node is missing from this map,
        then the observation is treated as missing.
    rate_on : float
        Transition rate from tolerance state off to tolerance_state on.
    rate_off : float
        Transition rate from tolerance state on to tolerance_state off.
    primary_distn : dict
        Map from root state to prior probability.
    uniformization_factor : float, optional
        A value greater than 1.
    nhistories : integer, optional
        Sample this many histories.
        If None, then sample an unlimited number of histories.

    """
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


#TODO under construction; modified from _sampler.get_restricted_feasible_history
def get_feasible_history(
        T, root,
        Q_primary, primary_distn,
        primary_to_part, rate_on, rate_off,
        node_to_primary_state):
    """
    Find an arbitrary feasible history.

    The first group of args defines the underlying rooted phylogenetic tree.
    The second group relates purely to the primary process.
    The third group relates to the rest of the compound tolerance process.
    The final group consists of observation data.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer, optional
        Root of the tree.
    Q_primary : weighted directed networkx graph
        Primary process rate matrix.
    primary_distn : dict
        Primary process state distribution.
    primary_to_part : dict
        Map from primary state to tolerance class.
    node_to_primary_state : dict
        A sparse map from a node to a known primary state.
    rate_on : float
        Transition rate from tolerance off -> on.
    rate_off : float
        Transition rate from tolerance on -> off.

    Returns
    -------
    primary_trajectory : weighted undirected acyclic networkx graphs
        Primary process trajectory.
    tolerance_trajectories : seq of weighted undirected acyclic networkx graphs
        Sequence of tolerance trajectories.

    Notes
    -----
    The returned history is not sampled according to any particularly
    meaningful distribution.
    It is up to the caller to remove redundant self-transitions.
    The primary process is assumed to be time-reversible.

    """
    # Get the number of tolerance classes.
    nparts = len(set(primary_to_part.values()))

    # Get the tolerance state distribution.
    tolerance_distn = _tmjp.get_tolerance_distn(rate_off, rate_on)

    # Get a primary process proposal rate matrix
    # that approximates the primary component of the compound process.
    Q_proposal = _tmjp.get_primary_proposal_rate_matrix(
            Q_primary, primary_to_part, tolerance_distn)

    # Get the uniformized transition probability matrix
    # corresponding to the primary proposal transition rate matrix.
    P_proposal = _sampler.get_uniformized_transition_matrix(Q_proposal)

    # Sample the primary process trajectory using this proposal.
    primary_trajectory = _sampler.get_feasible_history(
            T, P_proposal, node_to_primary_state,
            root=root, root_distn=primary_distn)

    # Next use _graph_transform.get_event_map() to get the
    # times of the primary trajectory events along edges of the base tree.

    # Next, for each tolerance process,
    # build a merged tree with len(states)==1 corresponding to the
    # primary trajectory, and use the primary process event map
    # to add event nodes at times that interleave the primary process
    # event times -- in other words, there will be a uniformized
    # tolerance process event node at a random time
    # on each primary process trajectory segment,
    # for every tolerance class.
    return primary_trajectory, tolerance_trajectories

