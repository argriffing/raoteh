"""
Rao-Teh samples of tolerance MJP trajectories on trees.

This should use concepts related to inference of parameters
of continuous time Bayesian networks (CTBN),
but it will not be so general as to allow any arbitrary network.

"""
from __future__ import division, print_function, absolute_import

import itertools
import random
import math
from collections import defaultdict

import numpy as np
import networkx as nx

from raoteh.sampler import (
        _graph_transform,
        _mjp, _tmjp,
        _sampler, _sample_mcz,
        )

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

from raoteh.sampler._sample_mjp import (
        get_uniformized_transition_matrix,
        )


__all__ = []



# TODO under construction, modified from _sampler.gen_restricted_histories
def xxx_gen_histories(T, root, Q_primary, primary_to_part,
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
def xxx_resample_poisson(T, state_to_rate, root=None):
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
    # Get initial jointly feasible trajectories
    # for the components of the compound process.
    primary_traj, tolerance_trajectories = get_feasible_history(
            T, root,
            Q_primary, primary_distn,
            primary_to_part, rate_on, rate_off,
            node_to_primary_state)

    # Define the initial set of nodes.
    initial_nodes = set(T)


    # TODO unmodified dead code after here

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


#TODO this function is wrong,
#TODO because the poisson samples for uniformization
#TODO depend on the rate out of the current state.
def sample_edge_to_event_times(T, root, event_rate):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.
    event_rate : float
        The poisson rate of new events on edges.

    Returns
    -------
    edge_to_event_times : dict
        Sparse map from edge to collection of event times.
        The edge is an ordered pair of nodes in T,
        where the ordering is away from the root.

    """
    edge_to_event_times = {}
    for edge in nx.bfs_edges(T, root):
        na, nb = edge
        weight = T[na][nb]['weight']
        ntimes_expectation = event_rate * weight
        ntimes = np.random.sample(ntimes_expectation)
        if ntimes:
            event_times = set(np.random.uniform(weight, size=ntimes))
            edge_to_event_times[edge] = event_times
    return edge_to_event_times


def resample_primary_states(
        T, root,
        primary_to_part,
        P_primary, primary_distn, absorption_rate_map,
        tolerance_trajectories, edge_to_time_events):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.

    Returns
    -------

    Notes
    -----
    This function is not involved in resampling uniformization times.

    """
    # Precompute the set of all tolerance classes.
    tolerance_classes = set(primary_to_part.values())

    # Build a merged tree corresponding to the tolerance trajectories,
    # with event nodes corresponding to uniformization times
    # for the primary process.
    T_merged, event_nodes = _graph_transform.add_trajectories(T, root,
            tolerance_trajectories,
            edge_to_event_times=edge_to_event_times)

    # Construct the 'chunk tree' whose edges
    # are in correspondence with the primary process event nodes.
    info = _graph_transform.get_chunk_tree_type_b(
            T_merged, root, event_nodes)
    chunk_tree, edge_to_chunk_node, event_node_to_chunk_edge = info

    # Get the map from each chunk node to the set of
    # tolerance classes in the 'on' state
    # that fall within the subtree represented by the chunk node.
    chunk_node_to_tol_set = defaultdict(set)
    for merged_edge in nx.bfs_edges(T_merged, root):

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]

        # For each tolerance class,
        # check if its state along this edge is 'on',
        # and if so, add the tolerance class to the set of 'on'
        # tolerance classes in the chunk node that includes this edge.
        for tolerance_class in tolerance_classes:
            tolerance_state = T_merged[na][nb]['state'][tolerance_class]
            if tolerance_state:
                chunk_node_to_tol_set[chunk_node].add(tolerance_class)

    # The 'absorption' represents missed opportunities for the primary
    # trajectory to have transitioned to primary states
    # in enabled tolerance classes.
    chunk_node_to_prim_to_absorption = dict(
            (chunk_node, defaultdict(float)) for chunk_node in chunk_tree)
    for merged_edge in nx.bfs_edges(T_merged, root):

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]
        prim_to_absorption = chunk_node_to_prim_to_absorption[chunk_node]

        # Define the set of tolerance classes that are enabled on this edge.
        enabled_tolerance_classes = set()
        for tolerance_class in tolerance_classes:
            if T_merged[na][nb]['state'][tolerance_class]:
                enabled_tolerance_classes.add(tolerance_class)

        # For each primary state,
        # add the absorption contribution for each enabled tolerance class.
        for primary_state in set(primary_distn):
            tol_to_absorption = absorption_rate_map[primary_state]
            for tolerance_class in enabled_tolerance_classes:
                if tolerance_class in tol_to_absorption:
                    absorption_contrib = tol_to_absorption[tolerance_class]
                    prim_to_absorption[prim] += absorption_contrib

    # For each chunk node, construct a map from tolerance state
    # to an emission likelihood.
    # The idea is that subtrees corresponding to chunk nodes
    # are more or less hospitable to various primary states.
    # A chunk node that contains a segment in the 'off' tolerance state
    # for a primary state is completely inhospitable for that primary state,
    # and so the emission likelihood for that primary state
    # for that chunk node will be zero.
    # If the chunk node is not completely inhospitable,
    # then it is comfortable to a primary state to the degree
    # that it the region has low "absorption" with respect to that state.
    # A low absorption means that the sum of rates from the primary state
    # to other primary states that are tolerated in the region is low.
    chunk_node_to_state_to_likelihood = {}
    for chunk_node in chunk_tree:
        prim_to_absorption = chunk_node_to_prim_to_absorption[chunk_node]
        state_to_likelihood = {}
        for prim in set(primary_distn):
            if primary_to_part[prim] in chunk_node_to_tol_set:
                if absorption in prim_to_absorption:
                    state_to_likelihood[prim] = math.exp(-absorption)
                else:
                    state_to_likelihood[prim] = 1
        chunk_node_to_state_to_likelihood[chunk_node] = state_to_likelihood

    # Use mcz-type conditional sampling to
    # sample primary states at each node of the chunk tree.
    chunk_node_to_sampled_state = _sample_mcz.resample_states(
            chunk_tree, root,
            node_to_state_to_likelihood=chunk_node_to_state_to_likelihood,
            root_distn=primary_distn, P_default=P_primary)

    # Map the sampled chunk node primary states back onto
    # the base tree to give the sampled primary process trajectory.
    sampled_traj = nx.Graph()
    for merged_edge in nx.bfs_edges(T_merged, root):
        merged_na, merged_nb = merged_edge
        weight = T_merged[merged_na][merged_nb]['weight']
        chunk_node = edge_to_chunk_node[merged_edge]
        sampled_state = chunk_node_to_sampled_state[chunk_node]
        sampled_traj.add_edge(
                merged_na, merged_nb,
                weight=weight, state=sampled_state)

    # Return the resampled primary trajectory.
    return sampled_traj


def resample_tolerance_states(
        T, root,
        primary_to_part,
        Q_primary, absorption_rate_map, P_tolerance, tolerance_distn,
        primary_trajectory, edge_to_event_times, tolerance_class):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.

    Returns
    -------

    Notes
    -----
    This function resamples a tolerance trajectory for only a single
    tolerance class, and this function is not involved in
    resampling uniformization times.

    """
    # Build a merged tree corresponding to the primary trajectory,
    # with event nodes corresponding to uniformization times
    # for the tolerance process of the tolerance class of interest.
    T_merged, event_nodes = _graph_transform.add_trajectories(T, root,
            [primary_trajectory],
            edge_to_event_times=edge_to_event_times)

    # Construct the 'chunk tree' whose edges
    # are in correspondence with the tolerance event nodes.
    info = _graph_transform.get_chunk_tree_type_b(
            T_merged, root, event_nodes)
    chunk_tree, edge_to_chunk_node, event_node_to_chunk_edge = info

    # Get the map from each chunk node to the set of
    # tolerance classes of primary states that fall within
    # the trajectory subtree represented by that chunk node.
    chunk_node_to_tol_set = defaultdict(set)
    for merged_edge in nx.bfs_edges(T_merged, root):

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]

        # Get the tolerance class of the primary state of the trajectory
        # on the merged edge, and add its tolerance class to
        # the set of tolerance classes associated with the chunk node.
        primary_state = T_merged[na][nb]['states'][0]
        primary_tol_class = primary_to_part[primary_state]
        chunk_node_to_tol_set[chunk_node].add(primary_tol_class)

    # The 'absorption' represents missed opportunities for the primary
    # trajectory to have transitioned to primary states
    # of the tolerance class of interest.
    chunk_node_to_absorption = defaultdict(float)
    for merged_edge in nx.bfs_edges(T_merged, root):

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]
        
        # Get the tolerance class of the primary state of the trajectory
        # on the merged edge, and add its 'absorption'
        # to the cumulative absorption associated with the chunk node.
        primary_state = T_merged[na][nb]['states'][0]
        tol_to_absorption = absorption_rate_map[primary_state]
        if tolerance_class in tol_to_absorption:
            absorption_contrib = tol_to_absorption[tolerance_class]
            chunk_node_to_absorption[chunk_node] += absorption_contrib

    # For each chunk node, construct a map from tolerance state
    # to an emission likelihood.
    # This construction uses the set of tolerance classes
    # represented within each chunk node,
    # and it also uses the 'absorption' of each chunk node
    # with respect to the current tolerance class.
    # If the current tolerance class is represented in a given
    # chunk node, then the tolerance state 0 is penalized
    # by assigning a likelihood of zero.
    # Otherwise if the current tolerance class is not represented
    # in the given chunk node, then the tolerance state 1 is penalized
    # according to the missed opportunities of the primary process
    # to have transitioned to a primary state of the given tolerance class;
    # this multiplicative likelihood penalty is
    # the exponential of the negative of the chunk node 'absorption'
    # with respect to the current tolerance class.
    chunk_node_to_state_to_likelihood = {}
    for chunk_node in chunk_tree:
        if tolerance_class in chunk_node_to_tol_set:
            state_to_likelihood = {1:1}
        else:
            if chunk_node in chunk_node_to_absorption:
                absorption = chunk_node_to_absorption[chunk_node]
                state_to_likelihood = {0:1, 1:math.exp(-absorption)}
            else:
                state_to_likelihood = {0:1, 1:1}
        chunk_node_to_state_to_likelihood[chunk_node] = state_to_likelihood

    # Use mcz-type conditional sampling to
    # sample tolerance states at each node of the chunk tree.
    chunk_node_to_tolerance_state = _sample_mcz.resample_states(
            chunk_tree, root,
            node_to_state_to_likelihood=chunk_node_to_state_to_likelihood,
            root_distn=tolerance_distn, P_default=P_tolerance)

    # Map the sampled chunk node tolerance states back onto
    # the base tree to give the sampled tolerance process trajectory.
    tolerance_traj = nx.Graph()
    for merged_edge in nx.bfs_edges(T_merged, root):
        merged_na, merged_nb = merged_edge
        weight = T_merged[merged_na][merged_nb]['weight']
        chunk_node = edge_to_chunk_node[merged_edge]
        tolerance_state = chunk_node_to_tolerance_state[chunk_node]
        tolerance_traj.add_edge(
                merged_na, merged_nb,
                weight=weight, state=tolerance_state)

    # Return the resampled tolerance trajectory.
    return tolerance_traj


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

    # Get the tolerance transition rate matrix
    # and the uniformized tolerance transition probability matrix.
    Q_tolerance = nx.DiGraph()
    if rate_on:
        Q_tolerance.add_edge(0, 1, weight=rate_on)
    if rate_off:
        Q_tolerance.add_edge(1, 0, weight=rate_off)
    P_tolerance = get_uniformized_transition_matrix(Q_tolerance)

    # Get a primary process proposal rate matrix
    # that approximates the primary component of the compound process.
    Q_proposal = _tmjp.get_primary_proposal_rate_matrix(
            Q_primary, primary_to_part, tolerance_distn)

    # Get the uniformized transition probability matrix
    # corresponding to the primary proposal transition rate matrix.
    P_proposal = get_uniformized_transition_matrix(Q_proposal)

    # Sample the primary process trajectory using this proposal.
    primary_trajectory = _sampler.get_feasible_history(
            T, P_proposal, node_to_primary_state,
            root=root, root_distn=primary_distn)

    # Get the times of the primary trajectory events
    # along edges of the base tree.
    primary_event_map = _graph_transform.get_event_map(
            T, root, primary_trajectory, predecessors=None)

    # For each primary process state and each tolerance class,
    # compute the sum of rates from the primary state
    # into primary states of that tolerance class.
    absorption_rate_map = {}
    for prim_sa, tol_prim_sa in primary_to_part.items():
        absorption_rates = defaultdict(float)
        for prim_sb in Q_primary[prim_sa]:
            tol_prim_sb = primary_to_part[prim_sb]
            rate = Q_primary[prim_sa][prim_sb]['weight']
            absorption_rates[tol_prim_sb] += rate
        absorption_rate_map[prim_sa] = dict(absorption_rates)

    # Initialize the list of tolerance process trajectories.
    tolerance_trajectories = []
    for tolerance_class in range(nparts):

        # Define tolerance process uniformization event times,
        # so that an assignment of tolerance states will be possible.
        # This can be accomplished by putting a tolerance process
        # uniformization event at a random time on each
        # primary trajectory segment.
        edge_to_event_times = {}
        for base_edge, events in primary_event_map.items():

            # Unpack the base edge nodes, ordered away from the root.
            base_na, base_nb = base_edge

            # Initialize the set of tolerance event times.
            tolerance_event_times = set()

            # Add a tolerance process event
            # before the first primary process transition.
            dead_time = T[base_na][base_nb]['weight']
            if events:
                dead_time = min(tm for tm, obj in events)
            tolerance_event_times.add(random.uniform(0, dead_time)

            # Add a tolerance process event
            # into every segment that follows a primary process transition.
            for tm, primary_edge_object in events:
                edge_length = primary_edge_object['weight']
                tolerance_event_times.add(random.uniform(tm, tm + edge_length))

            # Define the set of tolerance process event times for this edge.
            edge_to_event_times[base_edge] = tolerance_event_times

        # Sample the rest of the tolerance trajectory
        # by sampling the tolerance states given the uniformized timings.
        tolerance_traj = resample_tolerance_states(
                T, root,
                primary_to_part,
                Q_primary, absorption_rate_map, P_tolerance, tolerance_distn,
                primary_trajectory, edge_to_event_times, tolerance_class)

        # Add the tolerance trajectory to the list.
        tolerance_trajectories.append(tolerance_traj)

    # Return the feasible trajectories.
    return primary_trajectory, tolerance_trajectories

