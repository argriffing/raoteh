"""
Rao-Teh samples of tolerance MJP trajectories on trees.

This should use concepts related to inference of parameters
of continuous time Bayesian networks (CTBN),
but it will not be so general as to allow any arbitrary network.

In this module,
the disease_data variable is a list, indexed by tolerance class,
of maps from a node to a set of allowed tolerance states.

This module uses dense arrays and ndarrays.

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
        _util,
        _mjp, _tmjp,
        _mjp_dense, _tmjp_dense,
        _sample_mcy, _sample_mjp,
        _sample_mcx_dense, _sample_mcy_dense, _sample_mjp_dense,
        )

from raoteh.sampler._sample_mjp_dense import get_uniformized_transition_matrix


__all__ = []


def gen_histories_v1(ctm, T, root, node_to_primary_state,
        disease_data=None, uniformization_factor=2, nhistories=None):
    """
    Use the Rao-Teh method to sample histories on trees.

    Edges of the yielded trees will be augmented
    with weights and states.
    The weighted size of each yielded tree should be the same
    as the weighted size of the input tree.

    Parameters
    ----------
    ctm : instance of _tmjp_dense.CompoundToleranceModel
        Model defining the compound Markov process.
    T : weighted undirected acyclic networkx graph
        Weighted tree.
    root : integer
        Root of the tree.
    node_to_primary_state : dict
        A map from a node to a primary process state observation.
        If a node is missing from this map,
        then the observation is treated as missing.
    disease_data : list, optional
        A list, indexed by tolerance class,
        of maps from a node to a set of allowed tolerance states.
    uniformization_factor : float, optional
        A value greater than 1.
    nhistories : integer, optional
        Sample this many histories.
        If None, then sample an unlimited number of histories.

    """
    # Get initial jointly feasible trajectories
    # for the components of the compound process.
    primary_trajectory, tolerance_trajectories = get_feasible_history(
            ctm, T, root, node_to_primary_state,
            disease_data=disease_data)

    # Summarize the primary process in ways that are useful for Rao-Teh.
    primary_total_rates = _mjp_dense.get_total_rates(ctm.Q_primary)
    primary_max_total_rate = primary_total_rates.max()
    primary_omega = uniformization_factor * primary_max_total_rate
    primary_poisson_rates = primary_omega - primary_total_rates
    P_primary = get_uniformized_transition_matrix(
            ctm.Q_primary, omega=primary_omega)

    # Summarize the 2-state tolerance process
    # in ways that are useful for Rao-Teh.
    Q_tolerance = _tmjp_dense.get_tolerance_rate_matrix(
            ctm.rate_off, ctm.rate_on)
    tolerance_total_rates = _mjp_dense.get_total_rates(Q_tolerance)
    tolerance_max_total_rate = tolerance_total_rates.max()
    tolerance_omega = uniformization_factor * tolerance_max_total_rate
    tolerance_poisson_rates = tolerance_omega - tolerance_total_rates
    P_tolerance = get_uniformized_transition_matrix(
            Q_tolerance, omega=tolerance_omega)

    # Generate histories using Rao-Teh sampling.
    for i in itertools.count():

        # Remove redundant nodes in the tolerance trajectories.
        new_tolerance_trajectories = []
        for traj in tolerance_trajectories:
            all_rnodes = _graph_transform.get_redundant_degree_two_nodes(traj)
            extra_rnodes = all_rnodes - set(T)
            traj = _graph_transform.remove_redundant_nodes(traj, extra_rnodes)
            new_tolerance_trajectories.append(traj)
        tolerance_trajectories = new_tolerance_trajectories

        # Yield the sampled trajectories.
        yield primary_trajectory, tolerance_trajectories

        # If we have sampled enough histories, then return.
        if nhistories is not None:
            nsampled = i + 1
            if nsampled >= nhistories:
                return

        # Resample poisson events on the primary trajectory,
        # then extract the event map from the resulting timing trajectory.
        timing_traj = _sample_mjp_dense.resample_poisson(
                primary_trajectory, primary_poisson_rates)
        event_map = _graph_transform.get_event_map(T, root, timing_traj)
        edge_to_event_times = {}
        for edge, events in event_map.items():
            event_times = set(tm for tm, obj in events)
            edge_to_event_times[edge] = event_times

        # Resample primary states given the base tree and tolerances.
        primary_trajectory = resample_primary_states_v1(
                T, root,
                ctm.primary_to_part,
                P_primary, ctm.primary_distn,
                node_to_primary_state,
                tolerance_trajectories, edge_to_event_times)

        # Remove redundant nodes in the primary process trajectory
        # so that it can be more efficiently used as a background
        # for sampling the tolerance process trajectories.
        all_rnodes = _graph_transform.get_redundant_degree_two_nodes(
                primary_trajectory)
        expendable_rnodes = all_rnodes - set(T)
        primary_trajectory = _graph_transform.remove_redundant_nodes(
                primary_trajectory, expendable_rnodes)

        # Resample tolerance process trajectories.
        new_tolerance_trajectories = []
        for tol, tol_traj in enumerate(tolerance_trajectories):

            # Resample poisson events on the tolerance trajectory,
            # then extract the event map from the resulting timing trajectory.
            timing_traj = _sample_mjp.resample_poisson(
                tol_traj, tolerance_poisson_rates)
            event_map = _graph_transform.get_event_map(T, root, timing_traj)
            edge_to_event_times = {}
            for edge, events in event_map.items():
                event_times = set(tm for tm, obj in events)
                edge_to_event_times[edge] = event_times

            # Resample the tolerance states.
            traj = resample_tolerance_states_v1(
                    T, root,
                    ctm.primary_to_part,
                    P_tolerance, ctm.tolerance_distn,
                    primary_trajectory, edge_to_event_times, tol,
                    disease_data=disease_data)

            # Add the tolerance trajectory.
            new_tolerance_trajectories.append(traj)

        # Update the list of tolerance trajectories.
        # Note that these have redundant nodes which should be removed.
        tolerance_trajectories = new_tolerance_trajectories


def resample_primary_states_v1(
        T, root,
        primary_to_part,
        P_primary, primary_distn,
        node_to_primary_state,
        tolerance_trajectories, edge_to_event_times):
    """
    Resample primary states.

    The v1 in the function name indicates that this function
    addresses the dependence among components of the tolerance process
    strictly through conditioning rather than through rate dependence.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.
    primary_to_part : x
        x
    P_primary : x
        x
    primary_distn : x
        x
    tolerance_trajectories : x
        x
    edge_to_event_times : x
        x

    Returns
    -------
    T_primary_aug : x
        x

    Notes
    -----
    This function is not involved in resampling uniformization times.

    """
    nprimary = len(primary_to_part)

    # Precompute the set of all tolerance classes.
    tolerance_classes = set(primary_to_part.values())

    # Build a merged tree corresponding to the tolerance trajectories,
    # with event nodes corresponding to uniformization times
    # for the primary process.
    T_merged, event_nodes = _graph_transform.add_trajectories(T, root,
            tolerance_trajectories,
            edge_to_event_times=edge_to_event_times)

    # Precompute the breath first edges for the merged tree.
    T_merged_bfs_edges = list(nx.bfs_edges(T_merged, root))

    # Construct the 'chunk tree' whose edges
    # are in correspondence with the primary process event nodes.
    info = _graph_transform.get_chunk_tree_type_b(
            T_merged, root, event_nodes)
    chunk_tree, edge_to_chunk_node, event_node_to_chunk_edge = info

    # Get the map from each chunk node to the set of
    # tolerance classes in the 'off' state
    # that fall within the subtree represented by the chunk node.
    chunk_node_to_forbidden_tols = defaultdict(set)
    for merged_edge in T_merged_bfs_edges:

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]

        # For each tolerance class,
        # check if its state along this edge is 'off',
        # and if so, add the tolerance class to the set of forbidden
        # tolerance classes in the chunk node that includes this edge.
        for tol in tolerance_classes:
            tolerance_state = T_merged[na][nb]['states'][tol]
            if not tolerance_state:
                chunk_node_to_forbidden_tols[chunk_node].add(tol)

    # Check that no chunk node forbids all tolerance classes.
    for chunk_node, forbidden_tols in chunk_node_to_forbidden_tols.items():
        bad_tols = set(forbidden_tols) - set(tolerance_classes)
        if bad_tols:
            raise Exception('internal error: '
                    'for this chunk node, '
                    'the set of forbidden tolerance classes contains some '
                    'unrecognized entries: ' + str(sorted(bad_tols)))
        if set(forbidden_tols) == set(tolerance_classes):
            raise Exception('internal error: '
                    'for this chunk node, all tolerance classes are forbidden')

    # The chunk node may be constrained by primary state data.
    chunk_node_to_obs_state = {}
    for merged_edge in T_merged_bfs_edges:

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]

        # If a state has been observed for the given edge node,
        # then set the observation of the chunk node.
        for n in (na, nb):
            if n in node_to_primary_state:
                obs_state = node_to_primary_state[n]
                if chunk_node in chunk_node_to_obs_state:
                    if chunk_node_to_obs_state[chunk_node] != obs_state:
                        raise Exception('internal error: '
                                'multiple conflicting observations '
                                'within the same chunk node')
                chunk_node_to_obs_state[chunk_node] = obs_state

    # For each chunk node, construct the set of allowed tolerance states.
    # This is the set of primary states that do not belong
    # to any of the tolerance classes that are forbidden somewhere in the
    # region of the tree corresponding to the chunk node.
    chunk_node_to_allowed_states = {}
    for chunk_node in chunk_tree:

        # Get the set of forbidden tolerance classes for this chunk node.
        forbidden_tols = chunk_node_to_forbidden_tols[chunk_node]

        # Initialize the set of allowed states to
        # the set of all primary states not forbidden
        # by the tolerance class trajectory within the chunk node.
        allowed_states = set()
        for prim, prob in enumerate(primary_distn):
            if not prob:
                continue
            if primary_to_part[prim] not in forbidden_tols:
                allowed_states.add(prim)

        # Further restrict the set of allowed state according to
        # observations at points within the chunk node.
        if chunk_node in chunk_node_to_obs_state:
            obs_state = chunk_node_to_obs_state[chunk_node]
            allowed_states.intersection_update({obs_state})

        # If no state is allowed then this is a problem.
        if not allowed_states:
            print()
            print('error report...')
            print('T:')
            for na, nb in nx.bfs_edges(T, root):
                print(na, nb, T[na][nb]['weight'])
            print('root:', root)
            for i, t_traj in enumerate(tolerance_trajectories):
                print('tolerance trajectory', i, ':')
                for na, nb in nx.bfs_edges(t_traj, root):
                    weight = t_traj[na][nb]['weight']
                    state = t_traj[na][nb]['state']
                    print(na, nb, weight, state)
            print('merged tree:')
            for na, nb in nx.bfs_edges(T_merged, root):
                weight = T_merged[na][nb]['weight']
                states = T_merged[na][nb]['states']
                print(na, nb, weight, states)
            print('node to primary state:')
            for node, primary_state in node_to_primary_state.items():
                print(node, primary_state)
            print('chunk tree:')
            for na, nb in nx.bfs_edges(chunk_tree, root):
                print(na, nb)
            print('edge to chunk node:')
            for edge, cnode in sorted(edge_to_chunk_node.items()):
                na, nb = edge
                print(na, nb, cnode)
            print('chunk node to forbidden tolerance classes:')
            for cnode, forbidden_tols in chunk_node_to_forbidden_tols.items():
                print(cnode, forbidden_tols)
            raise Exception('internal error: '
                    'for this chunk node no primary state is allowed')

        # Store the set of allowed states for this chunk node.
        chunk_node_to_allowed_states[chunk_node] = allowed_states

    # Use mcy-type conditional sampling to
    # sample primary states at each node of the chunk tree.
    chunk_node_to_sampled_state = _sample_mcy_dense.resample_states(
            chunk_tree, root, nprimary,
            node_to_allowed_states=chunk_node_to_allowed_states,
            root_distn=primary_distn, P_default=P_primary)

    # Map the sampled chunk node primary states back onto
    # the base tree to give the sampled primary process trajectory.
    sampled_traj = nx.Graph()
    for merged_edge in T_merged_bfs_edges:
        merged_na, merged_nb = merged_edge
        weight = T_merged[merged_na][merged_nb]['weight']
        chunk_node = edge_to_chunk_node[merged_edge]
        sampled_state = chunk_node_to_sampled_state[chunk_node]
        sampled_traj.add_edge(
                merged_na, merged_nb,
                weight=weight, state=sampled_state)

    # Return the resampled primary trajectory.
    return sampled_traj


def resample_tolerance_states_v1(
        T, root,
        primary_to_part,
        P_tolerance, tolerance_distn,
        primary_trajectory, edge_to_event_times, tolerance_class,
        disease_data=None):
    """
    Resample tolerance states.

    The v1 in the function name indicates that this function
    addresses the dependence among components of the tolerance process
    strictly through conditioning rather than through rate dependence.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer
        Root of the tree.
    primary_to_part : x
        x
    P_tolerance : x
        x
    tolerance_distn : x
        x
    primary_trajectory : x
        x
    edge_to_event_times : x
        x
    tolerance_class : x
        x
    disease_data : list, optional
        A list, indexed by tolerance class,
        of maps from a node to a set of allowed tolerance states.

    Returns
    -------

    Notes
    -----
    This function resamples a tolerance trajectory for only a single
    tolerance class, and this function is not involved in
    resampling uniformization times.

    """
    # This function only uses disease_data through disease_map.
    if disease_data is not None:
        disease_map = disease_data[tolerance_class]

    # Build a merged tree corresponding to the primary trajectory,
    # with event nodes corresponding to uniformization times
    # for the tolerance process of the tolerance class of interest.
    T_merged, event_nodes = _graph_transform.add_trajectories(T, root,
            [primary_trajectory],
            edge_to_event_times=edge_to_event_times)

    # Precompute the breath first edges for the merged tree.
    T_merged_bfs_edges = list(nx.bfs_edges(T_merged, root))

    # Construct the 'chunk tree' whose edges
    # are in correspondence with the tolerance event nodes.
    info = _graph_transform.get_chunk_tree_type_b(
            T_merged, root, event_nodes)
    chunk_tree, edge_to_chunk_node, event_node_to_chunk_edge = info

    # Get the map from each chunk node to the set of
    # tolerance classes of primary states that fall within
    # the trajectory subtree represented by that chunk node.
    chunk_node_to_tol_set = defaultdict(set)
    for merged_edge in T_merged_bfs_edges:

        # Unpack the merged edge and get the chunk node that it maps to.
        na, nb = merged_edge
        chunk_node = edge_to_chunk_node[merged_edge]

        # Get the tolerance class of the primary state of the trajectory
        # on the merged edge, and add its tolerance class to
        # the set of tolerance classes associated with the chunk node.
        primary_state = T_merged[na][nb]['states'][0]
        primary_tol_class = primary_to_part[primary_state]
        chunk_node_to_tol_set[chunk_node].add(primary_tol_class)

    # Get the map from each chunk node to the set of tolerance states
    # allowed by the disease data.
    if disease_data is not None:
        chunk_node_to_disease_restriction = dict(
                (n, {0, 1}) for n in chunk_tree)
        for merged_edge in T_merged_bfs_edges:
            na, nb = merged_edge
            chunk_node = edge_to_chunk_node[merged_edge]
            for n in (na, nb):
                if n in disease_map:
                    restriction = chunk_node_to_disease_restriction[chunk_node]
                    restriction.intersection_update(disease_map[n])

    # For each chunk node, construct the set of allowed tolerance states.
    # This will be {1} if the primary process belongs to the
    # tolerance class of interest at any point within the subtree
    # corresponding to the chunk node.
    # Otherwise this set will be {0, 1}.
    # Unless the disease data further restricts the tolerance state.
    chunk_node_to_allowed_states = {}
    for chunk_node in chunk_tree:
        allowed_states = {0, 1}
        if disease_data is not None:
            disease_restriction = chunk_node_to_disease_restriction[chunk_node]
            allowed_states.intersection_update(disease_restriction)
        if tolerance_class in chunk_node_to_tol_set[chunk_node]:
            allowed_states.intersection_update({1})
        chunk_node_to_allowed_states[chunk_node] = allowed_states

    # Use mcy-type conditional sampling to
    # sample tolerance states at each node of the chunk tree.
    ntolerance_states = 2
    chunk_node_to_tolerance_state = _sample_mcy_dense.resample_states(
            chunk_tree, root, ntolerance_states,
            node_to_allowed_states=chunk_node_to_allowed_states,
            root_distn=tolerance_distn, P_default=P_tolerance)

    # Map the sampled chunk node tolerance states back onto
    # the base tree to give the sampled tolerance process trajectory.
    tolerance_traj = nx.Graph()
    for merged_edge in T_merged_bfs_edges:
        merged_na, merged_nb = merged_edge
        weight = T_merged[merged_na][merged_nb]['weight']
        chunk_node = edge_to_chunk_node[merged_edge]
        tolerance_state = chunk_node_to_tolerance_state[chunk_node]
        tolerance_traj.add_edge(
                merged_na, merged_nb,
                weight=weight, state=tolerance_state)

    # Return the resampled tolerance trajectory.
    return tolerance_traj


def get_feasible_history(ctm, T, root, node_to_primary_state,
        disease_data=None):
    """
    Find an arbitrary feasible history.

    Parameters
    ----------
    ctm : instance of _tmjp_dense.CompoundToleranceModel
        Model defining the compound Markov process.
    T : weighted undirected acyclic networkx graph
        This is the original tree.
    root : integer, optional
        Root of the tree.
    node_to_primary_state : dict
        A sparse map from a node to a known primary state.
    disease_data : list, optional
        A list, indexed by tolerance class,
        of maps from a node to a set of allowed tolerance states.

    Returns
    -------
    primary_trajectory : weighted undirected acyclic networkx graphs
        Primary process trajectory.
        Redundant nodes have been removed.
    tolerance_trajectories : seq of weighted undirected acyclic networkx graphs
        Sequence of tolerance trajectories.
        Redundant nodes have not been removed.

    Notes
    -----
    The returned history is not sampled according to any particularly
    meaningful distribution.
    It is up to the caller to remove redundant self-transitions.
    The primary process is assumed to be time-reversible.

    """
    # Get the tolerance state distribution, rate matrix,
    # and uniformized tolerance transition probability matrix.
    Q_tolerance = _tmjp_dense.get_tolerance_rate_matrix(
            ctm.rate_off, ctm.rate_on)
    P_tolerance = get_uniformized_transition_matrix(Q_tolerance)

    # Get a primary process proposal rate matrix
    # that approximates the primary component of the compound process.
    Q_proposal = _tmjp_dense.get_primary_proposal_rate_matrix(
            ctm.Q_primary, ctm.primary_to_part, ctm.tolerance_distn)

    # Get the uniformized transition probability matrix
    # corresponding to the primary proposal transition rate matrix.
    P_proposal = get_uniformized_transition_matrix(Q_proposal)

    # Sample the primary process trajectory using this proposal.
    primary_trajectory = _sample_mcx_dense.get_feasible_history(
            T, node_to_primary_state, ctm.nprimary,
            root=root, root_distn=ctm.primary_distn, P_default=P_proposal)

    # Remove redundant nodes in the primary process trajectory
    # so that it can be more efficiently used as a background
    # for sampling the tolerance process trajectories.
    all_rnodes = _graph_transform.get_redundant_degree_two_nodes(
            primary_trajectory)
    expendable_rnodes = all_rnodes - set(T)
    primary_trajectory = _graph_transform.remove_redundant_nodes(
            primary_trajectory, expendable_rnodes)

    # Get the times of the primary trajectory events
    # along edges of the base tree.
    primary_event_map = _graph_transform.get_event_map(
            T, root, primary_trajectory, predecessors=None)

    # Initialize the list of tolerance process trajectories.
    tolerance_trajectories = []
    for tolerance_class in range(ctm.nparts):

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
            tolerance_event_times.add(random.uniform(0, dead_time))

            # Add a tolerance process event
            # into every segment that follows a primary process transition.
            for tm, primary_edge_object in events:
                edge_length = primary_edge_object['weight']
                tolerance_event_times.add(random.uniform(tm, tm + edge_length))

            # Define the set of tolerance process event times for this edge.
            edge_to_event_times[base_edge] = tolerance_event_times

        # Sample the rest of the tolerance trajectory
        # by sampling the tolerance states given the uniformized timings.
        tolerance_traj = resample_tolerance_states_v1(
                T, root,
                ctm.primary_to_part,
                P_tolerance, ctm.tolerance_distn,
                primary_trajectory, edge_to_event_times, tolerance_class,
                disease_data=disease_data)

        # Add the tolerance trajectory to the list.
        tolerance_trajectories.append(tolerance_traj)

    # Return the feasible trajectories.
    return primary_trajectory, tolerance_trajectories

