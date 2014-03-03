"""
Rao-Teh samples of tolerance MJP trajectories on trees.

This should use concepts related to inference of parameters
of continuous time Bayesian networks (CTBN),
but it will not be so general as to allow any arbitrary network.

In this module,
the disease_data variable is a list, indexed by tolerance class,
of maps from a node to a set of allowed tolerance states.

This version is unfinished.

"""
from __future__ import division, print_function, absolute_import

import math
from collections import defaultdict

import networkx as nx

from raoteh.sampler import _graph_transform, _sample_mcz


__all__ = []


###############################################################################
# v2


#TODO untested
#TODO needs to also address transition dependences not just absorptions
def resample_primary_states_v2(
        T, root,
        primary_to_part,
        P_primary, primary_distn, absorption_rate_map,
        tolerance_trajectories, edge_to_event_times):
    """
    Resample primary states.

    The v2 in the function name indicates that this function
    addresses the dependence among components of the tolerance process
    through dependence of the rate matrices on the background processes
    rather than strictly through conditioning.

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
    # tolerance classes in the 'off' state
    # that fall within the subtree represented by the chunk node.
    chunk_node_to_forbidden_tols = defaultdict(set)
    for merged_edge in nx.bfs_edges(T_merged, root):

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
            if primary_to_part[prim] not in chunk_node_to_forbidden_tols:
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


#TODO untested
#TODO needs to also address transition dependences not just absorptions
def resample_tolerance_states_v2(
        T, root,
        primary_to_part,
        Q_primary, absorption_rate_map, P_tolerance, tolerance_distn,
        primary_trajectory, edge_to_event_times, tolerance_class):
    """
    The v2 in the function name indicates that this function
    addresses the dependence among components of the tolerance process
    through dependence of the rate matrices on the background processes
    rather than strictly through conditioning.

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

