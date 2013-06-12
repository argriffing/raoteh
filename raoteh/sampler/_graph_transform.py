"""
Graph transformations related to the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import heapq
from collections import defaultdict

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)


__all__ = []


def get_edge_bisected_graph(G):
    """

    Parameters
    ----------
    G : weighted undirected networkx graph
        Input graph whose edges are to be bisected in the output.

    Returns
    -------
    G_out : weighted undirected networkx graph
        Every edge in the original tree will correspond to two edges
        in the bisected tree.

    Notes
    -----
    Weights in the output graph will be adjusted accordingly.
    Nodes in the original graph are assumed to be integers.
    Nodes in the output graph will be a superset of the nodes
    in the original graph, and the newly added nodes representing
    degree-two points will have greater values than the max of the
    node values in the input graph.

    """
    max_node = max(G)
    G_out = nx.Graph()
    for i, (a, b, data) in enumerate(G.edges(data=True)):
        full_weight = data['weight']
        half_weight = 0.5 * full_weight
        mid = max_node + i + 1
        G_out.add_edge(a, mid, weight=half_weight)
        G_out.add_edge(mid, b, weight=half_weight)
    return G_out


def get_redundant_degree_two_nodes(T):
    """

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Edges are annotated with states.

    Returns
    -------
    redundant_nodes : set of integers
        Set of degree 2 nodes for which adjacent edges have identical states.

    Notes
    -----
    This function does not take into account any property of the nodes
    themselves, other than the number of edges they are adjacent to
    and the states of these adjacent edges.
    In particular, it doesn't distinguish event nodes vs. non-event nodes.

    """
    redundant_nodes = set()
    for a in T:
        degree = len(T[a])
        if degree == 2:
            state_set = set(T[a][b]['state'] for b in T[a])
            if len(state_set) == 1:
                redundant_nodes.add(a)
    return redundant_nodes


def remove_selected_degree_two_nodes(T, root, expendable_nodes):
    """
    Returns a tree with the specified nodes removed.

    The removal disregards all edge attributes.

    Parameters
    ----------
    T : networkx graph
        tree
    root : integer
        Root node that is not expendable.
    expendable_nodes : set of nodes
        Nodes to be removed.

    Returns
    -------
    T_out : networkx graph
        tree with redundant nodes removed

    """
    # Initialize the output.
    T_out = nx.Graph()

    # Bookkeeping.
    degrees = T.degree()

    # Input validation.
    if root in expendable_nodes:
        raise ValueError('the root is not allowed to be removed')

    # Check that the caller is not trying to remove nodes
    # that have unweighted degree other than degree 2.
    if any(degrees[n] != 2 for n in expendable_nodes):
        raise ValueError('only degree 2 nodes may be considered expendable')

    # Build the new tree, tracking the info and skipping the extra nodes.
    rnode_to_ancestor = {}
    for na, nb in nx.bfs_edges(T, root):

        # Get the essential ancestor of the current node.
        if na in expendable_nodes:
            ancestor = rnode_to_ancestor[na]

        # If nb is not expendable, then an edge will be added.
        # If nb is expendable then expendable edge info will be
        # initialized or extended.
        if nb not in expendable_nodes:
            if na in expendable_nodes:
                T_out.add_edge(ancestor, nb)
            else:
                T_out.add_edge(na, nb)
        else:
            if na not in expendable_nodes:
                rnode_to_ancestor[nb] = na

    # Return the new tree as a networkx graph.
    return T_out


def remove_redundant_nodes(T, redundant_nodes):
    """
    Returns a tree with the specified redundant nodes removed.

    The removal respects edge 'weight' and edge 'state'.
    Currently only nodes of degree two can be removed.
    The 'state' of both edges adjacent to such a node must be the same.
    The returned tree will have the same weighted tree size
    as the original tree.

    Parameters
    ----------
    T : networkx graph
        tree
    redundant_nodes : set of nodes
        candidates for removal

    Returns
    -------
    T_out : networkx graph
        tree with redundant nodes removed

    """
    T_out = nx.Graph()

    # Pick a root with only one neighbor.
    degrees = T.degree()
    tips = set(n for n, d in degrees.items() if d == 1)
    root = get_first_element(tips)

    # Check that the caller is not trying to remove nodes
    # that have unweighted degree other than degree 2.
    if any(degrees[n] != 2 for n in redundant_nodes):
        raise ValueError('only degree 2 nodes may be considered redundant')

    # Set up some bookkeeping.
    successors = nx.dfs_successors(T, root)

    # Build the new tree, tracking the info and skipping the extra nodes.
    rnode_to_info = {}
    for a, b in nx.bfs_edges(T, root):

        # Get the data associated with the current edge.
        weight = T[a][b]['weight']
        state = T[a][b]['state']

        # If the first node is redundant then get its info,
        # check the state, and update the weight.
        if a in redundant_nodes:
            info = rnode_to_info[a]
            if info['state'] != state:
                raise ValueError('edge state mismatch')
            weight += info['weight']

        # If b is not redundant, then an edge will be added.
        # If b is redundant then redundant edge info will be
        # initialized or extended.
        if b not in redundant_nodes:
            if a in redundant_nodes:
                T_out.add_edge(info['ancestor'], b,
                        weight=weight, state=state)
            else:
                T_out.add_edge(a, b,
                        weight=weight, state=state)
        else:
            if a not in redundant_nodes:
                info = dict(ancestor=a, state=state, weight=weight)
            else:
                info['weight'] = weight
            rnode_to_info[b] = info

    # Return the new tree as a networkx graph.
    return T_out


def get_chunk_tree(T, event_nodes, root=None):
    """
    Construct a certain kind of dual graph.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Input tree graph with integer nodes.
    event_nodes : set of integers
        This subset of nodes in the input graph
        will have bijective correspondence to edges in the output graph.
    root : integer, optional
        Tree root.

    Returns
    -------
    chunk_tree : undirected unweighted acyclic networkx graph
        A kind of dual tree with new nodes and edges.
    non_event_node_map : dict
        A map from each non-event node in the original graph
        to a node in the output graph.
    event_node_map : dict
        A map from each event node in the original graph
        to a pair of adjacent nodes in the output graph.

    Notes
    -----
    Edges in the output chunk graph correspond bijectively to event nodes
    in the original graph.

    """

    # Partition the input nodes into event and non-event nodes.
    non_event_nodes = set(T) - set(event_nodes)

    # Initialize the root.
    if root is None:
        root = get_first_element(non_event_nodes)
    if root not in non_event_nodes:
        raise ValueError('the root must be a non-event node')

    # Initialize the outputs.
    chunk_tree = nx.Graph()
    non_event_node_map = {}
    event_node_map = {}

    # Populate the outputs,
    # traversing the input graph in preorder from an arbitrary non-event root.
    next_chunk = 0
    non_event_node_map[root] = next_chunk
    chunk_tree.add_node(next_chunk)
    next_chunk += 1
    for a, b in nx.bfs_edges(T, root):

        # Get the chunk associated with the parent node.
        if a in non_event_nodes:
            parent_chunk = non_event_node_map[a]
        elif a in event_nodes:
            grandparent_chunk, parent_chunk = event_node_map[a]
        else:
            raise Exception('internal error')

        # If the child node is an event node,
        # then begin a new chunk and add an edge to the output graph.
        # Otherwise update the output maps but do not modify the output graph.
        if b in event_nodes:
            chunk_edge = (parent_chunk, next_chunk)
            event_node_map[b] = chunk_edge
            chunk_tree.add_edge(*chunk_edge)
            next_chunk += 1
        elif b in non_event_nodes:
            non_event_node_map[b] = parent_chunk
        else:
            raise Exception('internal error')

    # Return the outputs.
    return chunk_tree, non_event_node_map, event_node_map


def get_node_to_state(T, query_nodes):
    """
    Get the states of some nodes, by looking at states of adjacent edges.

    Parameters
    ----------
    T : undirected networkx graph
        A tree whose edges are annotated with states.
    query_nodes : set
        A set of nodes whose states are to be queried.

    Returns
    -------
    node_to_state : dict
        A map from query node to state.

    """
    # Input validation.
    bad = set(query_nodes) - set(T)
    if bad:
        raise ValueError('some query nodes are missing '
                'from the tree: ' + str(sorted(bad)))

    # Get the map from nodes to states.
    node_to_state = {}
    for na in query_nodes:
        adjacent_states = set(T[na][nb]['state'] for nb in T[na])
        if len(adjacent_states) != 1:
            raise ValueError('query node %s has inconsistent state '
                    'as determined by its adjacent edges' % na)
        node_to_state[na] = get_first_element(adjacent_states)
    return node_to_state


def get_event_map(T, root, traj, predecessors=None):
    """

    Parameters
    ----------
    T : undirected weighted networkx graph
        The base tree.
    root : integer
        The root node.
        This is used as the root for both the base tree
        and for the trajectory tree.
    traj : undirected weighted networkx graph
        Like the base tree, but with some extra degree-2 nodes.
        The naming of this argument is inspired by the term trajectory,
        although it is not required to be a trajectory in any technical sense.
    predecessors : dict, optional
        Map from a node in the base tree T to its predecessor in the base tree.
        The directionality of the predecessor vs. successor
        is determined by the root.

    Returns
    -------
    event_map : dict
        Map from an edge of the base tree to a
        (time, augmented tree edge object) pair.
        The edge of the base tree is represented by an ordered pair,
        with the ordering being in the direction away from the root.
        The edge object of the augmented tree
        corresponds to the edge following the transition event.

    """
    # Initialize the event map.
    event_map = defaultdict(list)

    # Bookkeeping.
    if predecessors is not None:
        predecessors = nx.dfs_predecessors(T, root)
    traj_successors = nx.dfs_successors(traj, root)

    # Associate each trajectory edge to a base tree edge.
    traj_edge_to_base_edge = {}
    traj_preorder_edges = list(tuple(x) for x in nx.bfs_edges(traj, root))
    for traj_edge in reversed(traj_preorder_edges):
        traj_na, traj_nb = traj_edge
        if traj_nb in T:
            base_nb = traj_nb
            base_na = predecessors[base_nb]
            base_edge = (base_na, base_nb)
        else:
            traj_nb_successors = traj_successors[traj_nb]
            if len(traj_nb_successors) != 1:
                raise Exception('internal error')
            traj_nb_successor = traj_nb_successors[0]
            traj_child_edge = (traj_nb, traj_nb_successor)
            base_edge = traj_edge_to_base_edge[traj_child_edge]
        traj_edge_to_base_edge[traj_edge] = base_edge
    
    # Each traj node that is not in T is a traj transition event.
    # Put each transition event into the priority queue
    # of the corresponding edge of the base tree.
    base_edge_to_tm = {}
    for traj_edge in traj_preorder_edges:

        # Unpack the endpoints of the trajectory edge.
        traj_na, traj_nb = traj_edge

        # If there is no event on this edge then continue.
        if (traj_na in T) and (traj_nb in T):
            continue

        # Map the trajectory event back to an edge of the base tree.
        base_edge = traj_edge_to_base_edge[traj_edge]

        # Get the timing of the current event along the edge.
        tm = base_edge_to_tm.get(base_edge, 0)

        # Define the networkx edge
        # corresponding to the segment of the trajectory.
        traj_edge_object = traj[traj_na][traj_nb]

        # If traj_na is a transition event,
        # then add its information to the event map.
        if traj_na not in T:
            event_info = (tm, traj_edge_object)
            event_map[base_edge].append(event_info)

        # Update the timing along the edge.
        traj_weight = traj_edge_object['weight']
        base_edge_to_tm[base_edge] = tm + traj_weight

    # Return the event map.
    return event_map


def add_trajectories(T, root, trajectories, edge_to_event_times=None):
    """
    Construct a tree with merged trajectories.

    Parameters
    ----------
    T_base : undirected weighted networkx graph
        A weighted base tree.
    root : integer
        Root node common to all trajectories.
    trajectories : sequence of undirected weighted networkx graphs
        Edges should be annotated with 'weight' and with 'state'.
        The state should change only at nodes of degree two.
    edge_to_event_times : dict, optional
        If available, this provides a map from an edge of T
        to a collection of times.
        The format of each edge key is as an ordered pair of nodes of T,
        in bfs order radiating from the root.
        The times are with respect to the node that is closer to the root.
        The purpose of this arg is to optionally break the
        trajectory history into pieces for resampling the trajectory
        of a variable that depends on the trajectories of the variables
        provided to this function.

    Returns
    -------
    T_merged : undirected weighted networkx graph
        A new tree with more nodes.
        Edges are annotated with 'states' which gives a state
        for each trajectory.
    dummy_nodes : set of integers
        A set of dummy nodes added to the tree.
        These correspond to edge_to_event_times provided as input
        to this function.

    """
    # Bookkeeping.
    predecessors = nx.dfs_predecessors(T, root)
    successors = nx.dfs_successors(T, root)
    T_bfs_edges = list(tuple(x) for x in nx.bfs_edges(T, root))

    # Check that the trajectories have correct shape.
    for traj in trajectories:
        traj_specific_nodes = set(traj) - set(T)
        traj_skeleton = remove_selected_degree_two_nodes(
                traj, root, traj_specific_nodes)
        if set(T_bfs_edges) != set(nx.bfs_edges(traj_skeleton, root)):
            raise ValueError('expected the trajectory to follow '
                    'the basic shape of the base tree')

    # Check that the trajectories have the correct total edge weight.
    total_base_edge_weight = T.size(weight='weight')
    for traj in trajectories:
        traj_weight = traj.size(weight='weight')
        if not np.allclose(traj_weight, total_base_edge_weight):
            raise ValueError('each trajectory should have '
                    'the same total weight as the base tree\n'
                    'base tree weight: %s\n'
                    'traj tree weight: %s' % (
                        total_base_edge_weight, traj_weight))

    # For each trajectory get the map from base node to state.
    traj_node_to_state = []
    for traj in trajectories:
        query_nodes = set(T)
        node_to_state = get_node_to_state(traj, query_nodes)
        traj_node_to_state.append(node_to_state)

    # For each directed edge of the base tree,
    # maintain a priority queue of interleaved transitions along trajectories.
    base_edge_to_q = {}
    for na, nb in T_bfs_edges:
        base_edge = (na, nb)
        base_edge_to_q[base_edge] = []

    # Put dummy events into the edge-specific priority queues.
    if edge_to_event_times is not None:
        for edge, times in edge_to_event_times.items():
            q = base_edge_to_q[edge]
            for tm in times:
                q_item = (tm, None, None)
                heapq.heappush(q, q_item)

    # For each trajectory, put events in the priority queue of each edge.
    for traj_index, traj in enumerate(trajectories):

        # Get the timing and nature of events along the edges.
        event_map = get_event_map(T, root, traj, predecessors=predecessors)

        # Add the event information to the priority queue
        # of the appropriate edge.
        for base_edge, events in event_map.items():
            base_na, base_nb = base_edge
            for tm, traj_edge_object in events:
                traj_state = traj_edge_object['state']
                q_item = (tm, traj_index, traj_state)
                heapq.heappush(base_edge_to_q[base_edge], q_item)

    # Initialize the return values.
    T_merged = nx.Graph()
    dummy_nodes = set()

    # For each edge of the original tree,
    # add segments to the merged tree, such that no trajectory
    # transition occurs within any segment.
    # Annotate every segment with the state of every trajectory.
    next_new_node = max(T) + 1
    for base_edge in T_bfs_edges:

        # Unpack the edge endpoints.
        base_na, base_nb = base_edge

        # Get the edge weight from the base tree.
        base_edge_weight = T[base_na][base_nb]['weight']

        # Initialize the most recent segment node.
        prev_node = base_na

        # Define the trajectory states at the beginning of the edge.
        current_states = []
        for node_to_state in traj_node_to_state:
            current_states.append(node_to_state[base_na])

        # Iterate through the priority queue, adding an edge
        # when a transition is encountered on any trajectory.
        q = base_edge_to_q[base_edge]
        tm = 0
        while q:
            tm_event, traj_index, traj_state = heapq.heappop(q)
            T_merged.add_edge(
                    prev_node, next_new_node,
                    weight=tm_event-tm,
                    states=list(current_states))

            # If the traj_index is None then it means that
            # the event is a dummy event.
            # Dummy events do not change the state.
            if traj_index is None:
                dummy_nodes.add(next_new_node)
            else:
                current_states[traj_index] = traj_state

            # Update for the next iteration.
            prev_node = next_new_node
            next_new_node += 1
            tm = tm_event

        # Check that we have reached the states
        # that we had expected to reach.
        for traj_index, node_to_state in enumerate(traj_node_to_state):
            if node_to_state[base_nb] != current_states[traj_index]:
                raise Exception('internal error')

        # Add the final segment.
        T_merged.add_edge(
                prev_node, base_nb,
                weight=base_edge_weight-tm,
                states=list(current_states))

    # Return the merged tree and the set of dummy nodes.
    return T_merged, dummy_nodes
