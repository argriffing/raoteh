"""
Graph transformations related to the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

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
    expendable_nodes : set of nodes
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
        raise Exception('only degree 2 nodes may be considered redundant')

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
                raise Exception('edge state mismatch')
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


def add_trajectory_layer(T_base, root, T_current, T_traj):
    """
    Merge a trajectory layer onto the base tree.

    Parameters
    ----------
    T_base : undirected networkx graph
        A base tree.
    root : integer
        Root node common to all trees.
    T_current : undirected weighted networkx graph
        A base tree with edge weights and specific optional edge annotation.
        Edges of this tree may or may not be annotated with a 'states' list.
        Any 'state' annotation of the edges will be ignored.
    T_traj : undirected weighted networkx graph
        A trajectory to be layered on top of the base tree.
        The edges should be annotated with 'weight' and with 'state'.

    Returns
    -------
    T_merged : undirected weighted networkx graph
        A new tree with more states.

    """
    # All nodes in the base tree must be in the non-base trees.
    for T, name in ((T_current, 'current'), (T_traj, 'traj')):
        bad = set(T_base) - set(T)
        if bad:
            raise ValueError('the %s tree is missing the nodes %s '
                    'which are present in the base tree' % (name, sorted(bad)))

    # All non-degree-2 nodes in the non-base trees must be in the base tree.
    for T, name in ((T_current, 'current'), (T_traj, 'traj')):
        degree = T.degree()
        bad = set(n for n in set(T) - set(T_base) if degree[n] != 2)
        if bad:
            raise ValueError('the %s tree has the nodes %s '
                    'of degree other than two '
                    'which are not in the base tree' % (name, sorted(bad)))

    # Initialize the output graph.
    T_merged = nx.Graph()

    # Bookkeeping.
    successors = nx.dfs_successors(T_base, root)

    return T_merged
    
