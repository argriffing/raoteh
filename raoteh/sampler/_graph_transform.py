"""
Graph transformations related to the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = []



def get_first_element(elements):
    for x in elements:
        return x


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


def remove_redundant_nodes(T, redundant_nodes):
    """

    Parameters
    ----------
    T : networkx graph
        tree
    expendable_nodes : set of nodes
        candidates for removal
    """
    T = nx.Graph()
    for foo in bar:
        pass



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

    # Initialize the outputs.
    chunk_tree = nx.Graph()
    non_event_node_map = {}
    event_node_map = {}

    # Populate the outputs,
    # traversing the input graph in preorder from an arbitrary non-event root.
    next_chunk = 0
    if root is None:
        root = get_first_element(non_event_nodes)
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

