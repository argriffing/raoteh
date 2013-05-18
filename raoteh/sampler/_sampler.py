"""
Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx

__all__ = ['mysum']


class SamplingError(Exception):
    pass

class SamplingInfeasibility(Exception):
    pass


# This is for checking the plumbing.
def mysum(*args):
    return sum(args)


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


def resample_states(T, P, node_to_state, root=None, root_distn=None):
    """

    Parameters
    ----------
    T : unweighted undirected acyclic networkx graph
        States do not change within chunks represented by nodes in this tree.
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
    node_to_sampled_state : dict
        A map from each node of T to its state.
        If the state was not defined by the node_to_state argument,
        then the state will have been sampled.

    Notes
    -----
    Nodes other than tips of the tree may have known states.
    If no root is provided,
    then an arbitrary node with a known state will be chosen as the root.

    """

    # If the root has not been provided, then pick one with a known state.
    if root is None:
        root = get_first_element(node_to_state)

    # Bookkeeping structures related to tree traversal.
    successors = nx.dfs_successors(T, root)
    predecessors = nx.dfs_predecessors(T, root)

    # A bookkeeping structure related to state sampling.
    P_for_sampling = {}
    for source in P:
        sinks = []
        probs = []
        for sink in P[source]:
            sinks.append(sink)
            probs.append(P[source][sink]['weight'])
        P_for_sampling[source] = (sinks, probs)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):
        if node in node_to_state:
            node_state = node_to_state[node]
            node_to_pmap[node] = {node_state : 1.0}
        else:
            pmap = {}
            for node_state in P:
                cprob = 1.0
                for n in successors[node]:

                    # Get the list of possible child node states.
                    # These are limited by sparseness of the matrix of
                    # transitions from the parent state,
                    # and also by the possibility
                    # that the state of the child node is already known.
                    valid_states = set(P[node_state]) & set(node_to_pmap[n])
                    nprob = 0.0
                    for s in valid_states:
                        nprob += P[node_state][s]['weight'] * node_to_pmap[n][s]
                    cprob *= nprob
                pmap[node_state] = cprob
            node_to_pmap[node] = pmap

    # Sample the node states, beginning at the root.
    node_to_sampled_state = {}

    # Treat the root separately.
    # If only one state is possible at the root, then we do not have to sample.
    # Otherwise consult the map from root states to probabilities.
    if len(node_to_pmap[root]) == 1:
        root_state = get_first_element(node_to_pmap[root])
        if not node_to_pmap[root][root_state]:
            raise SamplingInfeasibility(
                    'the only feasible state at the root '
                    'gives a subtree probability of zero')
    else:
        if root_distn is None:
            raise ValueError('expected a distribution over states at the root')
        prior_distn = root_distn
        states = list(set(prior_distn) & set(node_to_pmap[root]))
        if not states:
            raise SamplingInfeasibility('no root is feasible')
        weights = []
        for s in states:
            weights.append(prior_distn[s] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise SamplingInfeasibility('numerical problem at the root')
        probs = np.array(weights, dtype=float) / weight_sum
        sampled_state = np.random.choice(states, p=probs)
        root_state = sampled_state
    node_to_sampled_state[root] = root_state

    # Sample the states at the rest of the nodes.
    for node in nx.dfs_preorder_nodes(T, root):

        # The root has already been sampled.
        if node == root:
            continue

        # Get the parent node and its state.
        parent_node = predecessors[node]
        parent_state = node_to_sampled_state[parent_node]

        # Sample the state of a non-root node.
        # A state is possible if it is reachable in one step from the
        # parent state which has already been sampled
        # and if it gives a subtree probability that is not structurally zero.
        states = list(set(P[parent_state]) & set(node_to_pmap[node]))
        if not states:
            raise SamplingInfeasibility('found a non-root infeasibility')
        weights = []
        for s in states:
            weights.append(P[parent_state][s]['weight'] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise SamplingInfeasibility('numerical problem at a non-root node')
        probs = np.array(weights, dtype=float) / weight_sum
        sampled_state = np.random.choice(states, p=probs)
        node_to_sampled_state[node] = sampled_state

    # Return the map of sampled states.
    return node_to_sampled_state



def get_feasible_history(rate_matrix, tip_states, tree):
    """
    Find an arbitrary feasible history.

    Parameters
    ----------
    rate_matrix : weighted directed networkx graph
        This is a potentially sparse rate matrix.
    tip_states : dict
        Maps tip vertex ids to states.
    tree : weighted undirected networkx graph
        A representation of a tree.  Following networks convention,
        the distance along an edge is its 'weight'.

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

    """
    pass

