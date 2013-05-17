"""
Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

__all__ = ['mysum']

# This is for checking the plumbing.
def mysum(*args):
    return sum(args)

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

