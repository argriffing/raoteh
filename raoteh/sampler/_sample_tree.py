"""
Sample networkx trees without edge weights.

These functions could eventually be merged into networkx.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx


__all__ = ['get_random_agglom_tree', 'get_random_branching_tree']


def get_random_agglom_tree(maxnodes):
    """
    Sample an unweighted tree by adding branches to random previous nodes.

    Parameters
    ----------
    maxnodes : integer
        Max number of nodes in the sampled tree.

    Returns
    -------
    T : undirected acyclic networkx graph
        This is a rooted tree with at least one edge.
        The root of the tree is node 0.

    """
    # Check the input.
    if maxnodes < 2:
        raise ValueError('maxnodes should be >= 2')

    # Initialize.
    T = nx.Graph()
    T.add_node(len(T))

    # Keep adding nodes until the cap is reached.
    while len(T) < maxnodes:
        na = np.random.randint(len(T))
        nb = len(T)
        T.add_edge(na, nb)
    return T


def get_random_branching_tree(branching_distn, maxnodes=None):
    """
    Sample an unweighted tree according to a random branching process.

    Start with a single node.
    Each node has a random number of descendents
    drawn from a discrete distribution.
    An extra descendent is added to the root node.

    Parameters
    ----------
    branching_distn : array
        This defines the distribution of the number of child nodes per node.
        It is a finite distribution over the first few non-negative integers.
    maxnodes : integer
        Cap the number of nodes in the tree.

    Returns
    -------
    T : undirected acyclic networkx graph
        This is a rooted tree with at least one edge.
        The root of the tree is node 0.

    """
    # Check the input.
    if (maxnodes is not None) and (maxnodes < 2):
        raise ValueError('if maxnodes is not None then it should be >= 2')

    # Initialize.
    T = nx.Graph()
    root = 0
    next_node = 0
    active_nodes = {0}

    # Keep adding nodes until the cap is reached or all lineages have ended.
    while active_nodes:
        node = active_nodes.pop()
        nbranches = np.random.choice(
                range(len(branching_distn)), p=branching_distn)
        if node == root:
            nbranches += 1
        for i in range(nbranches):
            c = next_node
            next_node += 1
            T.add_edge(node, c)
            active_nodes.add(c)
            if (maxnodes is not None) and (len(T) == maxnodes):
                return T
    return T

