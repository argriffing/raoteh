"""
Generic functions relevant to algorithms involving Markov chains.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn,
        )


__all__ = []


def get_node_to_set(T, root, node_to_pset, P_default=None):
    """
    For each node get the set of states that give positive likelihood.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_pset : dict
        A map from a node to the set of states with positive subtree likelihood.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.

    Returns
    -------
    node_to_set : dict
        Maps each node to a set of states that give positive likelihood.

    Notes
    -----
    This function depends on the nature of the data observations
    only through the node_to_pset map.
    Another way of thinking about this function is that it gives
    the set of states that have positive posterior probability.

    """
    # Define the set for the root node.
    node_to_set = {root : node_to_pset[root]}

    # Define the set for the child node of each edge.
    for na, nb in nx.bfs_edges(T, root):

        # Construct the set of child states reachable from the
        # allowed parent states.
        P = T[na][nb].get('P', P_default)
        constraint_set = set()
        for sa in node_to_set[na]:
            constraint_set.update(P[sa])

        # Define the set of allowed child states.
        node_to_set[nb] = constraint_set & node_to_pset[nb]

    # Return the map.
    return node_to_set


def get_likelihood(root_pmap, root_distn=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    root_pmap : dict
        A map from states at the root to conditional likelihoods.
    root_distn : dict, optional
        A sparse finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    # Check whether the prior by itself causes the likelihood to be zero.
    if (root_distn is not None) and not root_distn:
        raise StructuralZeroProb('no root state has nonzero prior likelihood')

    # Check whether the likelihoods at the root, by themselves,
    # cause the likelihood to be zero.
    if not root_pmap:
        raise StructuralZeroProb(
                'all root states give a subtree likelihood of zero')

    # Construct the set of possible root states.
    # If no root state is possible raise the exception indicating
    # that the likelihood is zero by sparsity.
    feasible_rstates = set(root_pmap)
    if root_distn is not None:
        feasible_rstates.intersection_update(set(root_distn))
    if not feasible_rstates:
        raise StructuralZeroProb(
                'all root states have either zero prior likelihood '
                'or give a subtree likelihood of zero')

    # Compute the likelihood.
    if root_distn is not None:
        likelihood = sum(root_pmap[s] * root_distn[s] for s in feasible_rstates)
    else:
        likelihood = sum(root_pmap.values())

    # Return the likelihood.
    return likelihood

