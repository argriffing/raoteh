"""
Sample Markov chain trajectories on trees.

Sample states using the observation data and constraints
only through a pre-computed sparse map, for each node, from a feasible state
to a subtree likelihood.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler import _mc0

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_unnormalized_dict_distn,
        dict_random_choice,
        )


__all__ = []


def resample_states(T, root, node_to_pmap, root_distn=None, P_default=None):
    """
    This function applies to a tree for which nodes will be assigned states.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_pmap : dict
        A map from a node to a sparse map from a feasible state
        to a subtree likelihood.
    root_distn : dict, optional
        A sparse finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.
    P_default : directed weighted networkx graph, optional
        If an edge is not annotated with a transition matrix P,
        then this default transition matrix will be used.

    Returns
    -------
    node_to_sampled_state : dict
        A map from each node of T to its state.
        If the state was not defined by the node_to_state argument,
        then the state will have been sampled.

    """
    # Get the root pmap.
    root_pmap = node_to_pmap[root]

    # Try to compute the likelihood.
    # This will raise an informative exception if no path is possible.
    # If the likelihood is numerically zero then raise a different exception.
    likelihood = _mc0.get_likelihood(root_pmap, root_distn=root_distn)
    if likelihood <= 0:
        raise NumericalZeroProb(
                'numerically intractably small likelihood: %s' % likelihood)

    # Bookkeeping structure related to tree traversal.
    predecessors = nx.dfs_predecessors(T, root)

    # Sample the node states, beginning at the root.
    node_to_sampled_state = {}
    for node in nx.dfs_preorder_nodes(T, root):

        # Get the precomputed pmap associated with the node.
        # This is a sparse map from state to subtree likelihood.
        pmap = node_to_pmap[node]

        # Define a prior distribution.
        if node == root:
            prior = root_distn
        else:

            # Get the parent node and its state.
            parent_node = predecessors[node]
            parent_state = node_to_sampled_state[parent_node]

            # Get the transition probability matrix.
            P = T[parent_node][node].get('P', P_default)

            # Get the distribution of a non-root node.
            sinks = set(P[parent_state]) & set(pmap)
            prior = dict((s, P[parent_state][s]['weight']) for s in sinks)

        # Sample the state from the posterior distribution.
        dpost = get_unnormalized_dict_distn(pmap, prior)
        node_to_sampled_state[node] = dict_random_choice(dpost)

    # Return the map of sampled states.
    return node_to_sampled_state

