"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
For continuous time processes (as opposed to discrete time processes)
use the Markov jump process module instead.
Everything related to hold times, dwell times, instantaneous rates,
total rates, and edges or lengths or expectations associated with edges
is out of the scope of this module.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn)

from raoteh.sampler import _mc0, _mcx, _mcy


__all__ = []


#XXX under destruction
def construct_node_to_restricted_pmap(
        T, root, node_to_allowed_states=None,
        P_default=None, states_default=None):
    if states_default is not None:
        raise NotImplementedError
    return _mcy.get_node_to_pmap(T, root,
            node_to_allowed_states=node_to_allowed_states,
            P_default=P_default)

#XXX under destruction
def xxx_construct_node_to_restricted_pmap(
        T, root, node_to_allowed_states=None,
        P_default=None, states_default=None):
    """
    For each node, construct the map from state to subtree likelihood.

    This function allows each node to be restricted to its own
    arbitrary set of allowed states.
    Applications include likelihood calculation,
    calculations of conditional expectations, and conditional state sampling.
    Some care is taken to distinguish between values that are zero
    because of structural reasons as opposed to values that are zero
    for numerical reasons.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices.
        The annotation uses the P attribute,
        and the transition matrices are themselves represented by
        networkx directed graphs with transition probabilities
        as the weight attribute of the edge.
    root : integer
        The root node.
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
    P_default : directed weighted acyclic networkx graph, optional
        A default transition matrix.
    states_default : dict, optional
        Nodes that are not in the node_to_allowed_states map
        will use this default set of allowed states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    # Input validation.
    if (node_to_allowed_states is None) and (states_default is None):
        raise ValueError('expected either a map from nodes to allowed states '
                'or a set of default allowed states or both')

    # Bookkeeping.
    if root not in T:
        raise ValueError('the specified root node is not in the tree')
    successors = nx.dfs_successors(T, root)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):

        # Get the set of valid states at this node.
        if node_to_allowed_states is None:
            valid_node_states = states_default
        else:
            valid_node_states = node_to_allowed_states.get(node, states_default)
        if valid_node_states is None:
            raise ValueError(
                    'the set of valid states is undefined at this node '
                    '(note that this is not the same as a defined but '
                    'empty set of valid states)')

        if node not in successors:
            node_to_pmap[node] = dict((s, 1.0) for s in valid_node_states)
        else:
            pmap = {}
            for node_state in valid_node_states:

                # Check for a structural subtree failure given this node state.
                structural_failure = False
                for n in successors[node]:

                    # Define the transition matrix according to the edge.
                    P = T[node][n].get('P', P_default)
                    if P is None:
                        raise ValueError('one of the edges is not annotated '
                                'with a transition matrix, and no default '
                                'transition matrix has been provided')

                    # Check that a transition away from the parent state
                    # is possible along this edge.
                    if node_state not in P:
                        structural_failure = True
                        break

                    # Get the list of possible child node states.
                    # These are limited by sparseness of the matrix of
                    # transitions from the parent state,
                    # and also by the possibility
                    # that the state of the child node is restricted.
                    valid_states = set(P[node_state]) & set(node_to_pmap[n])
                    if not valid_states:
                        structural_failure = True
                        break

                # If there is no structural failure or error,
                # then add the subtree probability to the node state pmap.
                if not structural_failure:
                    cprob = 1.0
                    for n in successors[node]:
                        P = T[node][n].get('P', P_default)
                        valid_states = set(P[node_state]) & set(node_to_pmap[n])
                        nprob = 0.0
                        for s in valid_states:
                            a = P[node_state][s]['weight']
                            b = node_to_pmap[n][s]
                            nprob += a * b
                        cprob *= nprob
                    pmap[node_state] = cprob

            # Add the map from state to subtree likelihood.
            node_to_pmap[node] = pmap

    # Return the map from node to the map from state to subtree likelihood.
    return node_to_pmap


def get_restricted_likelihood(T, root, node_to_allowed_states,
        root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    This is a general likelihood calculator for
    a Markov chain on tree-structured domains.
    At each node in the tree, the set of possible states may be restricted.
    Lack of state restriction at a node corresponds to missing data;
    a common example of such missing data would be missing states
    at internal nodes in a tree.
    Alternatively, a node could have a completely specified state,
    as could be the case if the state of the process is completely
    known at the tips of the tree.
    More generally, a node could be restricted to an arbitrary set of states.
    The first three args are used to construct a map from each node
    to a map from the state to the subtree likelihood,
    and the last arg defines the initial conditions at the root.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_allowed_states : dict
        A map from a node to a set of allowed states.
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
    likelihood : float
        The likelihood.

    """
    # Get likelihoods conditional on the root state.
    node_to_pmap = construct_node_to_restricted_pmap(
            T, root, node_to_allowed_states, P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Return the likelihood.
    return _mc0.get_likelihood(root_pmap, root_distn=root_distn)

