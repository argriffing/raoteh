"""
Generic functions relevant to algorithms involving Markov chains.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx

import pyfelscore

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn,
        )


__all__ = []


def get_example_transition_matrix():
    # This returns a sparse transition matrix for testing.
    # It uses an hack for the node indices, because I want to keep integers.
    # This transition graph looks kind of like the following ascii art.
    #
    # 41 --- 42 --- 43 --- 44
    #  |      |      |      |
    # 31 --- 32 --- 33 --- 34
    #  |      |      |      |
    # 21 --- 22 --- 23 --- 24
    #  |      |      |      |
    # 11 --- 12 --- 13 --- 14
    #
    P = nx.DiGraph()
    weighted_edges = []
    for i in (1, 2, 3, 4):
        for j in (1, 2, 3, 4):
            source = i*10 + j
            sinks = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = i + di
                    nj = j + dj
                    if not (di and dj):
                        if (1 <= ni <= 4) and (1 <= nj <= 4):
                            sink = ni*10 + nj
                            sinks.append(sink)
            nsinks = len(sinks)
            weight = 1 / float(nsinks)
            for sink in sinks:
                weighted_edges.append((source, sink, weight))
    P.add_weighted_edges_from(weighted_edges)
    return P


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
    return get_node_to_set_unaccelerated(
            T, root, node_to_pset, P_default=P_default)


def get_node_to_set_unaccelerated(T, root, node_to_pset, P_default=None):
    """
    For each node get the set of states that give positive likelihood.

    Does not use pyfelscore acceleration.

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


def get_history_log_likelihood(T, root, node_to_state,
        root_distn=None, P_default=None):
    """
    Compute the log likelihood for a fully augmented history.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree optionally annotated with transition matrices.
    root : integer
        Root node.
    node_to_state : dict
        Each node in the tree is mapped to an integer state.
    root_distn : dict, optional
        Sparse prior distribution over states at the root.
    P_default : weighted directed networkx graph, optional
        A default universal probability transition matrix.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the fully augmented history.

    """
    # Input validation.
    bad = set(T) - set(node_to_state)
    if bad:
        raise ValueError(
                'to compute the history log likelihood all nodes in the tree '
                'must have a known state, but this state has not been '
                'provided for the following nodes: ' + str(sorted(bad)))

    # Initialize the log likelihood.
    log_likelihood = 0

    # Add the log likelihood contribution from the root.
    root_state = node_to_state[root]
    if root_distn is not None:
        if root_state not in root_distn:
            raise StructuralZeroProb('zero prior for the root')
        log_likelihood += np.log(root_distn[root_state])

    # Add the log likelihood contribution from state transitions.
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        P = edge.get('P', P_default)
        if P is None:
            raise ValueError('undefined transition matrix on this edge')
        sa = node_to_state[na]
        sb = node_to_state[nb]
        if not P.has_edge(sa, sb):
            raise StructuralZeroProb(
                    'the states of the endpoints of an edge '
                    'are incompatible with the transition matrix on the edge')
        p = P[sa][sb]['weight']
        log_likelihood += np.log(p)

    # Return the log likelihood.
    return log_likelihood


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
    if root_pmap is None:
        raise ValueError('root_pmap is None')
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


def get_joint_endpoint_distn(T, root, node_to_pmap, node_to_distn):
    """

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree with edges annotated with sparse transition probability
        matrices as directed weighted networkx graphs P.
    root : integer
        Root node.
    node_to_pmap : dict
        Map from node to a map from a state to the subtree likelihood.
        This map incorporates state restrictions.
    node_to_distn : dict
        Conditional marginal state distribution at each node.

    Returns
    -------
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as networkx digraphs.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is written in
        single letter caps reminiscent of matrix notation.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        pmap = node_to_pmap[nb]
        P = T[na][nb]['P']
        weighted_edges = []
        for sa, pa in node_to_distn[na].items():

            # Construct the conditional transition probabilities.
            feasible_sb = set(P[sa]) & set(pmap)
            sb_weights = {}
            for sb in feasible_sb:
                a = P[sa][sb]['weight']
                b = pmap[sb]
                sb_weights[sb] = a*b
            tot = sum(sb_weights.values())
            sb_distn = dict((sb, w / tot) for sb, w in sb_weights.items())

            # Add to the joint distn.
            for sb, pb in sb_distn.items():
                weighted_edges.append((sa, sb, pa * pb))

        # Add the joint distribution.
        J = nx.DiGraph()
        J.add_weighted_edges_from(weighted_edges)
        T_aug.add_edge(na, nb, J=J)

    # Return the augmented tree.
    return T_aug


def get_joint_endpoint_distn_naive(T, root, node_to_set,
        root_distn=None, P_default=None):
    """

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        Root node.
    node_to_set : dict
        Map from node to collection of allowed states.
    root_distn : dict, optional
        A finite distribution over root states.
    P_default : weighted directed networkx graph, optional
        A default universal probability transition matrix.

    Returns
    -------
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as networkx digraphs.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is writeen in
        single letter caps reminiscent of matrix notation.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        T_aug.add_edge(na, nb, J=nx.DiGraph())
    nodes, allowed_states = zip(*node_to_set.items())
    for assignment in itertools.product(*allowed_states):

        # Get the map corresponding to the assignment.
        node_to_state = dict(zip(nodes, assignment))

        # Compute the log likelihood for the assignment.
        # If the log likelihood cannot be computed,
        # then skip to the next state assignment.
        try:
            ll = get_history_log_likelihood(T, root, node_to_state,
                    root_distn=root_distn, P_default=P_default)
        except StructuralZeroProb as e:
            continue

        # Add the likelihood to weights of ordered node pairs on edges.
        likelihood = np.exp(ll)
        for na, nb in nx.bfs_edges(T, root):
            J = T_aug[na][nb]['J']
            sa = node_to_state[na]
            sb = node_to_state[nb]
            if not J.has_edge(sa, sb):
                J.add_edge(sa, sb, weight=0.0)
            J[sa][sb]['weight'] += likelihood

    # For each edge, normalize the distribution over ordered state pairs.
    for na, nb in nx.bfs_edges(T, root):
        J = T_aug[na][nb]['J']
        weights = []
        for sa, sb in J.edges():
            weights.append(J[sa][sb]['weight'])
        if not weights:
            raise Exception('internal error')
        total_weight = np.sum(weights)
        for sa, sb in J.edges():
            J[sa][sb]['weight'] /= total_weight
    
    # Return the tree with the sparse joint distributions on edges.
    return T_aug


def get_node_to_distn(T, root, node_to_pmap, root_distn=None, P_default=None):
    """
    Get marginal state distributions at nodes in a tree.

    This function is similar to the Rao-Teh state sampling function,
    except that instead of sampling a state at each node,
    this function computes marginal distributions over states at each node.
    Also, each edge of the input tree for this function has been
    annotated with its own transition probability matrix,
    whereas the Rao-Teh sampling function uses a single
    uniformized transition probability matrix for all edges.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        Root node.
    node_to_pmap : dict
        Map from node to a map from a state to the subtree likelihood.
        This map incorporates state restrictions.
    root_distn : dict, optional
        A finite distribution over root states.
    P_default : weighted directed networkx graph, optional
        Default transition matrix.

    Returns
    -------
    node_to_distn : dict
        Sparse map from node to sparse map from state to probability.

    """
    # Bookkeeping.
    predecessors = nx.dfs_predecessors(T, root)

    # Get the distributions.
    node_to_distn = {}
    for node in nx.dfs_preorder_nodes(T, root):

        # Get the map from state to subtree likelihood.
        pmap = node_to_pmap[node]

        # Compute the prior distribution at the root separately.
        # If the prior distribution is not provided,
        # then treat it as uninformative.
        if node == root:
            distn = get_normalized_dict_distn(pmap, root_distn)
        else:
            parent_node = predecessors[node]
            parent_distn = node_to_distn[parent_node]

            # Get the transition matrix associated with this edge.
            P = T[parent_node][node].get('P', P_default)
            if P is None:
                raise ValueError('no transition matrix is available')

            # For each parent state,
            # get the distribution over child states;
            # this distribution will include both the P matrix
            # and the pmap of the child node.
            distn = defaultdict(float)
            for sa, pa in parent_distn.items():

                # Construct the conditional transition probabilities.
                feasible_sb = set(P[sa]) & set(node_to_pmap[node])
                sb_weights = {}
                for sb in feasible_sb:
                    a = P[sa][sb]['weight']
                    b = node_to_pmap[node][sb]
                    sb_weights[sb] = a*b
                sb_distn = get_normalized_dict_distn(sb_weights)

                # Add to the marginal distn.
                for sb, pb in sb_distn.items():
                    distn[sb] += pa * pb

        # Set the node_to_distn.
        node_to_distn[node] = distn

    # Return the marginal state distributions at nodes.
    return node_to_distn


def get_node_to_distn_naive(T, root, node_to_set,
        root_distn=None, P_default=None):
    """
    Get marginal state distributions at nodes in a tree.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        Root node.
    node_to_set : dict
        Map from node to collection of allowed states.
    root_distn : dict, optional
        A finite distribution over root states.
    P_default : weighted directed networkx graph, optional
        A default universal probability transition matrix.

    Returns
    -------
    node_to_distn : dict
        Maps each node to a distribution over states.

    """
    nodes, allowed_states = zip(*node_to_set.items())
    node_to_state_to_weight = dict((n, defaultdict(float)) for n in nodes)
    for assignment in itertools.product(*allowed_states):

        # Get the map corresponding to the assignment.
        node_to_state = dict(zip(nodes, assignment))

        # Compute the log likelihood for the assignment.
        # If the log likelihood cannot be computed,
        # then skip to the next state assignment.
        try:
            ll = get_history_log_likelihood(T, root, node_to_state,
                    root_distn=root_distn, P_default=P_default)
        except StructuralZeroProb as e:
            continue

        # Add the likelihood to weights of states assigned to nodes.
        likelihood = np.exp(ll)
        for node, state in zip(nodes, assignment):
            d = node_to_state_to_weight[node]
            d[state] += likelihood

    # For each node, normalize the distribution over states.
    node_to_distn = {}
    for node in nodes:
        d = node_to_state_to_weight[node]
        node_to_distn[node] = get_normalized_dict_distn(d)

    # Return the map from node to distribution.
    return node_to_distn

