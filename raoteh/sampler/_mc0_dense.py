"""
Generic functions relevant to algorithms involving Markov chains.

This module uses dense transition probability matrices.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx

import pyfelscore

from raoteh.sampler import _density

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_ndarray_distn)


__all__ = []


def get_example_transition_matrix():
    """
    Return a dense transition matrix for testing.

    Use a hack for the node indices, because I want to keep integers.
    This transition graph looks kind of like the following ascii art.

      41 --- 42 --- 43 --- 44
       |      |      |      |
      31 --- 32 --- 33 --- 34
       |      |      |      |
      21 --- 22 --- 23 --- 24
       |      |      |      |
      11 --- 12 --- 13 --- 14
     
    Returns
    -------
    P : 2d ndarray
        Transition probability matrix.
    state_names : integer sequence
        A sequence of state names.
        Because this module works with dense matrices,
        the states are the indices (as opposed to the names).

    """
    # Define the states.
    state_names = (
            11, 12, 13, 14,
            21, 22, 23, 24,
            31, 32, 33, 34,
            41, 42, 43, 44)
    nstates = len(state_names)
    name_to_state = ((name, s) for s, name in enumerate(state_names))

    # Define the transition matrix.
    P = np.zeros((nstates, nstates), dtype=float)
    for i in (1, 2, 3, 4):
        for j in (1, 2, 3, 4):
            source_name = i*10 + j
            source = name_to_state[source_name]
            sinks = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = i + di
                    nj = j + dj
                    if not (di and dj):
                        if (1 <= ni <= 4) and (1 <= nj <= 4):
                            sink_name = ni*10 + nj
                            sinks.append(name_to_state[sink_name])
            nsinks = len(sinks)
            prob = 1 / float(nsinks)
            for sink in sinks:
                P[source, sink] = prob
    return P, state_names


def get_history_log_likelihood(T, root, node_to_state,
        root_distn=None, P_default=None):
    """
    Compute the log likelihood for a fully augmented history.

    This is for a discrete-time chain so it does not include dwell times.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree optionally annotated with transition matrices.
    root : integer
        Root node.
    node_to_state : dict
        Each node in the tree is mapped to an integer state.
    root_distn : 1d ndarray, optional
        Prior distribution over states at the root.
    P_default : 2d ndarray, optional
        A default probability transition matrix.

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
    if P_default is not None:
        _density.check_square_dense(P_default)

    # Compute the log likelihood contribution from the root.
    ll_initial = 0
    root_state = node_to_state[root]
    if root_distn is not None:
        if not root_distn[root_state]:
            raise StructuralZeroProb('zero prior for the root')
        ll_initial = np.log(root_distn[root_state])

    # Compute the log likelihood contribution from state transitions.
    ll_transitions = 0
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        P = edge.get('P', P_default)
        _density.check_square_dense(P)
        sa = node_to_state[na]
        sb = node_to_state[nb]
        p = P[sa, sb]
        if not p:
            raise StructuralZeroProb(
                    'the states of the endpoints of an edge '
                    'are incompatible with the transition matrix on the edge')
        ll_transitions += np.log(p)

    # Return the log likelihood.
    log_likelihood = ll_initial + ll_transitions
    return log_likelihood


def get_likelihood(root_pmap, root_distn=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    root_pmap : 1d ndarray
        An ndarry giving conditional subtree likelihoods at the root.
    root_distn : 2d ndarray, optional
        A finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    # Check for shape compatibility.
    if root_distn is not None:
        if root_pmap.shape != root_distn.shape:
            raise ValueError('root shape mismatch: '
                    '%s %s' % (root_pmap.shape, root_distn.shape))

    # Check whether the prior by itself causes the likelihood to be zero.
    if root_distn is not None:
        prior_feasible_rstates = set(s for s, p in enumerate(root_distn) if p)
        if not prior_feasible_rstates:
            raise StructuralZeroProb(
                    'no root state has nonzero prior likelihood')

    # Check whether the likelihoods at the root, by themselves,
    # cause the likelihood to be zero.
    if root_pmap is None:
        raise ValueError('root_pmap is None')
    if root_pmap.min() < 0:
        raise ValueError('root_pmap should have non-negative entries')
    if not root_pmap.sum():
        raise StructuralZeroProb(
                'all root states give a subtree likelihood of zero')

    # Construct the set of possible root states.
    # If no root state is possible raise the exception indicating
    # that the likelihood is zero by sparsity.
    feasible_rstates = set(s for s, p in enumerate(root_pmap) if p)
    if root_distn is not None:
        feasible_rstates.intersection_update(prior_feasible_rstates)
    if not feasible_rstates:
        raise StructuralZeroProb(
                'all root states have either zero prior likelihood '
                'or give a subtree likelihood of zero')

    # Compute the likelihood.
    if root_distn is not None:
        likelihood = root_distn.dot(root_pmap)
    else:
        likelihood = root_pmap.sum()

    # Return the likelihood.
    return likelihood


#TODO Add a cython implementation,
#TODO and also add more tests.
def get_joint_endpoint_distn(T, root, node_to_pmap, node_to_distn, nstates):
    """

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree with edges annotated with sparse transition probability
        matrices as 2d ndarrays P.
    root : integer
        Root node.
    node_to_pmap : dict
        Map from a node to a 1d array giving subtree likelihoods per state.
        This map incorporates state restrictions.
    node_to_distn : dict
        Conditional marginal state distribution at each node.
    nstates : integer
        Number of states.

    Returns
    -------
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as 2d ndarrays.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is written in
        single letter caps reminiscent of matrix notation.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        pmap = node_to_pmap[nb]
        P = T[na][nb]['P']
        _density.check_square_dense(P)
        J = np.zeros_like(P)
        distn = node_to_distn[na]
        if distn.shape[0] != nstates:
            raise Exception('nstates inconsistency')
        for sa in range(nstates):
            pa = distn[sa]
            if pa:

                # Construct the conditional transition probabilities.
                sb_weights = P[sa] * pmap
                sb_distn = get_normalized_ndarray_distn(sb_weights)

                # Add to the joint distn.
                for sb, pb in enumerate(sb_distn):
                    J[sa, sb] = pa * pb

        # Add the joint distribution.
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
    root_distn : 1d ndarray, optional
        A finite distribution over root states.
    P_default : 2d ndarray, optional
        A default probability transition matrix.

    Returns
    -------
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as 2d ndarrays.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is writeen in
        single letter caps reminiscent of matrix notation.

    """
    if P_default is not None:
        _density.check_square_dense(P_default)

    # Initialize J arrays on the edges.
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        P = T[na][nb]['P']
        _density.check_square_dense(P)
        J = np.zeros_like(P)
        T_aug.add_edge(na, nb, J=J)

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
            J[sa, sb] += likelihood

    # For each edge, normalize the distribution over ordered state pairs.
    for na, nb in nx.bfs_edges(T, root):
        J = T_aug[na][nb]['J']
        J /= J.sum()
    
    # Return the tree with the sparse joint distributions on edges.
    return T_aug


#TODO Add a cython implementation,
#TODO and also add more tests.
def get_node_to_distn(T, root, node_to_pmap, nstates,
        root_distn=None, P_default=None):
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
        Map from a node to a 1d array giving subtree likelihoods per state.
        This map incorporates state restrictions.
    nstates : integer
        Number of states.
    root_distn : 1d ndarray, optional
        A finite distribution over root states.
    P_default : 2d ndarray, optional
        Default transition matrix.

    Returns
    -------
    node_to_distn : dict
        Sparse map from node to sparse map from state to probability.

    """
    if P_default is not None:
        _density.check_square_dense(P_default)
    if root_distn is not None:
        if root_distn.shape[0] != nstates:
            raise ValueError('inconsistent root distribution')

    # Bookkeeping.
    predecessors = nx.dfs_predecessors(T, root)

    # Get the distributions.
    node_to_distn = {}
    for node in nx.dfs_preorder_nodes(T, root):

        # Get the map from state to subtree likelihood.
        pmap = node_to_pmap[node]
        if pmap.shape[0] != nstates:
            raise ValueError('inconsistent pmap')

        # Compute the prior distribution at the root separately.
        # If the prior distribution is not provided,
        # then treat it as uninformative.
        if node == root:
            distn = get_normalized_ndarray_distn(pmap, root_distn)
        else:
            parent_node = predecessors[node]
            parent_distn = node_to_distn[parent_node]

            # Get the transition matrix associated with this edge.
            P = T[parent_node][node].get('P', P_default)
            _density.check_square_dense(P)
            if P.shape[0] != nstates:
                raise Exception('internal inconsistency')

            # For each parent state,
            # get the distribution over child states;
            # this distribution will include both the P matrix
            # and the pmap of the child node.
            distn = np.zeros(nstates, dtype=float)
            for sa in range(nstates):
                pa = parent_distn[sa]
                if pa:

                    # Construct the conditional transition probabilities.
                    sb_weights = P[sa] * pmap
                    sb_distn = get_normalized_ndarray_distn(sb_weights)

                    # Add to the marginal distn.
                    for sb in range(nstates):
                        distn[sb] += pa * sb_distn[sb]

        # Set the node_to_distn.
        node_to_distn[node] = distn

    # Return the marginal state distributions at nodes.
    return node_to_distn


def get_node_to_distn_naive(T, root, node_to_set, nstates,
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
    nstates : integer
        Number of states.
    root_distn : 1d ndarray, optional
        A finite distribution over root states.
    P_default : 2d ndarray, optional
        A default universal probability transition matrix.

    Returns
    -------
    node_to_distn : dict
        Maps each node to a distribution over states.

    """
    if P_default is not None:
        _density.check_square_dense(P_default)
    if root_distn is not None:
        if root_distn.shape[0] != nstates:
            raise ValueError('inconsistent root distribution')

    nodes, allowed_states = zip(*node_to_set.items())
    node_to_weights = dict((n, np.zeros(nstates, dtype=float)) for n in nodes)
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
            weights = node_to_weights[node]
            weights[state] += likelihood

    # For each node, normalize the distribution over states.
    node_to_distn = {}
    for node in nodes:
        weights = node_to_weights[node]
        node_to_distn[node] = get_normalized_ndarray_distn(weights)

    # Return the map from node to distribution.
    return node_to_distn

