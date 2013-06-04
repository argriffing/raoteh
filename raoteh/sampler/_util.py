"""
Exception classes and utility functions for the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np
import networkx as nx
import scipy.linalg
import pyfelscore

__all__ = []


class ZeroProbError(Exception):
    pass

class StructuralZeroProb(ZeroProbError):
    pass

class NumericalZeroProb(ZeroProbError):
    pass

def get_first_element(elements):
    for x in elements:
        return x

def get_dense_rate_matrix(Q_sparse):
    """

    Parameters
    ----------
    Q_sparse : directed weighted networkx graph
        Sparse rate matrix without diagonal entries.

    Returns
    -------
    states : sequence of integers
        The sequence of states represented by rows and columns of the
        dense rate matrix.
    Q_dense : ndarray
        The dense rate matrix with informative diagonal.

    """
    states = sorted(Q_sparse)
    nstates = len(states)
    Q_dense = np.zeros((nstates, nstates), dtype=float)
    for a, sa in enumerate(states):
        for b, sb in enumerate(states):
            if Q_sparse.has_edge(sa, sb):
                edge = Q_sparse[sa][sb]
                Q_dense[a, b] = edge['weight']
    Q_dense = Q_dense - np.diag(np.sum(Q_dense, axis=1))
    return states, Q_dense

def _get_awr(Q):
    # Helper function.
    # This is in a tight inner loop so checking Q.has_edge is too slow.
    # Also (if 0 in Q and 1 in Q[0]) is too slow.
    # I can't figure out how to speed up this function,
    # so I'm changing it back to use has_edge.
    a = 0
    w = 0
    r = 0
    if Q.has_edge(0, 1):
        a = Q[0][1]['weight']
    if Q.has_edge(1, 0):
        w = Q[1][0]['weight']
    if Q.has_edge(1, 2):
        r = Q[1][2]['weight']
    return a, w, r

def sparse_expm(Q, t):
    # This is a wrapper that dispatches to either clever or naive expm.
    # Check if the matrix has a certain extremely restrictive known form.
    edges = Q.edges()
    has_nonneg_weights = all(Q[sa][sb]['weight'] >= 0 for sa, sb in edges)
    if has_nonneg_weights and (set(edges) <= set(((0, 1), (1, 0), (1, 2)))):
        return sparse_expm_mmpp_block(Q, t)
    else:
        return sparse_expm_naive(Q, t)

def sparse_expm_mmpp_block(Q, t):
    a, w, r = _get_awr(Q)
    if w:
        P = pyfelscore.get_mmpp_block(a, w, r, t)
    else:
        P = pyfelscore.get_mmpp_block_zero_off_rate(a, r, t)
    """
    if np.any(np.isnan(P)):
        print('got nan')
        print(a, w, r)
        print(P)
        print()
        raise Exception
    """
    P_nx = nx.DiGraph()
    # Add diagonal entries.
    P_nx.add_edge(0, 0, weight=P[0, 0])
    P_nx.add_edge(1, 1, weight=P[1, 1])
    P_nx.add_edge(2, 2, weight=1)
    # Conditionally add other entries.
    if a:
        P_nx.add_edge(0, 1, weight = P[0, 1])
    if a and r:
        P_nx.add_edge(0, 2, weight = 1 - P[0, 0] - P[0, 1])
    if w:
        P_nx.add_edge(1, 0, weight = P[1, 0])
    if r:
        P_nx.add_edge(1, 2, weight = 1 - P[1, 0] - P[1, 1])
    return P_nx


def sparse_expm_naive(Q, t):
    states = sorted(Q)
    n = len(states)
    Q_dense = np.zeros((n, n), dtype=float)
    for a, sa in enumerate(states):
        if sa in Q:
            for b, sb in enumerate(states):
                if sb in Q[sa]:
                    Q_dense[a, b] = Q[sa][sb]['weight']
    Q_dense = Q_dense - np.diag(np.sum(Q_dense, axis=1))
    P_dense = scipy.linalg.expm(t * Q_dense)
    path_lengths = nx.all_pairs_shortest_path_length(Q)
    P_nx = nx.DiGraph()
    for a, sa in enumerate(states):
        if sa in path_lengths:
            for b, sb in enumerate(states):
                if sb in path_lengths[sa]:
                    P_nx.add_edge(sa, sb, weight=P_dense[a, b])
    return P_nx

def expm_frechet_is_simple(Q):
    if len(Q) > 3:
        return False
    allowed_edges = ((0, 1), (1, 0), (1, 2))
    if Q.size() > len(allowed_edges):
        return False
    observed_edges = Q.edges()
    if not (set(observed_edges) <= set(allowed_edges)):
        return False
    a, w, r = _get_awr(Q)
    if a < 0 or r < 0 or w < 0:
        return False
    return True


def simple_expm_frechet(Q, ai, bi, ci, di, t):
    # validating the simpleness of the matrix is too slow
    a, w, r = _get_awr(Q)
    if w:
        return pyfelscore.get_mmpp_frechet_all_positive(
                a, w, r, t, ai, bi, ci, di)
    elif a != r:
        return pyfelscore.get_mmpp_frechet_diagonalizable_w_zero(
                a, r, t, ai, bi, ci, di)
    else:
        return pyfelscore.get_mmpp_frechet_defective_w_zero(
                a, t, ai, bi, ci, di)


def dict_random_choice(d):
    # This algorithm is not appropriate for multiple random samples
    # from the same dictionary.
    # If you want to do that, then use np.random.choice instead.
    #
    #choices, p = zip(*d.items())
    #return np.random.choice(choices, p=p)
    x = random.random()
    for i, p in d.items():
        x -= p
        if x < 0:
            return i

def get_normalized_dict_distn(d):
    if not d:
        raise StructuralZeroProb('cannot normalize an empty distribution')
    total_weight = sum(d.values())
    if not total_weight:
        raise NumericalZeroProb('the normalizing denominator is zero')
    return dict((k, v / total_weight) for k, v in d.items())


def get_arbitrary_tip(T, degrees=None):
    """

    Parameters
    ----------
    T : undirected networkx graph with integer nodes
        An input graph.
    degrees : dict, optional
        Maps nodes to degree.

    Returns
    -------
    tip : integer
        An arbitrary degree-1 node.

    """
    if degrees is None:
        degrees = T.degree()
    tips = (n for n, d in degrees.items() if d == 1)
    return get_first_element(tips)
