"""
Special-cased linear algebra functions related to stochastic processes.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx
import scipy.linalg
import pyfelscore

__all__ = []

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

