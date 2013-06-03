"""
Exception classes and utility functions for the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx
import scipy.linalg

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


def sparse_expm(Q, t):
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


def dict_random_choice(d):
    choices, p = zip(*d.items())
    return np.random.choice(choices, p=p)


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
