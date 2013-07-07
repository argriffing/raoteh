"""
Exception classes and utility functions for the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import random

import numpy as np

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


def dict_random_choice(d):
    # This algorithm is not appropriate for multiple random samples
    # from the same dictionary.
    # If you want to do that, then use numpy random.choice instead.
    if not d:
        raise ValueError('the dictionary is empty')
    total = sum(d.values())
    if total <= 0:
        raise ValueError('weight is not positive')
    x = random.random() * total
    for i, w in d.items():
        x -= w
        if x < 0:
            return i


def get_unnormalized_dict_distn(d, prior=None):
    if d is None:
        raise ValueError('d is None')
    if not d:
        raise StructuralZeroProb('the main dict of weights is empty')
    if prior is None:
        return d
    else:
        if not prior:
            raise StructuralZeroProb('empty prior')
        states = set(d) & set(prior)
        if not states:
            raise StructuralZeroProb('empty intersection of main and prior')
        return dict((k, d[k] * prior[k]) for k in states)


def get_normalized_dict_distn(d, prior=None):
    dpost = get_unnormalized_dict_distn(d, prior)
    total_weight = sum(dpost.values())
    if not total_weight:
        raise NumericalZeroProb('the denominator is zero')
    return dict((k, v / total_weight) for k, v in dpost.items())


def get_unnormalized_ndarray_distn(d, prior=None):
    """

    Parameters
    ----------
    d : 1d ndarray
        first array
    prior : 1d ndarray, optional
        second array

    Returns
    -------
    out : 1d ndarray
        output array

    """
    if d.min() < 0:
        raise ValueError('expected non-negative entries')
    if prior is None:
        return d
    else:
        if prior.min() < 0:
            raise ValueError('expected non-negative prior entries')
        return d * prior


def get_normalized_ndarray_distn(d, prior=None):
    """

    Parameters
    ----------
    d : 1d ndarray
        first array
    prior : 1d ndarray, optional
        second array

    Returns
    -------
    out : 1d ndarray
        output array

    """
    dpost = get_unnormalized_ndarray_distn(d, prior)
    total_weight = dpost.sum()
    if not total_weight:
        raise NumericalZeroProb('the denominator is zero')
    return dpost / total_weight


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


def _check_root(T, root):
    if root not in T:
        raise Exception('internal error: the root is not in the tree')


class cached_property(object):
    """
    This is from the internet.

    A read-only @property that is only evaluated once. The value is cached
    on the object itself rather than the function or class; this should prevent
    memory leakage.
    """
    def __init__(self, fget, doc=None):
        self.fget = fget
        self.__doc__ = doc or fget.__doc__
        self.__name__ = fget.__name__
        self.__module__ = fget.__module__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        obj.__dict__[self.__name__] = result = self.fget(obj)
        return result

