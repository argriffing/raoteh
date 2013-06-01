"""
Exception classes and utility functions for the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

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

