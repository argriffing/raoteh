"""
Exception classes and utility functions for the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import


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

