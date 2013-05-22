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
