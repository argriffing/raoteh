"""
sampler __init__ docstring

"""
from __future__ import division, print_function, absolute_import

from ._sampler import *

__all__ = [s for s in dir() if not s.startswith('_')]

from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

