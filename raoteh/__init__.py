"""
raoteh __init__ docstring

"""
from __future__ import division, print_function, absolute_import

__all__ = ['test', 'bench']


from numpy.testing import Tester
test = Tester().test
bench = Tester().bench

