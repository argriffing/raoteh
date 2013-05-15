"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase, assert_equal)

from raoteh.sampler import _sampler


class TestSampler(TestCase):
    
    def test_sum(self):
        assert_equal(_sampler.mysum(2, 3), 5)


if __name__ == '__main__':
    run_module_suite()
