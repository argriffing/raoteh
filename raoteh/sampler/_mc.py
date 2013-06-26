"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
For continuous time processes (as opposed to discrete time processes)
use the Markov jump process module instead.
Everything related to hold times, dwell times, instantaneous rates,
total rates, and edges or lengths or expectations associated with edges
is out of the scope of this module.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn)

from raoteh.sampler import _mc0, _mcx, _mcy


__all__ = []


