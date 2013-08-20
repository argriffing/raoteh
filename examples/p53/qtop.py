"""
Functions that convert time intervals to a transition probability matrices.

The module name qtop is like Q to P where Q is the transition rate matrix 
and P is the transition probability matrix.
The interface functions should have t as the last argument
so that functools.partial() can be applied more easily to yield
functions that map a time interval to a transition probability matrix.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.linalg


###############################################################################
# The following three functions are meant to be called through
# functools.partial() so that the partial function will map a branch length
# to a probability transition matrix.

def getp_rate_matrix(Q, t):
    """
    Get a transition probability matrix using the undecomposed rate matrix.

    """
    return scipy.linalg.expm(Q * t)

def getp_sylvester(D0, D1, L, U0, U1, lam0, lam1, XQ, t):
    """
    Get a transition probability matrix using sylvester-like decomposition.

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    exp_lam0 = np.exp(t * lam0)
    exp_lam1 = np.exp(t * lam1)
    return get_reconstructed(D0, D1, L, U0, exp_lam0, exp_lam1, XQ)

def getp_spectral(d, U, w, t):
    """
    Get a transition probability matrix using the spectral decomposition.

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    V = np.dot(np.diag(np.sqrt(d)), U)
    exp_w = np.exp(t * w)
    X = np.dot(V, np.diag(np.sqrt(exp_w)))
    return np.dot(X, X.T)

