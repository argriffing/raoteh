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

from numpy.testing import run_module_suite, assert_equal, assert_allclose


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
    return reconstruct_sylvester(D0, D1, L, U0, exp_lam0, exp_lam1, XQ)

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


###############################################################################
# Miscellaneous helper functions.


###############################################################################
# Decomposition helper functions.


###############################################################################
# Reconstruction helper functions.

def ndot(*args):
    M = args[0]
    for B in args[1:]:
        M = np.dot(M, B)
    return M

def build_block_2x2(M):
    return np.vstack([np.hstack(M[0]), np.hstack(M[1])])

def reconstruct_sylvester(
        D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    """
    Return the reconstructed matrix given a spectral form.
    """
    R11 = ndot(
            np.diag(np.reciprocal(np.sqrt(D0))),
            U0,
            np.diag(lam0),
            U0.T,
            np.diag(np.reciprocal(D0)),
            )
    R22 = ndot(
            np.diag(np.reciprocal(np.sqrt(D1))),
            U1,
            np.diag(lam1),
            U1.T,
            np.diag(np.reciprocal(D1)),
            )
    Q_reconstructed = build_block_2x2([
        [R11, ndot(R11, XQ) - ndot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed


###############################################################################
# Tests.




if __name__ == '__main__':
    run_module_suite()

