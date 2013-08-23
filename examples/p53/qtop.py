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

def getp_spectral(D, U, lam, t):
    """
    Get a transition probability matrix using the spectral decomposition.

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    exp_lam = np.exp(t * lam)
    return reconstruct_spectral(D, U, exp_lam)


###############################################################################
# Miscellaneous helper functions.

def ndot(*args):
    M = args[0]
    for B in args[1:]:
        M = np.dot(M, B)
    return M

def build_block_2x2(A):
    return np.vstack([np.hstack(A[0]), np.hstack(A[1])])

def pseudo_reciprocal(A):
    # Zeros are only ignored when they are exactly zero.
    A = np.asarray(A)
    v = [1/x if x else 0 for x in A.flat]
    return np.array(v, dtype=A.dtype).reshape(A.shape)


###############################################################################
# Decomposition helper functions.

def decompose_spectral(S, D):
    """
    Decompose a time-reversible rate matrix.

    The rate matrix should be represented as Q = dot(S, diag(D)).
    The matrix S should be symmetric,
    and D should be a non-negative 1d array representing a diagonal matrix.

    """
    D_sqrt = np.sqrt(D)
    B = ndot(np.diag(D_sqrt), S, np.diag(D_sqrt))
    lam, U = scipy.linalg.eigh(B)
    return D, U, lam

def decompose_sylvester(S0, S1, D0, D1, L):
    """
    Decompose a rate matrix using the Sylvester equation.

    Disregarding diagonal elements, the reference process rate matrix
    should be dot(S0, diag(D0)).
    Again disregarding diagonal elements, the default process rate matrix
    should be dot(S1, diag(D1)).
    The matrices S0 and S1 should be symmetric,
    and D0, D1, and L are non-negative 1d arrays
    representing diagonal matrices.

    """
    # compute the first symmetric eigendecomposition
    D0_sqrt = np.sqrt(D0)
    H0 = ndot(np.diag(D0_sqrt), S0, np.diag(D0_sqrt)) - np.diag(L)
    lam0, U0 = scipy.linalg.eigh(H0)

    # compute the second symmetric eigendecomposition
    D1_sqrt = np.sqrt(D1)
    H1 = ndot(np.diag(D1_sqrt), S1, np.diag(D1_sqrt))
    lam1, U1 = scipy.linalg.eigh(H1)

    # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
    A = ndot(S0, np.diag(D0)) - np.diag(L)
    B = -ndot(S1, np.diag(D1))
    Q = np.diag(L)
    XQ = scipy.linalg.solve_sylvester(A, B, Q)

    # Return the decomposition.
    return D0, D1, L, U0, U1, lam0, lam1, XQ


###############################################################################
# Reconstruction helper functions.

def reconstruct_spectral(D, U, lam):
    """
    Return the reconstructed matrix given a spectral form.

    """
    Q_reconstructed = ndot(
            np.diag(pseudo_reciprocal(np.sqrt(D))),
            U,
            np.diag(lam),
            U.T,
            np.diag(np.sqrt(D)),
            )
    return Q_reconstructed

def reconstruct_sylvester(D0, D1, L, U0, U1, lam0, lam1, XQ):
    """
    Return the reconstructed matrix given a spectral form.

    """
    R11 = ndot(
            np.diag(pseudo_reciprocal(np.sqrt(D0))),
            U0,
            np.diag(lam0),
            U0.T,
            np.diag(np.sqrt(D0)),
            )
    R22 = ndot(
            np.diag(pseudo_reciprocal(np.sqrt(D1))),
            U1,
            np.diag(lam1),
            U1.T,
            np.diag(np.sqrt(D1)),
            )
    Q_reconstructed = build_block_2x2([
        [R11, ndot(R11, XQ) - ndot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed



###############################################################################
# Tests and test helper functions.

def assert_SD_reversible_rate_matrix(S, D):
    Q = np.dot(S, np.diag(D))
    assert_allclose(Q.sum(axis=1), 0, atol=1e-15)
    assert_allclose(D.sum(), 1)
    assert_allclose(S, S.T)
    assert_allclose(np.dot(np.diag(D), Q), np.dot(Q.T, np.diag(D)))


def random_reversible_rate_matrix(n):
    # Construct a random time-reversible rate matrix.
    # The first state will have zero stationary probability.
    if n < 1:
        raise ValueError('expected at least one state')

    # Construct random positive symmetric rates.
    S = np.square(np.random.randn(n, n))
    S = S + S.T

    # Set diagonal entries so that rows sum to zero.
    S = S - np.diag(S.sum(axis=1))
    
    # Zero the first row and column of rates.
    S[0, :] = 0
    S[:, 0] = 0

    # Sample a random stationary distribution,
    # forcing the first state to have zero probability.
    D = np.square(np.random.randn(n))
    D[0] = 0
    D /= D.sum()

    # Construct the time-reversible rate matrix,
    # defining diagonal entries such that rows sum to zero.
    pre_Q = np.dot(S, np.diag(D))
    Q = pre_Q - np.diag(pre_Q.sum(axis=1))

    # Construct the symmetric matrix S = Q D^-1
    # so that S D = Q D^-1 D = Q.
    S = np.dot(Q, np.diag(pseudo_reciprocal(D)))

    # Return the Q = dot(S, D) decomposition.
    return S, D


def random_for_sylvester(n, off_states):
    # Random matrices for sylvester-like decomposition.

    # Construct random positive symmetric rates.
    # Set diagonal entries so that rows sum to zero.
    S = np.square(np.random.randn(n, n))
    S = S + S.T
    S = S - np.diag(S.sum(axis=1))

    # Define the reference process stationary distribution.
    D = np.square(np.random.randn(n))
    D /= D.sum()

    # Construct the time-reversible rate matrix,
    # defining diagonal entries such that rows sum to zero.
    # Construct the symmetric matrix S = Q D^-1
    # so that S D = Q D^-1 D = Q.
    pre_Q = np.dot(S, np.diag(D))
    Q = pre_Q - np.diag(pre_Q.sum(axis=1))
    S = np.dot(Q, np.diag(pseudo_reciprocal(D)))
    S_default = S
    D_default = D

    # Construct random positive symmetric rates.
    # Set diagonal entries so that rows sum to zero.
    # Zero the off rows and columns of rates.
    S = np.square(np.random.randn(n, n))
    S = S + S.T
    S = S - np.diag(S.sum(axis=1))
    S[off_states, :] = 0
    S[:, off_states] = 0

    # Define the reference process stationary distribution.
    D = np.square(np.random.randn(n))
    D[off_states] = 0
    D /= D.sum()

    # Construct the time-reversible rate matrix,
    # defining diagonal entries such that rows sum to zero.
    # Construct the symmetric matrix S = Q D^-1
    # so that S D = Q D^-1 D = Q.
    pre_Q = np.dot(S, np.diag(D))
    Q = pre_Q - np.diag(pre_Q.sum(axis=1))
    S = np.dot(Q, np.diag(pseudo_reciprocal(D)))
    S_reference = S
    D_reference = D

    # Define the transition rates from reference to default.
    # Some of these are zero.
    L = np.square(np.random.randn(n))
    L[off_states] = 0

    return S_reference, S_default, D_reference, D_default, L


def test_spectral_round_trip():
    np.random.seed(1234)
    n = 4

    # Construct a random reversible rate matrix.
    S, D = random_reversible_rate_matrix(n)
    Q = np.dot(S, np.diag(D))

    # Check basic properties of the rate matrix and its
    # symmetric * diagonal decomposition.
    # Also check the detailed balance equations.
    assert_allclose(Q.sum(axis=1), 0, atol=1e-15)
    assert_allclose(D.sum(), 1)
    assert_allclose(S, S.T)
    assert_allclose(np.dot(np.diag(D), Q), np.dot(Q.T, np.diag(D)))

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    spectral_decomposition = decompose_spectral(S, D)
    Q_spectral_reconstruction = reconstruct_spectral(*spectral_decomposition)
    assert_allclose(Q, Q_spectral_reconstruction)


def test_sylvester_round_trip():
    np.set_printoptions(linewidth=200)
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]
    #n = 3
    #off_states = []

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    sylvester_decomposition = decompose_sylvester(S0, S1, D0, D1, L)
    Q_reconstruction = reconstruct_sylvester(*sylvester_decomposition)
    atol = 1e-14
    #atol = 0
    assert_allclose(Q, Q_reconstruction, atol=atol)


if __name__ == '__main__':
    run_module_suite()

