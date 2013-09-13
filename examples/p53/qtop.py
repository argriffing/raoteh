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
# These functions are meant to be called through
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
    P = reconstruct_sylvester(D0, D1, L, U0, U1, exp_lam0, exp_lam1, XQ)
    D0_off = (D0 == 0)
    P[D0_off, D0_off] = 1
    return P

def getp_sylvester_v2(D0, A0, B0, A1, B1, L, lam0, lam1, XQ, t):
    """
    Uses a decomposition more efficient for reconstruction.

    """
    exp_lam0 = np.exp(t * lam0)
    exp_lam1 = np.exp(t * lam1)
    P = reconstruct_sylvester_v2(A0, B0, A1, B1, L, exp_lam0, exp_lam1, XQ)
    D0_off = (D0 == 0)
    P[D0_off, D0_off] = 1
    return P

def getp_spectral(D, U, lam, t):
    """
    Get a transition probability matrix using the spectral decomposition.

    The idea is that when we are given a branch length,
    we will be able to convert this to a transition probability matrix
    using the reconstruction from the decomposition,
    where the reconstruction is treated as a black box.

    """
    exp_lam = np.exp(t * lam)
    P = reconstruct_spectral(D, U, exp_lam)
    D_off = (D == 0)
    P[D_off, D_off] = 1
    return P

def getp_spectral_v2(D, A, lam, B, t):
    """
    Uses a decomposition more efficient for reconstruction.

    """
    exp_lam = np.exp(t * lam)
    P = reconstruct_spectral_v2(A, exp_lam, B)
    D_off = (D == 0)
    P[D_off, D_off] = 1
    return P


###############################################################################
# Miscellaneous helper functions.

def build_block_2x2_old(A):
    return np.vstack([np.hstack(A[0]), np.hstack(A[1])])

def build_block_2x2(A):
    (M11, M12), (M21, M22) = A
    n = M11.shape[0]
    M = np.empty((2*n, 2*n))
    M[:n, :n] = M11
    M[:n, n:] = M12
    M[n:, :n] = M21
    M[n:, n:] = M22
    return M

def pseudo_reciprocal(v):
    with np.errstate(divide='ignore'):
        v_recip = np.reciprocal(v)
    return np.where(v==0, v, v_recip)

def dot_diag_square(v, M):
    return np.multiply(v[:, np.newaxis], M)

def dot_square_diag_square(P, v, Q):
    return np.dot(dot_square_diag(P, v), Q)

def dot_square_diag(M, v):
    return np.multiply(M, v)

def dot_diag_square_diag(v, M, w):
    return dot_diag_square(v, dot_square_diag(M, w))

def dot_dsdsd(u, P, v, Q, w):
    return np.dot(dot_diag_square_diag(u, P, v), dot_square_diag(Q, w))


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
    B = dot_diag_square_diag(D_sqrt, S, D_sqrt)
    lam, U = scipy.linalg.eigh(B)
    return D, U, lam

def decompose_spectral_v2(S, D):
    """
    This decomposition is more efficient for fast reconstruction.

    """
    D, U, lam = decompose_spectral(S, D)
    D_sqrt = np.sqrt(D)
    A = dot_diag_square(pseudo_reciprocal(D_sqrt), U)
    B = dot_square_diag(U.T, D_sqrt)
    return A, lam, B

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
    H0 = dot_diag_square_diag(D0_sqrt, S0, D0_sqrt) - np.diag(L)
    lam0, U0 = scipy.linalg.eigh(H0)

    # compute the second symmetric eigendecomposition
    D1_sqrt = np.sqrt(D1)
    H1 = dot_diag_square_diag(D1_sqrt, S1, D1_sqrt)
    lam1, U1 = scipy.linalg.eigh(H1)

    # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
    A = dot_square_diag(S0, D0) - np.diag(L)
    B = -dot_square_diag(S1, D1)
    Q = np.diag(L)
    XQ = scipy.linalg.solve_sylvester(A, B, Q)

    # Return the decomposition.
    return D0, D1, L, U0, U1, lam0, lam1, XQ


def decompose_sylvester_v2(S0, S1, D0, D1, L):
    """
    This decomposition is more efficient for reconstruction.

    """
    D0, D1, L, U0, U1, lam0, lam1, XQ = decompose_sylvester(S0, S1, D0, D1, L)
    D0_sqrt = np.sqrt(D0)
    D1_sqrt = np.sqrt(D1)
    A0 = dot_diag_square(pseudo_reciprocal(D0_sqrt), U0)
    B0 = dot_square_diag(U0.T, D0_sqrt)
    A1 = dot_diag_square(pseudo_reciprocal(D1_sqrt), U1)
    B1 = dot_square_diag(U1.T, D1_sqrt)
    return A0, B0, A1, B1, L, lam0, lam1, XQ


###############################################################################
# Two-stage decomposition helper functions.

def partial_syl_decomp_v3(S1, D1):
    """
    Prepare for multiple full sylvester decompositions with const S1 and D1.

    """
    # Compute an eigendecomposition.
    D1_sqrt = np.sqrt(D1)
    H1 = dot_diag_square_diag(D1_sqrt, S1, D1_sqrt)
    lam1, U1 = scipy.linalg.eigh(H1)

    # Post-processing of the decomposition, for faster reconstruction.
    A1 = dot_diag_square(pseudo_reciprocal(D1_sqrt), U1)
    B1 = dot_square_diag(U1.T, D1_sqrt)

    # Compute a Schur decomposition,
    # preparing to solve a sylvester equation.
    B = -dot_square_diag(S1, D1)
    schur_S1, schur_V1 = scipy.linalg.schur(B.T, output='real')

    return A1, B1, lam1, schur_S1, schur_V1


def full_syl_decomp_v3(S0, D0, L, A1, B1, lam1, schur_S1, schur_V1):
    """
    Full decomposition.

    Reconstruct using reconstruct_sylvester_v2().

    """
    # compute the first symmetric eigendecomposition
    D0_sqrt = np.sqrt(D0)
    H0 = dot_diag_square_diag(D0_sqrt, S0, D0_sqrt) - np.diag(L)
    lam0, U0 = scipy.linalg.eigh(H0)

    # Post-process the eigendecomposition, for faster reconstruction.
    A0 = dot_diag_square(pseudo_reciprocal(D0_sqrt), U0)
    B0 = dot_square_diag(U0.T, D0_sqrt)

    # Compute a Schur decomposition,
    # preparing to solve a sylvester equation.
    A = dot_square_diag(S0, D0) - np.diag(L)
    schur_R0, schur_U0 = scipy.linalg.schur(A, output='real')

    # Solve the sylvester equation.
    # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
    schur_F = dot_square_diag_square(schur_U0.T, L, schur_V1)
    trsyl, = scipy.linalg.get_lapack_funcs(
            ('trsyl',), (schur_R0, schur_S1, schur_F))
    if trsyl is None:
        raise Exception('lapack fail')
    schur_Y, schur_scale, schur_info = trsyl(
            schur_R0, schur_S1, schur_F, tranb='C')
    schur_Y = schur_scale * schur_Y
    if schur_info < 0:
        raise Exception('lapack trsyl fail')
    XQ = np.dot(np.dot(schur_U0, schur_Y), schur_V1.T)

    # Return the same decomposition as the v2 sylvester decomposition function.
    return A0, B0, A1, B1, L, lam0, lam1, XQ


###############################################################################
# Reconstruction helper functions.

def reconstruct_spectral(D, U, lam):
    """
    Return the reconstructed matrix given a spectral form.

    """
    D_sqrt = np.sqrt(D)
    Q_reconstructed = dot_dsdsd(
            pseudo_reciprocal(D_sqrt),
            U,
            lam,
            U.T,
            D_sqrt,
            )
    return Q_reconstructed

def reconstruct_spectral_v2(A, lam, B):
    """
    This uses a decomposition that is more efficient for reconnstruction.
    
    """
    return dot_square_diag_square(A, lam, B)

def reconstruct_sylvester(D0, D1, L, U0, U1, lam0, lam1, XQ):
    """
    Return the reconstructed matrix given a spectral form.

    """
    D0_sqrt = np.sqrt(D0)
    D1_sqrt = np.sqrt(D1)
    R11 = dot_dsdsd(
            pseudo_reciprocal(D0_sqrt),
            U0,
            lam0,
            U0.T,
            D0_sqrt,
            )
    R22 = dot_dsdsd(
            pseudo_reciprocal(D1_sqrt),
            U1,
            lam1,
            U1.T,
            D1_sqrt,
            )
    Q_reconstructed = build_block_2x2([
        [R11, np.dot(R11, XQ) - np.dot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed

def reconstruct_sylvester_v2(A0, B0, A1, B1, L, lam0, lam1, XQ):
    """
    This uses a decomposition that is more efficient for reconnstruction.
    
    """
    R11 = dot_square_diag_square(A0, lam0, B0)
    R22 = dot_square_diag_square(A1, lam1, B1)
    Q_reconstructed = build_block_2x2([
        [R11, np.dot(R11, XQ) - np.dot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed


###############################################################################
# Tests and test helper functions.

def assert_SD_reversible_rate_matrix(S, D):
    # Reversibility is defined as (d_i Q_ij == d_j Q_ji)
    # which in matrix notation is (D Q == Q.T D).
    # Note that this will always be true if (Q == S D) where
    # S is a symmetric matrix and D is a diagonal matrix.
    Q = dot_square_diag(S, D)
    assert_allclose(Q.sum(axis=1), 0, atol=1e-15)
    assert_allclose(D.sum(), 1)
    assert_allclose(S, S.T)
    assert_allclose(dot_diag_square(D, Q), dot_square_diag(Q.T, D))


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
    assert_SD_reversible_rate_matrix(S, D)

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    spectral_decomposition = decompose_spectral(S, D)
    Q_spectral_reconstruction = reconstruct_spectral(*spectral_decomposition)
    assert_allclose(Q, Q_spectral_reconstruction)


def test_spectral_v2_round_trip():
    np.random.seed(1234)
    n = 4

    # Construct a random reversible rate matrix.
    S, D = random_reversible_rate_matrix(n)
    Q = np.dot(S, np.diag(D))

    # Check basic properties of the rate matrix and its
    # symmetric * diagonal decomposition.
    # Also check the detailed balance equations.
    assert_SD_reversible_rate_matrix(S, D)

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    decomp = decompose_spectral_v2(S, D)
    Q_reconstruction = reconstruct_spectral_v2(*decomp)
    assert_allclose(Q, Q_reconstruction)


def test_sylvester_round_trip():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    sylvester_decomposition = decompose_sylvester(S0, S1, D0, D1, L)
    Q_reconstruction = reconstruct_sylvester(*sylvester_decomposition)
    atol = 1e-13
    #atol = 1e-14
    #atol = 0
    assert_allclose(Q, Q_reconstruction, atol=atol)


def test_sylvester_v2_round_trip():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    sylvester_decomposition = decompose_sylvester_v2(S0, S1, D0, D1, L)
    Q_reconstruction = reconstruct_sylvester_v2(*sylvester_decomposition)
    atol = 1e-13
    #atol = 1e-14
    #atol = 0
    assert_allclose(Q, Q_reconstruction, atol=atol)

def test_two_stage_sylvester_round_trip():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # First stage decomposition
    A1, B1, lam1, schur_S1, schur_V1 = partial_syl_decomp_v3(S1, D1)

    # Second stage decomposition
    sylvester_decomposition = full_syl_decomp_v3(
            S0, D0, L, A1, B1, lam1, schur_S1, schur_V1)

    # Check that the reconstruction from the spectral decomposition
    # gives back the original rate matrix.
    Q_reconstruction = reconstruct_sylvester_v2(*sylvester_decomposition)
    atol = 1e-13
    #atol = 1e-14
    #atol = 0
    assert_allclose(Q, Q_reconstruction, atol=atol)

def test_spectral_expm():
    np.random.seed(1234)
    n = 4
    t = 0.23

    # Construct a random reversible rate matrix.
    S, D = random_reversible_rate_matrix(n)
    Q = np.dot(S, np.diag(D))

    # Check basic properties of the rate matrix and its
    # symmetric * diagonal decomposition.
    # Also check the detailed balance equations.
    assert_SD_reversible_rate_matrix(S, D)

    D, U, lam = decompose_spectral(S, D)

    # Compute the transition probability matrix in two ways.
    P = getp_rate_matrix(Q, t)
    P_spectral = getp_spectral(D, U, lam, t)

    # Check that the transition probability matrices are close.
    atol = 1e-14
    #atol = 0
    assert_allclose(P, P_spectral, atol=atol)

def test_spectral_v2_expm():
    np.random.seed(1234)
    n = 4
    t = 0.23

    # Construct a random reversible rate matrix.
    S, D = random_reversible_rate_matrix(n)
    Q = np.dot(S, np.diag(D))

    # Check basic properties of the rate matrix and its
    # symmetric * diagonal decomposition.
    # Also check the detailed balance equations.
    assert_SD_reversible_rate_matrix(S, D)

    A, lam, B = decompose_spectral_v2(S, D)

    # Compute the transition probability matrix in two ways.
    P = getp_rate_matrix(Q, t)
    P_spectral = getp_spectral_v2(D, A, lam, B, t)

    # Check that the transition probability matrices are close.
    atol = 1e-14
    #atol = 0
    assert_allclose(P, P_spectral, atol=atol)

def test_sylvester_expm():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]
    t = 0.23

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # Get the sylvester-like decomposition.
    decomp = decompose_sylvester(S0, S1, D0, D1, L)
    D0, D1, L, U0, U1, lam0, lam1, XQ = decomp

    # Compute the transition probability matrix in two ways.
    P = getp_rate_matrix(Q, t)
    P_sylvester = getp_sylvester(D0, D1, L, U0, U1, lam0, lam1, XQ, t)

    # Check that the transition probability matrices are close.
    atol = 1e-14
    #atol = 0
    assert_allclose(P, P_sylvester, atol=atol)

def test_sylvester_v2_expm():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]
    t = 0.23

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # Get the sylvester-like decomposition.
    decomp = decompose_sylvester_v2(S0, S1, D0, D1, L)
    A0, B0, A1, B1, L, lam0, lam1, XQ = decomp

    # Compute the transition probability matrix in two ways.
    P = getp_rate_matrix(Q, t)
    P_sylvester = getp_sylvester_v2(D0, A0, B0, A1, B1, L, lam0, lam1, XQ, t)

    # Check that the transition probability matrices are close.
    atol = 1e-14
    #atol = 0
    assert_allclose(P, P_sylvester, atol=atol)

def test_two_stage_sylvester_expm():
    np.random.seed(1234)
    n = 5
    off_states = [0, 2]
    t = 0.23

    # Construct random matrices.
    S0, S1, D0, D1, L = random_for_sylvester(n, off_states)
    assert_SD_reversible_rate_matrix(S0, D0)
    assert_SD_reversible_rate_matrix(S1, D1)

    # Build the compound switching-model rate matrix.
    Q00 = np.dot(S0, np.diag(D0)) - np.diag(L)
    Q01 = np.diag(L)
    Q10 = np.zeros((n, n))
    Q11 = np.dot(S1, np.diag(D1))
    Q = build_block_2x2([[Q00, Q01], [Q10, Q11]])

    # First stage decomposition
    A1, B1, lam1, schur_S1, schur_V1 = partial_syl_decomp_v3(S1, D1)

    # Second stage decomposition
    A0, B0, A1, B1, L, lam0, lam1, XQ = full_syl_decomp_v3(
            S0, D0, L, A1, B1, lam1, schur_S1, schur_V1)

    # Compute the transition probability matrix in two ways.
    P = getp_rate_matrix(Q, t)
    P_sylvester = getp_sylvester_v2(D0, A0, B0, A1, B1, L, lam0, lam1, XQ, t)

    # Check that the transition probability matrices are close.
    atol = 1e-14
    #atol = 0
    assert_allclose(P, P_sylvester, atol=atol)

if __name__ == '__main__':
    run_module_suite()

