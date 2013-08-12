"""
Utility functions specific to the toy example named code2x3.

Some of these functions may eventually be pushed back
into the main raoteh python library.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
import scipy

from raoteh.sampler import _density, _mc0_dense, _mcy_dense, _mjp_dense

__all__ = ['get_expected_ntransitions']


def get_expected_ntransitions(
        T, node_to_allowed_states, root, nstates,
        root_distn=None, Q_default=None, E=None):
    """
    This function is roughly analogous to _mjp_dense.get_history_statistics().

    One difference is that this function records expectations per branch.
    Another difference is that this function uses expm_frechet to efficiently
    simultaneously compute the sum of expectations of all primary state
    state transition counts, rather than computing each expectation separately.
    Another difference is that the dwell times are not returned,
    and neither is the posterior distribution at the root.
    Note that this function does not depend on Rao-Teh sampling.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Edges of this tree are annotated with weights and possibly with
        edge-specific Q rate matrices.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    nstates : integer
        Number of states.
    root_distn : 1d ndarray, optional
        Distribution over states at the root.
    Q_default : 2d ndarray, optional
        A rate matrix.
    E : 2d ndarray, optional
        The coefficients for the weighted sum of transition count expectations.

    Returns
    -------
    edge_to_expectation : dict
        Maps each edge (a directed pair of nodes)
        to the expected number of transitions
        of the requested type (as specified by the coefficients in E)
        along the edge, conditional on the data.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # If E is None (default) then count expectations of all transitions,
    # as opposed to counting only specified transition types.
    if E is None:
        E = np.ones((nstates, nstates), dtype=float)
        np.fill_diagonal(E, 0)

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = _mjp_dense.get_expm_augmented_tree(T, root, Q_default=Q_default)

    # Construct the node to pmap dict.
    node_to_pmap = _mcy_dense.get_node_to_pmap(T_aug, root, nstates,
            node_to_allowed_states=node_to_allowed_states)

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    node_to_distn = _mc0_dense.get_node_to_distn(
            T_aug, root, node_to_pmap, nstates,
            root_distn=root_distn)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    T_joint = _mc0_dense.get_joint_endpoint_distn(
            T_aug, root, node_to_pmap, node_to_distn, nstates)

    # Compute the expectations of the dwell times and the transition counts
    # by iterating over all edges and using the edge-specific
    # joint distribution of the states at the edge endpoints.
    edge_to_expectation = {}
    for na, nb in nx.bfs_edges(T, root):

        # Get the rate matrix to use for this edge.
        Q = T[na][nb].get('Q', Q_default)
        _density.check_square_dense(Q)

        # Get the elapsed time along the edge.
        t = T[na][nb]['weight']

        # Get the conditional probability matrix associated with the edge.
        P = T_aug[na][nb]['P']

        # Get the joint probability matrix associated with the edge.
        J = T_joint[na][nb]['J']

        # Compute the Frechet derivative of the matrix exponential.
        C = E * Q
        interact = scipy.linalg.expm_frechet(
                t*Q, t*C, compute_expm=False)

        # Sum over all combinations of endpoint states.
        expectation = 0
        for sa in range(nstates):
            for sb in range(nstates):
                if not J[sa, sb]:
                    continue
                cond_prob = P[sa, sb]
                joint_prob = J[sa, sb]
                #rate = Q[sc, sd]
                x = interact[sa, sb]
                # XXX simplify the joint_prob / cond_prob
                #contrib = (joint_prob * rate * x) / cond_prob
                contrib = (joint_prob * x) / cond_prob
                expectation += contrib

        # Associate the expectation with the edge, in the output dictionary.
        edge_to_expectation[na, nb] = expectation

    # Return the expectation map.
    return edge_to_expectation

