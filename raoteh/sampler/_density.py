"""
Utility functions relating sparse and dense matrix representations.

The sparse representation of trees in this module is usually
as a directed networkx graph relating arbitrary nodes,
together with a pre-order sequence of the nodes.

The sparse representation of transition rate matrices
and transition probability matrices are both as
directed weighted networkx graphs.
In the case of rate matrices, the networkx graph does not have loops.

The dense representation of transition rate matrices
and transition probability matrices are both as
numpy ndarrays with float dtype.

"""

import networkx as nx
import numpy as np


__all__ = [
        'check_square_dense',
        'digraph_to_bool_csr',
        'get_esd_transitions',
        'dict_to_numpy_array',
        'rate_matrix_to_numpy_array',
        ]


def rate_matrix_to_numpy_array(Q_sparse, **kwargs):
    """
    Analogous to networkx.to_numpy_matrix().

    This function also sets the diagonal entries of the output matrix
    so that the rows sum to zero.

    Parameters
    ----------
    Q_sparse : directed weighted networkx graph
        A sparse representation of a rate matrix.
    kwargs : dict
        These keyword args are passed through to nx.to_numpy_matrix().

    Returns
    -------
    Q_dense : 2d ndarray
        A dense representation of a rate matrix.

    """
    pre_Q_dense = nx.to_numpy_matrix(Q_sparse, **kwargs).A
    Q_dense = pre_Q_dense - np.diag(pre_Q_dense.sum(axis=1))
    return Q_dense


def dict_to_numpy_array(d, nodelist=None):
    """
    This function is like a 1d analog of networkx.to_numpy_array().

    Parameters
    ----------
    d : dict
        Maps nodes to floats.
    nodelist : sequence of nodes
        Sequence of nodes in the order they appear in the array.

    Returns
    -------
    out : 1d ndarray
        Dense array of nodes.

    """
    if nodelist is None:
        nodelist = tuple(d)
    return np.array([d.get(n, 0) for n in nodelist], dtype=float)


def check_square_dense(M):
    """

    Parameters
    ----------
    M : square 2d ndarray
        Matrix as a numpy ndarray.

    """
    if M is None:
        raise ValueError('the matrix is None')
    try:
        shape = M.shape
    except AttributeError as e:
        try:
            nnodes = M.number_of_nodes()
            raise ValueError('expected an ndarray but found a graph object')
        except AttributeError as e:
            raise ValueError('expected an ndarray')
    if len(shape) != 2:
        raise ValueError('expected len(M.shape) == 2')
    if M.shape[0] != M.shape[1]:
        raise ValueError('expected the array to be square')


def digraph_to_bool_csr(G, ordered_nodes):
    """
    This is a helper function for converting between networkx and cython.

    The output consists of two out of the three arrays of the csr interface.
    The third csr array (data) is not needed
    because we only care about the boolean sparsity structure.

    Parameters
    ----------
    G : networkx directed graph
        The unweighted graph to convert into csr form.
    ordered_nodes : sequence of nodes
        Nodes listed in a requested order.

    Returns
    -------
    csr_indices : ndarray of indices
        Part of the csr interface.
    csr_indptr : ndarray of pointers
        Part of the csr interface.

    """
    node_to_index = dict((n, i) for i, n in enumerate(ordered_nodes))
    csr_indices = []
    csr_indptr = [0]
    node_count = 0
    for na_index, na in enumerate(ordered_nodes):
        if na in G:
            for nb in G[na]:
                nb_index = node_to_index[nb]
                csr_indices.append(nb_index)
                node_count += 1
        csr_indptr.append(node_count)
    csr_indices = np.array(csr_indices, dtype=int)
    csr_indptr = np.array(csr_indptr, dtype=int)
    return csr_indices, csr_indptr


def get_esd_transitions(G, preorder_nodes, nstates, P_default=None):
    """
    Construct the edge-specific transition matrix as an ndim-3 numpy array.

    The 'esd' means 'edge-specific dense'.

    Parameters
    ----------
    G : directed networkx tree
        Edges of this tree are annotated with ndarray transition matrices.
    preorder_nodes : integer sequence
        An ordering of nodes in G.
    nstates : integer
        The number of states per transition matrix.
    P_default : ndarray
        This default transition matrix is used for edges that are not
        annotated with their own transition matrix.

    Returns
    -------
    esd_transitions : float ndarray with shape (nnodes, nstates, nstates).
        The ndarray representing the transition matrices on edges.

    """
    nnodes = len(preorder_nodes)
    if nnodes != G.number_of_nodes():
        raise ValueError('the number of nodes is inconsistent')
    node_to_index = dict((n, i) for i, n in enumerate(preorder_nodes))
    esd_transitions = np.zeros((nnodes, nstates, nstates), dtype=float)
    for na_index, na in enumerate(preorder_nodes):
        if na in G:
            for nb in G[na]:
                nb_index = node_to_index[nb]
                edge_object = G[na][nb]
                P = edge_object.get('P', P_default)
                check_square_dense(P)
                esd_transitions[nb_index] = P
    return esd_transitions

