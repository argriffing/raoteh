"""
Utility functions relating sparse and dense matrix representations.

The sparse representation of trees in this module is usually
as a directed networkx graph relating arbitrary nodes,
together with a pre-order sequence of the nodes.

"""


def _check_dense_P(P):
    """

    Parameters
    ----------
    P : (N, N) ndarray
        Transition matrix as a numpy array.

    """
    if P is None:
        raise ValueError('the transition matrix for this edge is None')
    try:
        if len(P.shape) != 2:
            raise ValueError('expected len(P.shape) == 2')
    except AttributeError as e:
        try:
            nnodes = P.number_of_nodes()
            raise ValueError('expected an ndarray but found a networkx graph')
        except AttributeError as e:
            raise ValueError('expected an ndarray')
    if P.shape[0] != P.shape[1]:
        raise ValueError('expected the array to be square')


def _digraph_to_bool_csr(G, ordered_nodes):
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


def _get_esd_transitions(G, preorder_nodes, nstates, P_default=None):
    """
    Construct the edge-specific transition matrix as an ndim-3 numpy array.

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
    node_to_index = dict((n, i) for i, n in enumerate(preorder_nodes))
    esd_transitions = np.zeros((nnodes, nstates, nstates), dtype=float)
    for na_index, na in enumerate(preorder_nodes):
        if na in G:
            for nb in G[na]:
                nb_index = node_to_index[nb]
                edge_object = G[na][nb]
                P = edge_object.get('P', P_default)
                if P is None:
                    raise ValueError('expected either a default '
                            'transition matrix '
                            'or a transition matrix on the edge '
                            'from node {0} to node {1}'.format(na, nb))
                esd_transitions[nb_index] = P
    return esd_transitions

