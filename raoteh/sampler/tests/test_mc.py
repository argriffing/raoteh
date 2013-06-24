"""
Test Markov chain functions that do not require random sampling.

The tests themselves may use random sampling.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler import _mc0

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        )

from raoteh.sampler._sample_tree import get_random_branching_tree


def _get_random_transition_matrix(nstates):
    """
    Sample a random sparse state transition matrix.

    Each row of the transition matrix will be missing an entry.

    Parameters
    ----------
    nstates : integer
        The number of states in the transition matrix.

    Returns
    -------
    P : directed weighted networkx graph
        The sparse transition matrix.

    """
    P = nx.DiGraph()
    for i in range(nstates):
        jmissing = np.random.randint(nstates)
        weights = np.random.exponential(size=nstates)
        weights[jmissing] = 0
        total_weight = np.sum(weights)
        for j, weight in enumerate(weights):
            if j != jmissing:
                p = weight / total_weight
                P.add_edge(i, j, weight=p)
    return P


def _get_random_test_setup(nstates):
    """

    Returns
    -------
    T : undirected networkx graph
        Edges are annotated with transition matrix P.
    root : integer
        Root node.
    root_distn : dict
        Probability distribution at the root.
    node_to_allowed_states : dict
        Map from node to set of allowed states.

    """
    # Sample a random tree.
    branching_distn = [0.7, 0.1, 0.1, 0.1]
    T = get_random_branching_tree(branching_distn, maxnodes=6)
    root = 0

    # For each edge on the tree,
    # sample a random sparse state transition matrix.
    for na, nb in nx.bfs_edges(T, root):
        T[na][nb]['P'] = _get_random_transition_matrix(nstates)

    # Sample a root distribution.
    # It should be a little bit sparse, for testing.
    weights = np.random.exponential(size=nstates)
    imissing = np.random.randint(nstates)
    pairs = [(i, w) for i, w in enumerate(weights) if i != imissing]
    weights[imissing] = 0
    total_weight = np.sum(weights)
    root_distn = dict((i, w / total_weight) for i, w in pairs)

    # Sample allowed states at each node.
    # Disallow a random state at each node.
    states = range(nstates)
    node_to_allowed_states = dict((n, set(states)) for n in T)
    for n in T:
        imissing = np.random.randint(nstates)
        node_to_allowed_states[n].remove(imissing)

    # Final check on transition matrices on edges of T.
    for na, nb in nx.bfs_edges(T, root):
        edge_object = T[na][nb]
        P = edge_object.get('P', None)
        if P is None:
            raise Exception('internal error')

    # Return the random info for testing.
    return T, root, root_distn, node_to_allowed_states


def _assert_distn_allclose(da, db):
    # This is a helper function for testing.
    assert_equal(set(da), set(db))
    da_vector = np.array(
            [v for k, v in sorted(da.items())], dtype=float)
    db_vector = np.array(
            [v for k, v in sorted(db.items())], dtype=float)
    assert_allclose(da_vector, db_vector)


def _assert_nx_matrix_allclose(U, V):
    # This is a helper function for testing.
    assert_equal(set(U), set(V))
    assert_equal(set(U.edges()), set(V.edges()))
    U_weights = []
    V_weights = []
    for a, b in U.edges():
        U_weights.append(U[a][b]['weight'])
        V_weights.append(V[a][b]['weight'])
    u = np.array(U_weights, dtype=float)
    v = np.array(V_weights, dtype=float)
    assert_allclose(u, v)


class TestMarkovChain(TestCase):

    def test_history_log_likelihood_P_default(self):
        T = nx.Graph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(0, 3)
        root = 0
        node_to_state = {0:0, 1:0, 2:0, 3:0}
        root_distn = {0 : 0.5, 1 : 0.5, 2 : 0, 3 : 0}
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            (0, 0, 0.5),
            (0, 1, 0.25),
            (0, 2, 0.25),
            (1, 1, 0.5),
            (1, 2, 0.25),
            (1, 0, 0.25),
            (2, 2, 0.5),
            (2, 0, 0.25),
            (2, 1, 0.25)])
        actual = _mc0.get_history_log_likelihood(T, root, node_to_state,
                root_distn=root_distn, P_default=P)
        desired = 4 * np.log(0.5)
        assert_equal(actual, desired)

    def test_node_to_distn(self):
        # Test the marginal distributions of node states.

        # Try an example where no state is initially known,
        # but for which the transition matrix will cause
        # the joint distribution of states at all nodes to be
        # all in the same state.
        # This will cause all states to have the same distribution
        # as the root distribution.
        nstates = 3
        states = range(nstates)
        P = nx.DiGraph()
        P.add_weighted_edges_from([(s, s, 1) for s in states])
        T = nx.Graph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(0, 3)
        root = 0
        node_to_allowed_states = dict((n, set(states)) for n in T)
        for root_distn in (
                {0 : 0.10, 1 : 0.40, 2 : 0.50},
                {0 : 0.25, 1 : 0.50, 2 : 0.25},
                ):

            # Get the node distributions naively.
            node_to_set = node_to_allowed_states
            node_to_distn_naive = _mc0.get_node_to_distn_naive(T, root,
                    node_to_set, root_distn=root_distn, P_default=P)

            # Get the node distributions more cleverly,
            # through the restricted pmap.
            node_to_pmap = construct_node_to_restricted_pmap(T, root,
                    node_to_allowed_states=node_to_allowed_states, P_default=P)
            node_to_distn_fast = _mc0.get_node_to_distn(T, root, node_to_pmap,
                    root_distn=root_distn, P_default=P)

            # Convert distributions to ndarrays for approximate comparison.
            for node, distn in node_to_distn_naive.items():
                _assert_distn_allclose(root_distn, distn)
            for node, distn in node_to_distn_fast.items():
                _assert_distn_allclose(root_distn, distn)

    def test_node_to_distn_naive_vs_fast_random(self):
        # Test the marginal distributions of node states.

        # This test uses a complicated tree with complicated transitions.
        # It checks the naive way of computing node_to_distn against
        # the more clever fast way to compute node_to_distn.
        # The two methods should give the same answer.
        np.random.seed(1234)
        nstates = 4
        nsamples = 10
        for i in range(nsamples):
            info = _get_random_test_setup(nstates)
            T, root, root_distn, node_to_allowed_states = info
            assert_equal(len(T), len(node_to_allowed_states))
            assert_(all(len(v) > 1 for v in node_to_allowed_states.values()))

            # Get the node distributions naively.
            node_to_set = node_to_allowed_states
            node_to_distn_naive = _mc0.get_node_to_distn_naive(T, root,
                    node_to_set, root_distn=root_distn)

            # Get the node distributions more cleverly,
            # through the restricted pmap.
            node_to_pmap = construct_node_to_restricted_pmap(T, root,
                    node_to_allowed_states=node_to_allowed_states)
            node_to_distn_fast = _mc0.get_node_to_distn(T, root, node_to_pmap,
                    root_distn=root_distn)

            # Compare distributions at the root.
            root_distn_naive = node_to_distn_naive[root]
            root_distn_fast = node_to_distn_fast[root]
            _assert_distn_allclose(root_distn_naive, root_distn_fast)

            # Compare distributions at all nodes.
            for node in T:
                distn_naive = node_to_distn_naive[node]
                distn_fast = node_to_distn_fast[node]
                _assert_distn_allclose(distn_naive, distn_fast)

    def test_node_to_distn_unrestricted(self):
        # Test the marginal distributions of node states.

        # This test uses a complicated tree with complicated transitions.
        # It checks the naive way of computing node_to_distn against
        # the more clever fast way to compute node_to_distn.
        # The two methods should give the same answer.
        nstates = 4
        nsamples = 10
        for i in range(nsamples):
            info = _get_random_test_setup(nstates)
            T, root, root_distn, node_to_allowed_states = info
            node_to_allowed_states = dict((n, set(range(nstates))) for n in T)
            assert_equal(len(T), len(node_to_allowed_states))
            assert_(all(len(v) > 1 for v in node_to_allowed_states.values()))

            # Get the node distributions naively.
            node_to_set = node_to_allowed_states
            node_to_distn_naive = _mc0.get_node_to_distn_naive(T, root,
                    node_to_set, root_distn=root_distn)

            # Get the node distributions more cleverly,
            # through the restricted pmap.
            node_to_pmap = construct_node_to_restricted_pmap(T, root,
                    node_to_allowed_states=node_to_allowed_states)
            node_to_distn_fast = _mc0.get_node_to_distn(T, root, node_to_pmap,
                    root_distn=root_distn)

            # Convert distributions to ndarrays for approximate comparison.
            for node in T:
                distn_naive = node_to_distn_naive[node]
                distn_fast = node_to_distn_fast[node]
                _assert_distn_allclose(distn_naive, distn_fast)

    def test_joint_endpoint_distn(self):
        # Test joint endpoint state distributions on edges.
        nstates = 4
        nsamples = 10
        for i in range(nsamples):
            info = _get_random_test_setup(nstates)
            T, root, root_distn, node_to_allowed_states = info
            assert_equal(len(T), len(node_to_allowed_states))
            assert_(all(len(v) > 1 for v in node_to_allowed_states.values()))
            T_aug_naive = _mc0.get_joint_endpoint_distn_naive(T, root,
                    node_to_allowed_states, root_distn=root_distn)
            node_to_pmap = construct_node_to_restricted_pmap(T, root,
                    node_to_allowed_states=node_to_allowed_states)
            node_to_distn = _mc0.get_node_to_distn(T, root, node_to_pmap,
                    root_distn=root_distn)
            T_aug_fast = _mc0.get_joint_endpoint_distn(
                    T, root, node_to_pmap, node_to_distn)
            for na, nb in nx.bfs_edges(T, root):
                assert_(T_aug_naive.has_edge(na, nb))
                assert_(T_aug_fast.has_edge(na, nb))
            for na, nb in nx.bfs_edges(T, root):
                J_naive = T_aug_naive[na][nb]['J']
                J_fast = T_aug_fast[na][nb]['J']
                _assert_nx_matrix_allclose(J_naive, J_fast)


if __name__ == '__main__':
    run_module_suite()

