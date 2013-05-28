"""
Test Markov chain functions that do not require random sampling.

The tests themselves may use random sampling.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler._mc import (
        get_history_log_likelihood, get_node_to_distn_naive, get_node_to_distn,
        construct_node_to_restricted_pmap,
        )


def _get_random_branching_tree(branching_distn, maxnodes=None):
    """
    Eventually try to merge this into networkx.

    Start with a single node.
    Each node has a random number of descendents
    drawn from a discrete distribution.
    An extra descendent is added to the root node.

    Parameters
    ----------
    branching_distn : array
        This defines the distribution of the number of child nodes per node.
        It is a finite distribution over the first few non-negative integers.
    maxnodes : integer
        Cap the number of nodes in the tree.

    Returns
    -------
    T : undirected acyclic networkx graph
        This is a rooted tree with at least one edge.
        The root of the tree is node 0.

    """
    # Check the input.
    if (maxnodes is not None) and (maxnodes < 2):
        raise ValueError('if maxnodes is not None then it should be >= 2')

    # Initialize.
    T = nx.Graph()
    root = 0
    next_node = 0
    active_nodes = {0}

    # Keep adding nodes until the cap is reached or all lineages have ended.
    while active_nodes:
        node = active_nodes.pop()
        nbranches = np.random.choice(
                range(len(branching_distn)), p=branching_distn)
        if node == root:
            nbranches += 1
        for i in range(nbranches):
            c = next_node
            next_node += 1
            T.add_edge(node, c)
            active_nodes.add(c)
            if (maxnodes is not None) and (len(T) == maxnodes):
                return T
    return T


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


def _get_random_test_setup():
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
    T = _get_random_branching_tree(branching_distn, maxnodes=6)
    root = 0

    # For each edge on the tree,
    # sample a random sparse state transition matrix.
    nstates = 4
    for na, nb in nx.bfs_edges(T, root):
        T[na][nb]['P'] = _get_random_transition_matrix(nstates)

    # Sample a root distribution.
    # It should be a little bit sparse, for testing.
    weights = np.random.exponential(size=4)
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

    # Return the random info for testing.
    return T, root, root_distn, node_to_allowed_states


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
        actual = get_history_log_likelihood(
                T, node_to_state, root, root_distn, P_default=P)
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
            node_to_distn_naive = get_node_to_distn_naive(
                    T, node_to_allowed_states, root, root_distn, P)

            # Get the node distributions more cleverly,
            # through the restricted pmap.
            node_to_pmap = construct_node_to_restricted_pmap(
                    T, root, node_to_allowed_states)
            node_to_distn_fast = get_node_to_distn(
                    T, node_to_pmap, root, P)

            # Convert distributions to ndarrays for approximate comparison.
            root_distn_vector = np.array(
                    [v for k, v in sorted(root_distn.items())], dtype=float)
            for node, distn in node_to_distn_naive.items():
                distn_vector = np.array(
                        [v for k, v in sorted(distn.items())], dtype=float)
                assert_allclose(distn_vector, root_distn_vector)
            for node, distn in node_to_distn_fast.items():
                distn_vector = np.array(
                        [v for k, v in sorted(distn.items())], dtype=float)
                assert_allclose(distn_vector, root_distn_vector)

    def test_node_to_distn(self):
        # Test the marginal distributions of node states.

        # This test uses a complicated tree with complicated transitions.
        # It checks the naive way of computing node_to_distn against
        # the more clever fast way to compute node_to_distn.
        # The two methods should give the same answer.
        nsamples = 10
        for i in range(nsamples):
            info = _get_random_test_setup()
            T, root, root_distn, node_to_allowed_states = info
            assert_equal(len(T), len(node_to_allowed_states))
            assert_(all(len(v) > 1 for v in node_to_allowed_states.values()))

            # Get the node distributions naively.
            node_to_distn_naive = get_node_to_distn_naive(
                    T, node_to_allowed_states, root, root_distn)

            # Get the node distributions more cleverly,
            # through the restricted pmap.
            node_to_pmap = construct_node_to_restricted_pmap(
                    T, root, node_to_allowed_states)
            node_to_distn_fast = get_node_to_distn(
                    T, node_to_pmap, root, root_distn)

            # Convert distributions to ndarrays for approximate comparison.
            for node in T:
                distn_naive = node_to_distn_naive[node]
                distn_fast = node_to_distn_fast[node]
                distn_naive_vector = np.array(
                        [v for k, v in sorted(distn_naive.items())])
                distn_fast_vector = np.array(
                        [v for k, v in sorted(distn_fast.items())])
                assert_allclose(distn_naive_vector, distn_fast_vector)

    def test_joint_endpoint_distributions(self):
        # Test the marginal distributions of node states.
        pass


if __name__ == '__main__':
    run_module_suite()

