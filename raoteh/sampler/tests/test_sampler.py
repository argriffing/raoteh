"""Test Rao-Teh sampler.
"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises)

from raoteh.sampler import _sampler, _graph_transform


# This is an official itertools recipe.
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s)+1))


class TestNodeStateSampler(TestCase):

    def test_resample_states_short_path(self):

        # Define a very sparse transition matrix as a path.
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            (0, 1, 1.0),
            (1, 0, 0.5),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 3, 0.5),
            (3, 2, 1.0)])

        # Define a very sparse tree as a path.
        T = nx.Graph()
        T.add_edges_from([
            (0, 1),
            (1, 2)])

        # Two of the three vertices of the tree have known states.
        # The intermediate state is unknown,
        # No value of the intermediate state can possibly connect
        # the states at the two endpoints of the path.
        node_to_state = {0: 0, 2: 3}
        assert_raises(
                _sampler.StructuralZeroProb,
                _sampler.resample_states,
                T, P, node_to_state)

        # But if the path endpoints have states
        # that allow the intermediate state to act as a bridge,
        # then sampling is possible.
        root_distn = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        node_to_state = {0: 0, 2: 2}
        for root in T:
            observed = _sampler.resample_states(
                    T, P, node_to_state, root, root_distn)
            expected = {0: 0, 1: 1, 2: 2}
            assert_equal(observed, expected)

        # Similarly if the root has a different distribution
        # and the endpoints are different but still bridgeable
        # by a single intermediate transitional state.
        root_distn = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        node_to_state = {0: 3, 2: 1}
        for root in T:
            observed = _sampler.resample_states(
                    T, P, node_to_state, root, root_distn)
            expected = {0: 3, 1: 2, 2: 1}
            assert_equal(observed, expected)

    def test_resample_states_infeasible(self):

        # Do not allow any transitions.
        P = nx.DiGraph()

        # Define a very sparse tree as a path.
        T = nx.Graph()
        T.add_edges_from([
            (0, 1),
            (1, 2)])

        # Sampling is not possible.
        # Check that the correct exception is raised.
        root_distn = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        node_to_state = {0: 0, 2: 2}
        for root in T:
            assert_raises(
                    _sampler.StructuralZeroProb,
                    _sampler.resample_states,
                    T, P, node_to_state, root, root_distn)

    def test_resample_states_separated_regions(self):
        # This test includes multiple regions of nodes with unknown states,
        # where the regions are separated from each other by nodes
        # with known states.

        # Define a very sparse transition matrix as a path.
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            (0, 1, 1.0),
            (1, 0, 0.5),
            (1, 2, 0.5),
            (2, 1, 0.5),
            (2, 3, 0.5),
            (3, 2, 1.0)])

        # Define a very sparse tree.
        T = nx.Graph()
        T.add_edges_from([
            (0, 10),
            (0, 20),
            (0, 30),
            (10, 11),
            (20, 21),
            (30, 31),
            (31, 32)])

        # Define the known states
        root_distn = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        node_to_state = {
                0 : 0,
                11 : 2,
                21 : 2,
                32 : 3}

        # Check that the correct internal states are sampled,
        # regardless of the root choice.
        for root in T:
            observed = _sampler.resample_states(
                    T, P, node_to_state, root, root_distn)
            expected = {
                    0 : 0,
                    10 : 1,
                    11 : 2,
                    20 : 1,
                    21 : 2,
                    30 : 1,
                    31 : 2,
                    32 : 3}
            assert_equal(observed, expected)


class TestEdgeStateSampler(TestCase):

    def test_resample_edge_states_separated_regions(self):
        # This test is analogous to the corresponding node state test.

        # Define a very sparse transition matrix as a path.
        # To avoid periodicity it allows self transitions.
        P = nx.DiGraph()
        P.add_weighted_edges_from([
            (0, 0, 0.5),
            (1, 1, 0.5),
            (2, 2, 0.5),
            (3, 3, 0.5),
            (0, 1, 0.5),
            (1, 0, 0.25),
            (1, 2, 0.25),
            (2, 1, 0.25),
            (2, 3, 0.25),
            (3, 2, 0.5)])

        # Define a very sparse tree.
        T = nx.Graph()
        T.add_weighted_edges_from([
            (0, 10, 1.0),
            (0, 20, 1.0),
            (0, 30, 1.0),
            (10, 11, 2.0),
            (11, 12, 2.0),
            (12, 13, 2.0),
            (13, 14, 2.0),
            (14, 15, 2.0),
            (15, 16, 2.0),
            (20, 21, 1.0),
            (21, 22, 1.0),
            (30, 31, 1.0),
            (31, 32, 1.0),
            (32, 33, 1.0)])

        # Define the known states
        root_distn = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        node_to_state = {
                0 : 0,
                16 : 0,
                22 : 2,
                33 : 3}

        # Define the event nodes.
        # These are the degree-two nodes for which the adjacent edges
        # are allowed to differ from each other.
        event_nodes = {20, 21, 30, 31, 32}
        non_event_nodes = set(T) - event_nodes

        # Check that the correct internal states are sampled,
        # regardless of the root choice.
        for root in non_event_nodes:

            # Sample the edges states.
            T_aug = _sampler.resample_edge_states(
                    T, P, node_to_state, event_nodes,
                    root=root, root_distn=root_distn)
            
            # The unweighted and weighted tree size should be unchanged.
            assert_equal(T.size(), T_aug.size())
            assert_allclose(
                    T.size(weight='weight'), T_aug.size(weight='weight'))

            # The edges along the long path should all have state 0
            # because this path does not include any event nodes.
            long_path_nodes = (10, 11, 12, 13, 14, 15, 16)
            long_path_edges = zip(long_path_nodes[:-1], long_path_nodes[1:])
            for a, b in long_path_edges:
                assert_equal(T_aug[a][b]['state'], 0)
            
            # Check the edge states along the short branch.
            assert_equal(T_aug[0][20]['state'], 0)
            assert_equal(T_aug[20][21]['state'], 1)
            assert_equal(T_aug[21][22]['state'], 2)

            # Check the edge states along the medium length branch.
            assert_equal(T_aug[0][30]['state'], 0)
            assert_equal(T_aug[30][31]['state'], 1)
            assert_equal(T_aug[31][32]['state'], 2)
            assert_equal(T_aug[32][33]['state'], 3)


class TestGraphTransform(TestCase):

    def test_get_edge_bisected_graph(self):

        # Create an example from the networkx documentation.
        G = nx.Graph()
        G.add_weighted_edges_from([
            (1, 2, 0.125),
            (1, 3, 0.75),
            (2, 4, 1.2),
            (3, 4, 0.375)])

        # Create a new graph by bisecting the edges of the old graph.
        H = _graph_transform.get_edge_bisected_graph(G)

        # The edge-bisected graph has twice as many edges.
        assert_equal(len(G.edges()) * 2, len(H.edges()))
        assert_equal(G.size()*2, H.size())

        # The sum of edge weights is unchanged.
        assert_allclose(G.size(weight='weight'), H.size(weight='weight'))

        # The node set of the edge-bisected graph includes that of the original.
        assert_(set(G) <= set(H))

        # The added nodes are each greater than each original node.
        assert_(max(G) < min(set(H) - set(G)))

    def test_get_chunk_tree(self):

        # Define the original tree and its event nodes.
        # This is taken from a doodle in my notebook,
        # and it is not particularly cleverly chosen.
        tree_edges = (
                (0, 1),
                (1, 2),
                (3, 4),
                (4, 2),
                (2, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (7, 10),
                (10, 11),
                (11, 12),
                (12, 13),
                (13, 14),
                (13, 15),
                (15, 16),
                (16, 17),
                )
        event_nodes = {1, 4, 5, 6, 8, 10, 11, 12, 15, 16}

        # Create a tree by specifying the edges.
        T = nx.Graph()
        T.add_edges_from(tree_edges)

        # Run tests, using all possible roots and also a default root.
        potential_roots = list(T) + [None]
        for root in potential_roots:

            # Construct the chunk tree and its associated node maps.
            results = _graph_transform.get_chunk_tree(T, event_nodes)
            chunk_tree, non_event_map, event_map = results
            
            # The nodes pointed to by the non_event_map
            # should be nodes in the chunk_tree.
            assert_(set(non_event_map.values()) <= set(T))

            # The output tree should have 11 nodes and 10 edges.
            assert_equal(len(chunk_tree), 11)
            assert_equal(len(chunk_tree.edges()), 10)

            # The 8 non-event nodes should map to 7 unique chunk nodes.
            assert_equal(len(non_event_map), 8)
            assert_equal(len(set(non_event_map.values())), 7)

            # The non-event nodes 13 and 14 should map to the same chunk.
            assert_equal(non_event_map[13], non_event_map[14])

    def test_remove_redundant_nodes_short_path(self):

        # Define a short path with one redundant
        # and one non-redundant internal node.
        T = nx.Graph()
        T.add_edge(0, 1, state=0, weight=1)
        T.add_edge(1, 2, state=0, weight=1)
        T.add_edge(2, 3, state=1, weight=1)

        # Try removing a redundant node.
        redundant_nodes = {1}
        T_out = _graph_transform.remove_redundant_nodes(T, redundant_nodes)
        assert_equal(set(T_out), set(T) - redundant_nodes)
        assert_equal(T_out[0][2]['weight'], 2)

        # Fail at removing a non-redundant node.
        redundant_nodes = {2}
        assert_raises(
                Exception,
                _graph_transform.remove_redundant_nodes,
                T, redundant_nodes)

    def test_remove_redundant_nodes_long_path(self):

        # Define a path with multiple consecutive redundant internal nodes.
        T = nx.Graph()
        T.add_edge(0, 1, state=0, weight=1.1)
        T.add_edge(1, 2, state=0, weight=1.2)
        T.add_edge(2, 3, state=1, weight=1.3)
        T.add_edge(3, 4, state=1, weight=1.4)
        T.add_edge(4, 5, state=1, weight=1.5)
        T.add_edge(5, 6, state=1, weight=1.6)
        T.add_edge(6, 7, state=1, weight=1.7)

        # Get the original weighted size.
        # This is the sum of weights of all edges.
        original_size = T.size(weight='weight')

        # Try removing all valid combinations of redundant nodes.
        for redundant_node_tuple in powerset((1, 3, 4, 5, 6)):
            redundant_nodes = set(redundant_node_tuple)
            T_out = _graph_transform.remove_redundant_nodes(T, redundant_nodes)
            assert_equal(set(T_out), set(T) - redundant_nodes)
            assert_allclose(T_out.size(weight='weight'), original_size)

    def test_remove_redundant_nodes_small_tree(self):

        # Define a short path with one redundant
        # and one non-redundant internal node.
        T = nx.Graph()
        T.add_edge(0, 1, state=0, weight=1)
        T.add_edge(0, 2, state=0, weight=1)
        T.add_edge(0, 3, state=0, weight=1)

        # None of the nodes are considered redundant in the current
        # implementation, because each node is of degree 1 or 3.
        for redundant_nodes in ({0}, {1}, {2}, {3}):
            assert_raises(
                    Exception,
                    _graph_transform.remove_redundant_nodes,
                    T, redundant_nodes)

    def test_remove_redundant_nodes_medium_tree(self):

        # Define a tree.
        T = nx.Graph()
        T.add_edge(0, 10, state=0, weight=1.1)
        T.add_edge(0, 20, state=0, weight=1.2)
        T.add_edge(0, 30, state=0, weight=1.3)
        T.add_edge(20, 21, state=0, weight=1.4)
        T.add_edge(30, 31, state=0, weight=1.5)
        T.add_edge(31, 32, state=0, weight=1.6)

        # Get the original weighted size.
        # This is the sum of weights of all edges.
        original_size = T.size(weight='weight')

        # Try removing all valid combinations of redundant nodes.
        for redundant_node_tuple in powerset((20, 30, 31)):
            redundant_nodes = set(redundant_node_tuple)
            T_out = _graph_transform.remove_redundant_nodes(T, redundant_nodes)
            assert_equal(set(T_out), set(T) - redundant_nodes)
            assert_allclose(T_out.size(weight='weight'), original_size)



if __name__ == '__main__':
    run_module_suite()

