"""Test Rao-Teh sampler.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_)

from raoteh.sampler import _sampler


class TestSampler(TestCase):
    
    def test_sum(self):
        assert_equal(_sampler.mysum(2, 3), 5)

    def test_get_edge_bisected_graph(self):

        # Create an example from the networkx documentation.
        G = nx.Graph()
        G.add_weighted_edges_from([
            (1, 2, 0.125),
            (1, 3, 0.75),
            (2, 4, 1.2),
            (3, 4, 0.375)])

        # Create a new graph by bisecting the edges of the old graph.
        H = _sampler.get_edge_bisected_graph(G)

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
            chunk_tree, non_event_map, event_map = _sampler.get_chunk_tree(
                    T, event_nodes)
            
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


if __name__ == '__main__':
    run_module_suite()

