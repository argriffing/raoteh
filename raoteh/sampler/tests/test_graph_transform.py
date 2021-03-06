"""Test graph algorithms relevant to Rao-Teh sampling.
"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises)

from raoteh.sampler._graph_transform import (
        get_edge_bisected_graph,
        get_node_to_state,
        remove_redundant_nodes,
        get_redundant_degree_two_nodes,
        get_chunk_tree,
        add_trajectories,
        )


# This is an official itertools recipe.
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s)+1))


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
        H = get_edge_bisected_graph(G)

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
            results = get_chunk_tree(T, event_nodes)
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
        T_out = remove_redundant_nodes(T, redundant_nodes)
        assert_equal(set(T_out), set(T) - redundant_nodes)
        assert_equal(T_out[0][2]['weight'], 2)

        # Fail at removing a non-redundant node.
        redundant_nodes = {2}
        assert_raises(
                Exception,
                remove_redundant_nodes,
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

        # Check the set of redundant nodes.
        all_redundant_nodes = {1, 3, 4, 5, 6}
        obs_nodes = get_redundant_degree_two_nodes(T)
        assert_equal(all_redundant_nodes, obs_nodes)

        # Try removing all valid combinations of redundant nodes.
        for redundant_node_tuple in powerset(all_redundant_nodes):
            redundant_nodes = set(redundant_node_tuple)
            T_out = remove_redundant_nodes(T, redundant_nodes)
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
                    remove_redundant_nodes,
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
            T_out = remove_redundant_nodes(T, redundant_nodes)
            assert_equal(set(T_out), set(T) - redundant_nodes)
            assert_allclose(T_out.size(weight='weight'), original_size)


class TestAddTrajectories(TestCase):

    def test_compatible_trees(self):
        T_base = nx.Graph()
        T_base.add_edge(0, 1, weight=0.1)
        T_base.add_edge(0, 2, weight=0.1)
        T_base.add_edge(0, 3, weight=0.1)
        T_traj = nx.Graph()
        T_traj.add_edge(0, 1, state=0, weight=0.1)
        T_traj.add_edge(0, 20, state=0, weight=0.05)
        T_traj.add_edge(20, 2, state=0, weight=0.05)
        T_traj.add_edge(0, 3, state=0, weight=0.1)
        root = 0
        T_merged, dummy_nodes = add_trajectories(T_base, root, [T_traj])

        # There should not be any dummy nodes.
        assert_equal(dummy_nodes, set())

        # The merged tree should have four edges.
        assert_equal(T_base.size(), 3)
        assert_equal(T_merged.size(), 4)

        # The total weight of the merged tree
        # should be the same as the total weight of the base tree.
        assert_allclose(
                T_merged.size(weight='weight'),
                T_base.size(weight='weight'))


    def test_incompatible_trees(self):
        T_base = nx.Graph()
        T_base.add_edge(0, 1, weight=0.1)
        T_base.add_edge(0, 2, weight=0.1)
        T_base.add_edge(0, 3, weight=0.1)
        root = 0

        # Define a trajectory that is bad because it adds a high degree node.
        traj = nx.Graph()
        traj.add_edge(0, 4, state=0, weight=0.1)
        traj.add_edge(4, 20, state=0, weight=0.05)
        traj.add_edge(20, 2, state=0, weight=0.05)
        traj.add_edge(4, 3, state=0, weight=0.1)
        assert_raises(ValueError, add_trajectories,
                T_base, root, [traj])

        # Define a trajectory that is bad because it adds a leaf node.
        traj = nx.Graph()
        traj.add_edge(0, 1, state=0, weight=0.1)
        traj.add_edge(0, 20, state=0, weight=0.05)
        traj.add_edge(20, 2, state=0, weight=0.05)
        traj.add_edge(0, 3, state=0, weight=0.05)
        traj.add_edge(3, 4, state=0, weight=0.05)
        assert_raises(ValueError, add_trajectories,
                T_base, root, [traj])

        # Define a trajectory that is bad
        # because it flips around the nodes in a way that is incompatible
        # with the original tree topology.
        traj = nx.Graph()
        traj.add_edge(1, 0, state=0, weight=0.1)
        traj.add_edge(1, 2, state=0, weight=0.1)
        traj.add_edge(1, 3, state=0, weight=0.1)
        assert_raises(ValueError, add_trajectories,
                T_base, root, [traj])

    def test_complicated_incompatible_trees(self):
        T_base = nx.Graph()
        T_base.add_edge(0, 1, weight=0.1)
        T_base.add_edge(0, 2, weight=0.1)
        T_base.add_edge(0, 3, weight=0.1)
        T_base.add_edge(3, 4, weight=0.1)
        T_base.add_edge(3, 5, weight=0.1)
        root = 0

        # Define a trajectory that is bad
        # because the topology is different in a way that cannot be detected
        # by checking the degrees of the nodes.
        traj = nx.Graph()
        traj.add_edge(3, 1, state=0, weight=0.1)
        traj.add_edge(3, 2, state=0, weight=0.1)
        traj.add_edge(3, 0, state=0, weight=0.1)
        traj.add_edge(0, 4, state=0, weight=0.1)
        traj.add_edge(0, 5, state=0, weight=0.1)
        assert_raises(ValueError, add_trajectories,
                T_base, root, [traj])

    def test_edge_to_event_times(self):
        # The merged tree will look like the following,
        # where 'x' is a node in the original tree,
        # and 'a' is a node introduced by trajectory merging,
        # and 'o' is an event node.
        #
        #      x
        #     /|\
        #    / | \
        #   |  |  |
        #   o  o  x
        #   |  |  |
        #   x  |  | (0, 0)
        #      x  |
        #         x
        #        /| (0, 0)
        #       / a
        #      /  | (0, 10)
        #     |   a
        #     x   | (5, 10)
        #         a
        #         | (5, 0)
        #         o
        #         | (5, 0)
        #         a
        #         | (0, 0)
        #         x
        #
        T = nx.Graph()
        T.add_edge(0, 1, weight=0.1)
        T.add_edge(0, 2, weight=0.1)
        T.add_edge(0, 3, weight=0.1)
        T.add_edge(3, 4, weight=0.1)
        T.add_edge(3, 5, weight=0.1)
        T.add_edge(4, 6, weight=0.1)
        root = 0

        # Define a trajectory with an extra segment along one edge.
        traj_a = nx.Graph()
        traj_a.add_edge(0, 1, weight=0.1, state=0)
        traj_a.add_edge(0, 2, weight=0.1, state=0)
        traj_a.add_edge(0, 3, weight=0.1, state=0)
        traj_a.add_edge(3, 4, weight=0.1, state=0)
        traj_a.add_edge(3, 5, weight=0.1, state=0)
        traj_a.add_edge(4, 10, weight=0.025, state=0)
        traj_a.add_edge(10, 11, weight=0.05, state=5)
        traj_a.add_edge(11, 6, weight=0.025, state=0)

        # Define a trajectory with an interleaving segment.
        traj_b = nx.Graph()
        traj_b.add_edge(0, 1, weight=0.1, state=0)
        traj_b.add_edge(0, 2, weight=0.1, state=0)
        traj_b.add_edge(0, 3, weight=0.1, state=0)
        traj_b.add_edge(3, 4, weight=0.1, state=0)
        traj_b.add_edge(3, 5, weight=0.1, state=0)
        traj_b.add_edge(4, 20, weight=0.02, state=0)
        traj_b.add_edge(20, 21, weight=0.02, state=10)
        traj_b.add_edge(21, 6, weight=0.06, state=0)

        # Define a few event times along directed edges,
        # where the edge direction radiates away from the root.
        edge_to_event_times = {
                (0, 1) : {0.06},
                (0, 2) : {0.02},
                (4, 6) : {0.045},
                }

        # Construct the merged tree.
        T_merged, event_nodes = add_trajectories(
                T, root,
                [traj_a, traj_b],
                edge_to_event_times=edge_to_event_times)

        # After this point are some tests.

        # Check the total number of nodes in the merged tree.
        assert_equal(len(T_merged.edges()), 13)

        # Check the multiset of edge state pairs in the merged tree.
        state_pair_to_count = defaultdict(int)
        for edge in nx.bfs_edges(T_merged, root):
            na, nb = edge
            states = T_merged[na][nb]['states']
            state_pair = tuple(states)
            assert_equal(len(state_pair), 2)
            state_pair_to_count[state_pair] += 1
        assert_equal(state_pair_to_count[(0, 10)], 1)
        assert_equal(state_pair_to_count[(5, 10)], 1)
        assert_equal(state_pair_to_count[(5, 0)], 2)
        expected_state_pairs = set([(0, 0), (0, 10), (5, 10), (5, 0)])
        assert_equal(set(state_pair_to_count), expected_state_pairs)

        # Check that the number of event nodes is correct.
        assert_equal(len(edge_to_event_times), len(event_nodes))

        # The merged tree must contain all of the nodes of the original tree.
        missing_nodes = set(T) - set(T_merged)
        assert_equal(missing_nodes, set())

        # The base tree, the two trajectories, and the merged tree
        # should all have the same weighted size.
        weighted_size = T.size(weight='weight')
        assert_allclose(traj_a.size(weight='weight'), weighted_size)
        assert_allclose(traj_b.size(weight='weight'), weighted_size)
        assert_allclose(T_merged.size(weight='weight'), weighted_size)

        # Each event node must be adjacent to exactly two edges
        # in the merged tree, and both of these edges
        # must be annotated with the same sequence of state values.
        for node in event_nodes:
            assert_equal(T_merged.degree(node), 2)
            na, nb = T_merged[node]
            na_states = T_merged[node][na]['states']
            nb_states = T_merged[node][nb]['states']
            assert_equal(na_states, nb_states)

        # Print the edges of the merged tree.
        """
        print()
        print('--- add_trajectories test output ---')
        print(event_nodes)
        for edge in nx.bfs_edges(T_merged, root):
            na, nb = edge
            weight = T_merged[na][nb]['weight']
            states = T_merged[na][nb]['states']
            print(na, nb, weight, states)
        print()
        """

        """
        0 8 0.02 [0, 0]
        0 3 0.1 [0, 0]
        0 7 0.06 [0, 0]
        8 2 0.08 [0, 0]
        3 4 0.1 [0, 0]
        3 5 0.1 [0, 0]
        7 1 0.04 [0, 0]
        4 9 0.02 [0, 0]
        9 10 0.005 [0, 10]
        10 11 0.015 [5, 10]
        11 12 0.005 [5, 0]
        12 13 0.03 [5, 0]
        13 6 0.025 [0, 0]
        """



class TestGetNodeToState(TestCase):

    def test_get_node_to_state_simple_tree_identical_states(self):
        T = nx.Graph()
        T.add_edge(0, 1, state=42)
        T.add_edge(1, 2, state=42)
        all_query_nodes = {0, 1, 2}
        for query_nodes in powerset(all_query_nodes):
            nnodes = len(query_nodes)
            node_to_state = get_node_to_state(T, query_nodes)
            assert_equal(set(node_to_state), set(query_nodes))
            assert_equal(set(node_to_state.values()), set([42]*nnodes))

    def test_get_node_to_state_simple_tree_different_states(self):
        T = nx.Graph()
        T.add_edge(0, 1, state=42)
        T.add_edge(1, 2, state=42)
        T.add_edge(2, 3, state=99)

        # Some of the nodes have defined states.
        query_nodes = {0, 1, 3}
        node_to_state = get_node_to_state(T, query_nodes)
        assert_equal(node_to_state, {0:42, 1:42, 3:99})

        # But node 2 does not have a defined state
        # because it represents a state transition.
        query_nodes = {0, 1, 2, 3}
        assert_raises(ValueError, get_node_to_state, T, query_nodes)

    def test_complicated_tree(self):
        T = nx.Graph()
        T.add_edge(0, 1, state=2)
        T.add_edge(0, 2, state=2)
        T.add_edge(0, 3, state=2)
        T.add_edge(3, 4, state=10)
        T.add_edge(4, 5, state=10)
        T.add_edge(4, 6, state=10)

        # Most of the nodes have defined states.
        query_nodes = {0, 1, 2, 4, 5, 6}
        expected_node_to_state = {0:2, 1:2, 2:2, 4:10, 5:10, 6:10}
        node_to_state = get_node_to_state(T, query_nodes)
        assert_equal(node_to_state, expected_node_to_state)

        # One of the nodes is a transition without a defined state.
        query_nodes = {0, 1, 2, 3, 4, 5, 6}
        assert_raises(ValueError, get_node_to_state, T, query_nodes)


if __name__ == '__main__':
    run_module_suite()

