"""
Test functions that Markov chain trajectories on a tree.

In this module we consider only discrete-time discrete-space Markov chains.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

from raoteh.sampler import _util, _mc0, _mcx, _sample_mcx


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
        for root in T:
            node_to_pset = _mcx.get_node_to_pset(T, root,
                    node_to_state=node_to_state, P_default=P)
            node_to_state_set = _mc0.get_node_to_set(T, root,
                    node_to_pset, P_default=P)
            assert_raises(
                    _util.StructuralZeroProb,
                    _sample_mcx.resample_states,
                    T, root,
                    node_to_state, P_default=P)

        # But if the path endpoints have states
        # that allow the intermediate state to act as a bridge,
        # then sampling is possible.
        root_distn = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        node_to_state = {0: 0, 2: 2}
        for root in T:
            observed = _sample_mcx.resample_states(
                    T, root,
                    node_to_state, root_distn=root_distn, P_default=P)
            expected = {0: 0, 1: 1, 2: 2}
            assert_equal(observed, expected)

        # Similarly if the root has a different distribution
        # and the endpoints are different but still bridgeable
        # by a single intermediate transitional state.
        root_distn = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        node_to_state = {0: 3, 2: 1}
        for root in T:
            observed = _sample_mcx.resample_states(
                    T, root,
                    node_to_state, root_distn=root_distn, P_default=P)
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
                    _util.StructuralZeroProb,
                    _sample_mcx.resample_states,
                    T, root,
                    node_to_state, root_distn=root_distn, P_default=P)

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
            observed = _sample_mcx.resample_states(T, root,
                    node_to_state, root_distn=root_distn, P_default=P)
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
        event_nodes = {10, 20, 21, 30, 31, 32}
        non_event_nodes = set(T) - event_nodes

        # Check that the correct internal states are sampled,
        # regardless of the root choice.
        for root in non_event_nodes:

            # Sample the edges states.
            T_aug = _sample_mcx.resample_edge_states(
                    T, root, event_nodes,
                    node_to_state=node_to_state, root_distn=root_distn,
                    P_default=P)
            
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

    def test_resample_edge_states_unknown_degree_three(self):
        # This test uses a more complicated transition matrix.

        # This transition matrix is on a 4x4 grid.
        P = _mc0.get_example_transition_matrix()

        # Define a very sparse tree.
        T = nx.Graph()
        T.add_weighted_edges_from([

            # first branch
            (0, 10, 1.0),
            (10, 11, 1.0),
            (11, 12, 1.0),

            # second branch
            (0, 20, 2.0),
            (20, 21, 2.0),
            (21, 22, 2.0),

            # third branch
            (0, 30, 1.0),
            (30, 31, 1.0),
            (31, 32, 1.0),
            ])

        # Define the known states
        node_to_state = {
                12 : 11,
                22 : 24,
                32 : 42}

        # Define the event nodes.
        # These are the degree-two nodes for which the adjacent edges
        # are allowed to differ from each other.
        event_nodes = {10, 11, 20, 21, 30, 31}
        non_event_nodes = set(T) - event_nodes

        # Sample the edges states.
        # Try all non-event nodes as roots.
        for root in non_event_nodes:

            # Sample the edges states.
            T_aug = _sample_mcx.resample_edge_states(
                    T, root, event_nodes,
                    node_to_state=node_to_state, root_distn=None,
                    P_default=P)
            
            # The unweighted and weighted tree size should be unchanged.
            assert_equal(T.size(), T_aug.size())
            assert_allclose(
                    T.size(weight='weight'),
                    T_aug.size(weight='weight'))

            # The origin node must have state 22.
            assert_equal(T_aug[0][10]['state'], 22)
            assert_equal(T_aug[0][20]['state'], 22)
            assert_equal(T_aug[0][30]['state'], 22)


class TestFeasibleStateSampler(TestCase):

    def test_get_feasible_history(self):

        # This transition matrix is on a 4x4 grid.
        P = _mc0.get_example_transition_matrix()

        # Define a very sparse tree.
        T = nx.Graph()
        T.add_weighted_edges_from([
            (0, 12, 1.0),
            (0, 23, 2.0),
            (0, 33, 1.0),
            ])

        # Define the known states
        node_to_state = {
                12 : 11,
                23 : 14,
                33 : 41}

        # Define a root at a node with a known state,
        # so that we can avoid specifying a distribution at the root.
        root = 12
        T_aug = _sample_mcx.get_feasible_history(T, node_to_state,
                root=root, P_default=P)

        # The unweighted and weighted tree size should be unchanged.
        assert_allclose(
                T.size(weight='weight'), T_aug.size(weight='weight'))

        # Check that for each node in the initial tree,
        # all adjacent edges in the augmented tree have the same state.
        # Furthermore if the state of the node in the initial tree is known,
        # check that the adjacent edges share this known state.
        for a in T:
            states = set()
            for b in T_aug.neighbors(a):
                states.add(T_aug[a][b]['state'])
            assert_equal(len(states), 1)
            state = _util.get_first_element(states)
            if a in node_to_state:
                assert_equal(node_to_state[a], state)

        # Check that every adjacent edge pair is a valid transition.
        successors = nx.dfs_successors(T_aug, root)
        for a, b in nx.bfs_edges(T_aug, root):
            if b in successors:
                for c in successors[b]:
                    ab = T_aug[a][b]['state']
                    bc = T_aug[b][c]['state']
                    assert_(ab in P)
                    assert_(bc in P[ab])
                    assert_(P[ab][bc]['weight'] > 0)


if __name__ == '__main__':
    run_module_suite()

