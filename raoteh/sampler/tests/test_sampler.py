"""Test Rao-Teh sampler.
"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.stats
from scipy import special

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

from raoteh.sampler import _sampler

from raoteh.sampler._util import (
                StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        get_node_to_distn)

from raoteh.sampler._mjp import (
        get_expm_augmented_tree,
        get_reversible_differential_entropy,
        get_total_rates,
        get_history_statistics,
        get_expected_history_statistics)

from raoteh.sampler._conditional_expectation import (
        get_jukes_cantor_rate_matrix,
        get_jukes_cantor_probability,
        get_jukes_cantor_interaction)


def get_test_transition_matrix():
    # This returns a sparse transition matrix for testing.
    # It uses an hack for the node indices, because I want to keep integers.
    # This transition graph looks kind of like the following ascii art.
    #
    # 41 --- 42 --- 43 --- 44
    #  |      |      |      |
    # 31 --- 32 --- 33 --- 34
    #  |      |      |      |
    # 21 --- 22 --- 23 --- 24
    #  |      |      |      |
    # 11 --- 12 --- 13 --- 14
    #
    P = nx.DiGraph()
    weighted_edges = []
    for i in (1, 2, 3, 4):
        for j in (1, 2, 3, 4):
            source = i*10 + j
            sinks = []
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni = i + di
                    nj = j + dj
                    if not (di and dj):
                        if (1 <= ni <= 4) and (1 <= nj <= 4):
                            sink = ni*10 + nj
                            sinks.append(sink)
            nsinks = len(sinks)
            weight = 1 / float(nsinks)
            for sink in sinks:
                weighted_edges.append((source, sink, weight))
    P.add_weighted_edges_from(weighted_edges)
    return P


def get_neg_ll(T_aug, Q, root, distn):
    # For testing.
    total_rates = get_total_rates(Q)
    info = get_history_statistics(T_aug, root=root)
    dwell_times, root_state, transitions = info
    log_likelihood = np.log(distn[root_state])
    for state, dwell in dwell_times.items():
        log_likelihood -= dwell * total_rates[state]
    for sa, sb in transitions.edges():
        ntransitions = transitions[sa][sb]['weight']
        rate = Q[sa][sb]['weight']
        log_likelihood += special.xlogy(ntransitions, rate)
    neg_ll = -log_likelihood
    return neg_ll


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
                StructuralZeroProb,
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
                    StructuralZeroProb,
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
        event_nodes = {10, 20, 21, 30, 31, 32}
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

    def test_resample_edge_states_unknown_degree_three(self):
        # This test uses a more complicated transition matrix.

        # This transition matrix is on a 4x4 grid.
        P = get_test_transition_matrix()

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
        T_aug = _sampler.resample_edge_states(
                T, P, node_to_state, event_nodes)
        
        # The unweighted and weighted tree size should be unchanged.
        assert_equal(T.size(), T_aug.size())
        assert_allclose(
                T.size(weight='weight'), T_aug.size(weight='weight'))

        # The origin node must have state 22.
        assert_equal(T_aug[0][10]['state'], 22)
        assert_equal(T_aug[0][20]['state'], 22)
        assert_equal(T_aug[0][30]['state'], 22)


class TestFeasibleHistorySampler(TestCase):

    def test_get_feasible_history(self):

        # This transition matrix is on a 4x4 grid.
        P = get_test_transition_matrix()

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
        T_aug = _sampler.get_feasible_history(T, P, node_to_state, root=root)

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
            state = get_first_element(states)
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


class TestResamplePoisson(TestCase):

    def test_resample_poisson(self):

        # Define a tree.
        T = nx.Graph()
        T.add_edge(0, 12, weight=1.0, state=2)
        T.add_edge(0, 23, weight=2.0, state=2)
        T.add_edge(0, 33, weight=1.0, state=2)

        # Define some random poisson rates.
        poisson_rates = dict(zip(range(4), np.random.random(size=4)))
        
        # The resampled tree should have the same weighted size.
        T_aug = _sampler.resample_poisson(T, poisson_rates)
        assert_allclose(T.size(weight='weight'), T_aug.size(weight='weight'))


class TestForwardSample(TestCase):

    def test_forward_sample(self):

        # Define a tree with branch lengths.
        T = nx.Graph()
        T.add_edge(0, 12, weight=1.0)
        T.add_edge(0, 23, weight=2.0)
        T.add_edge(0, 33, weight=1.0)

        # Define a rate matrix.
        Q = nx.DiGraph()
        Q.add_edge(0, 1, weight=4)
        Q.add_edge(0, 2, weight=2)
        Q.add_edge(1, 0, weight=1)
        Q.add_edge(1, 2, weight=2)
        Q.add_edge(2, 1, weight=1)
        Q.add_edge(2, 0, weight=2)

        # Define a root and a root distribution.
        root = 0
        root_distn = {0 : 0.25, 1 : 0.5, 2 : 0.25}

        # Generate a few histories.
        nsamples = 10
        for i in range(nsamples):
            T_aug = _sampler.get_forward_sample(T, Q, root, root_distn)

            # Check that the weighted size is constant.
            assert_allclose(
                    T.size(weight='weight'), T_aug.size(weight='weight'))

            # Check that for each node in the initial tree,
            # all adjacent edges in the augmented tree have the same state.
            for a in T:
                states = set()
                for b in T_aug.neighbors(a):
                    states.add(T_aug[a][b]['state'])
                assert_equal(len(states), 1)


class TestMJP_Entropy(TestCase):

    @decorators.skipif(True, 'this test prints stuff')
    def test_mjp_reversible_differential_entropy(self):

        # Define a distribution over some primary states.
        nstates = 4
        distn = {
                0 : 0.1,
                1 : 0.2,
                2 : 0.3,
                3 : 0.4}
        """
        distn = {
                0 : 0.91,
                1 : 0.02,
                2 : 0.03,
                3 : 0.04}
        """

        # Define the transition rates.
        rates = [
                (0, 1, 2 * distn[1]),
                (1, 0, 2 * distn[0]),
                (1, 2, distn[2]),
                (2, 1, distn[1]),
                (2, 3, 2 * distn[3]),
                (3, 2, 2 * distn[2]),
                (3, 0, distn[0]),
                (0, 3, distn[3]),
                ]

        # Define the transition rate matrix.
        Q = nx.DiGraph()
        Q.add_weighted_edges_from(rates)

        # Summarize the transition rate matrix.
        total_rates = get_total_rates(Q)

        # Check reversibility.
        for sa, sb in Q.edges():
            rate_ab = Q[sa][sb]['weight']
            rate_ba = Q[sb][sa]['weight']
            assert_allclose(distn[sa] * rate_ab, distn[sb] * rate_ba)

        # Define a tree with branch lengths.
        T = nx.Graph()
        """
        T.add_edge(0, 1, weight=100.0)
        T.add_edge(1, 2, weight=100.0)
        T.add_edge(2, 3, weight=100.0)
        T.add_edge(3, 4, weight=100.0)
        T.add_edge(4, 5, weight=100.0)
        T.add_edge(5, 6, weight=100.0)
        T.add_edge(6, 7, weight=100.0)
        T.add_edge(7, 8, weight=100.0)
        T.add_edge(8, 9, weight=100.0)
        T.add_edge(9, 10, weight=100.0)
        """
        T.add_edge(0, 1, weight=1.0)
        T.add_edge(0, 2, weight=2.0)
        T.add_edge(0, 3, weight=1.0)
        total_tree_length = T.size(weight='weight')
        root = 0

        # Define the number of samples.
        nsamples = 10000
        sqrt_nsamples = np.sqrt(nsamples)

        # Do some forward samples,
        # and get the negative expected log likelihood.
        neg_ll_contribs_init = []
        neg_ll_contribs_dwell = []
        neg_ll_contribs_trans = []
        for T_aug in _sampler.gen_forward_samples(T, Q, root, distn, nsamples):
            info = get_history_statistics(T_aug, root=root)
            dwell_times, root_state, transitions = info

            # contribution of root state to log likelihood
            ll = np.log(distn[root_state])
            neg_ll_contribs_init.append(-ll)

            # contribution of dwell times
            ll = 0.0
            for state, dwell in dwell_times.items():
                ll -= dwell * total_rates[state]
            neg_ll_contribs_dwell.append(-ll)

            # contribution of transitions
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            neg_ll_contribs_trans.append(-ll)

        diff_ent_init = 0.0
        diff_ent_dwell = 0.0
        diff_ent_trans = 0.0
        t = total_tree_length
        for sa in set(Q) & set(distn):
            prob = distn[sa]
            diff_ent_init -= prob * np.log(prob)
            for sb in Q[sa]:
                rate = Q[sa][sb]['weight']
                diff_ent_dwell += t * prob * rate
                diff_ent_trans -= t * prob * rate * np.log(rate)

        print()
        print('nsamples:', nsamples)
        print()
        print('diff ent init:', diff_ent_init)
        print('neg ll init  :', np.mean(neg_ll_contribs_init))
        print('error        :', np.std(neg_ll_contribs_init) / sqrt_nsamples)
        print()
        print('diff ent dwell:', diff_ent_dwell)
        print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
        print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamples)
        print()
        print('diff ent trans:', diff_ent_trans)
        print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
        print('error        :', np.std(neg_ll_contribs_trans) / sqrt_nsamples)
        print()
        raise Exception('print entropy stuff')

    def test_mjp_monte_carlo_rao_teh_differential_entropy(self):

        # Define a distribution over some primary states.
        nstates = 4
        """
        distn = {
                0 : 0.1,
                1 : 0.2,
                2 : 0.3,
                3 : 0.4}
        """
        distn = {
                0 : 0.91,
                1 : 0.02,
                2 : 0.03,
                3 : 0.04}

        # Define the transition rates.
        rates = [
                (0, 1, 2 * distn[1]),
                (1, 0, 2 * distn[0]),
                (1, 2, distn[2]),
                (2, 1, distn[1]),
                (2, 3, 2 * distn[3]),
                (3, 2, 2 * distn[2]),
                (3, 0, distn[0]),
                (0, 3, distn[3]),
                ]

        # Define the transition rate matrix.
        Q = nx.DiGraph()
        Q.add_weighted_edges_from(rates)

        # Summarize the transition rate matrix.
        total_rates = get_total_rates(Q)

        # Check reversibility.
        for sa, sb in Q.edges():
            rate_ab = Q[sa][sb]['weight']
            rate_ba = Q[sb][sa]['weight']
            assert_allclose(distn[sa] * rate_ab, distn[sb] * rate_ba)

        # Define a tree with branch lengths.
        T = nx.Graph()
        T.add_edge(0, 1, weight=1.0)
        T.add_edge(0, 2, weight=2.0)
        T.add_edge(0, 3, weight=1.0)
        total_tree_length = T.size(weight='weight')
        root = 0

        # Restrict the states at the leaves.
        node_to_state = {1:0, 2:0, 3:1}

        # Define the number of samples.
        nsamples = 1000
        sqrt_nsamples = np.sqrt(nsamples)

        # Do some forward samples,
        # and get the negative expected log likelihood.
        sampled_root_distn = defaultdict(float)
        neg_ll_contribs_init = []
        neg_ll_contribs_dwell = []
        neg_ll_contribs_trans = []
        for T_aug in itertools.islice(
                _sampler.gen_histories(
                    T, Q, node_to_state, uniformization_factor=2,
                    root=root, root_distn=distn), nsamples):
            info = get_history_statistics(T_aug, root=root)
            dwell_times, root_state, transitions = info

            # contribution of root state to log likelihood
            sampled_root_distn[root_state] += 1.0 / nsamples
            ll = np.log(distn[root_state])
            neg_ll_contribs_init.append(-ll)

            # contribution of dwell times
            ll = 0.0
            for state, dwell in dwell_times.items():
                ll -= dwell * total_rates[state]
            neg_ll_contribs_dwell.append(-ll)

            # contribution of transitions
            ll = 0.0
            for sa, sb in transitions.edges():
                ntransitions = transitions[sa][sb]['weight']
                rate = Q[sa][sb]['weight']
                ll += special.xlogy(ntransitions, rate)
            neg_ll_contribs_trans.append(-ll)

        """
        diff_ent_init = 0.0
        diff_ent_dwell = 0.0
        diff_ent_trans = 0.0
        t = total_tree_length
        for sa in set(Q) & set(distn):
            prob = distn[sa]
            diff_ent_init -= prob * np.log(prob)
            for sb in Q[sa]:
                rate = Q[sa][sb]['weight']
                diff_ent_dwell += t * prob * rate
                diff_ent_trans -= t * prob * rate * np.log(rate)
        """

        # Get the diff_ent_init through the root posterior marginal distn.
        node_to_allowed_states = {}
        for node in T:
            if node in node_to_state:
                allowed = {node_to_state[node]}
            else:
                allowed = set(range(nstates))
            node_to_allowed_states[node] = allowed
        T_trans = get_expm_augmented_tree(T, Q, root)
        node_to_pmap = construct_node_to_restricted_pmap(
                T_trans, root, node_to_allowed_states)
        posterior_node_to_distn = get_node_to_distn(
                T_trans, node_to_allowed_states, node_to_pmap,
                root, prior_root_distn=distn)
        posterior_root_distn = posterior_node_to_distn[root]
        diff_ent_init = 0.0
        for state, prob in posterior_node_to_distn[root].items():
            diff_ent_init -= special.xlogy(prob, distn[state])

        # Get some posterior expectations.
        dwell_times, transitions = get_expected_history_statistics(
                T, Q, node_to_allowed_states, root, root_distn=distn)

        # Use some posterior expectations
        # to get the dwell time contribution to differential entropy.
        diff_ent_dwell = 0.0
        for s, rate in total_rates.items():
            diff_ent_dwell += dwell_times[s] * rate

        # Use some posterior expectations
        # to get the transition contribution to differential entropy.
        diff_ent_trans = 0.0
        for sa in set(Q) & set(transitions):
            for sb in set(Q[sa]) & set(transitions[sa]):
                rate = Q[sa][sb]['weight']
                ntrans_expected = transitions[sa][sb]['weight']
                diff_ent_trans -= ntrans_expected * np.log(rate)

        print()
        print('nsamples:', nsamples)
        print()
        print('sampled root distn :', sampled_root_distn)
        print('analytic root distn:', posterior_root_distn)
        print('diff ent init:', diff_ent_init)
        print('neg ll init  :', np.mean(neg_ll_contribs_init))
        print('error        :', np.std(neg_ll_contribs_init) / sqrt_nsamples)
        print()
        print('diff ent dwell:', diff_ent_dwell)
        print('neg ll dwell  :', np.mean(neg_ll_contribs_dwell))
        print('error         :', np.std(neg_ll_contribs_dwell) / sqrt_nsamples)
        print()
        print('diff ent trans:', diff_ent_trans)
        print('neg ll trans  :', np.mean(neg_ll_contribs_trans))
        print('error        :', np.std(neg_ll_contribs_trans) / sqrt_nsamples)
        print()
        raise Exception('print entropy stuff')


class TestRaoTehSampler(TestCase):

    def test_gen_histories(self):

        # Define a tree with branch lengths.
        T = nx.Graph()
        T.add_edge(0, 12, weight=1.0)
        T.add_edge(0, 23, weight=2.0)
        T.add_edge(0, 33, weight=1.0)

        # Define a rate matrix.
        Q = nx.DiGraph()
        Q.add_edge(0, 1, weight=4)
        Q.add_edge(0, 2, weight=2)
        Q.add_edge(1, 0, weight=1)
        Q.add_edge(1, 2, weight=2)
        Q.add_edge(2, 1, weight=1)
        Q.add_edge(2, 0, weight=2)

        # Define some leaf states.
        node_to_state = {12:1, 23:2, 33:2}

        # Generate a few histories.
        nhistories = 10
        for T_sample in itertools.islice(
                _sampler.gen_histories(T, Q, node_to_state), nhistories):

            # Check that the weighted size is constant.
            assert_allclose(
                    T.size(weight='weight'), T_sample.size(weight='weight'))

            # Check that for each node in the initial tree,
            # all adjacent edges in the augmented tree have the same state.
            # Furthermore if the state of the node in the initial tree is known,
            # check that the adjacent edges share this known state.
            for a in T:
                states = set()
                for b in T_sample.neighbors(a):
                    states.add(T_sample[a][b]['state'])
                assert_equal(len(states), 1)
                state = get_first_element(states)
                if a in node_to_state:
                    assert_equal(node_to_state[a], state)

    #@decorators.skipif(True, 'benchmark monte carlo conditional expectation')
    def test_jukes_cantor_conditional_expectation(self):
        # Compare Monte Carlo conditional expectations to the true values.

        # Define the initial state, final state, and elapsed time.
        a = 0
        b = 1
        t = 0.5
        n = 4

        # Define the tree, which in this case is only a path.
        T = nx.Graph()
        T.add_edge(0, 1, weight=t)
        node_to_state = {0:a, 1:b}
        node_to_allowed_states = {0:{a}, 1:{b}}
        root = 0

        # Define the Jukes-Cantor rate matrix.
        Q = get_jukes_cantor_rate_matrix(n)

        # Do some Monte Carlo sampling.
        observed_dwell_times = np.zeros(n, dtype=float)
        nhistories = 1000
        for T_sample in itertools.islice(
                _sampler.gen_histories(T, Q, node_to_state), nhistories):

            # Accumulate the amount of time spent in each state.
            for na, nb in T_sample.edges():
                edge = T_sample[na][nb]
                state = edge['state']
                weight = edge['weight']
                observed_dwell_times[state] += weight

        # Compute the expected dwell times.
        expected_dwell_times = np.zeros(n, dtype=float)
        for i in range(n):
            interaction = get_jukes_cantor_interaction(a, b, i, i, t, n)
            probability = get_jukes_cantor_probability(a, b, t, n)
            expected_dwell_times[i] = nhistories * interaction / probability

        # Get the MJP expected history statistics.
        info = get_expected_history_statistics(
                T, Q, node_to_allowed_states, root)
        mjp_expected_dwell_times, mjp_expected_transitions = info

        # Compare to the expected dwell times.
        print(observed_dwell_times)
        print(expected_dwell_times)
        print(mjp_expected_dwell_times)
        raise Exception

if __name__ == '__main__':
    run_module_suite()

