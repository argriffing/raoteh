"""
Test Rao-Teh sampling on Markov-chain-like models on trees.

"""
from __future__ import division, print_function, absolute_import

import itertools
from collections import defaultdict

import numpy as np
import networkx as nx
from scipy import special

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

from raoteh.sampler import _mc0

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, sparse_expm,
        )

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        )

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

from raoteh.sampler._sampler import (
        gen_histories,
        gen_forward_samples,
        get_forward_sample,
        resample_poisson,
        get_feasible_history,
        )


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
        T_aug = resample_poisson(T, poisson_rates)
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
            T_aug = get_forward_sample(T, Q, root, root_distn)

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

        # Define a distribution over some states.
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
        for T_aug in gen_forward_samples(T, Q, root, distn, nsamples):
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
        print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamples)
        print()
        raise Exception('print entropy stuff')

    def test_mjp_monte_carlo_rao_teh_differential_entropy(self):

        # Define a distribution over some states.
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
        for T_aug in gen_histories(
                T, Q, node_to_state, root=root, root_distn=distn,
                uniformization_factor=2, nhistories=nsamples):

            # Get some stats of the histories.
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

        node_to_allowed_states = {}
        for node in T:
            if node in node_to_state:
                allowed = {node_to_state[node]}
            else:
                allowed = set(range(nstates))
            node_to_allowed_states[node] = allowed

        # Get some posterior expectations.
        expectation_info = get_expected_history_statistics(
                T, node_to_allowed_states,
                root, root_distn=distn, Q_default=Q)
        dwell_times, post_init_distn, transitions = expectation_info

        # Get the diff_ent_init through the root posterior marginal distn.
        diff_ent_init = 0.0
        for state, prob in post_init_distn.items():
            diff_ent_init -= special.xlogy(prob, distn[state])

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
        print('analytic root distn:', post_init_distn)
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
        print('error         :', np.std(neg_ll_contribs_trans) / sqrt_nsamples)
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
        for T_sample in gen_histories(
                T, Q, node_to_state, nhistories=nhistories):

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
        for T_sample in gen_histories(
                T, Q, node_to_state, nhistories=nhistories):

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
                T, node_to_allowed_states, root, Q_default=Q)
        mjp_dwell, mjp_init, mjp_trans = info

        # Compare to the expected dwell times.
        print(observed_dwell_times)
        print(expected_dwell_times)
        print(mjp_dwell)
        raise Exception


class TestFeasibleHistorySampler(TestCase):

    def test_sparse_expm_symmetric(self):
        Q = nx.DiGraph()
        Q.add_weighted_edges_from([
            (0, 1, 1),
            (1, 0, 1),
            (1, 2, 1),
            (2, 1, 1),
            (0, 2, 1),
            (2, 0, 1),
            ])
        P = sparse_expm(Q, 1)
        expected_edges = (
                (0, 0), (0, 1), (0, 2),
                (1, 0), (1, 1), (1, 2),
                (2, 0), (2, 1), (2, 2),
                )
        assert_equal(set(P.edges()), set(expected_edges))

    def test_sparse_expm_one_way_state_transitions(self):
        Q = nx.DiGraph()
        Q.add_weighted_edges_from([
            (0, 1, 1),
            (1, 2, 1)])
        P = sparse_expm(Q, 1)
        expected_edges = (
                (0, 1), (1, 2), (0, 2),
                (0, 0), (1, 1), (2, 2))
        assert_equal(set(P.edges()), set(expected_edges))
        T = nx.Graph()
        T.add_weighted_edges_from([
            (0, 1, 1.0),
            (0, 2, 1.0),
            (0, 3, 1.0)])
        root = 0
        root_distn = {1 : 0.5, 2 : 0.5}

        # Check infeasibility of a transition to state 0.
        node_to_state = {1 : 1, 2 : 1, 3 : 0}
        assert_raises(
                Exception,
                get_feasible_history,
                T, P, node_to_state,
                root=root,
                root_distn=root_distn,
                )

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
        T_aug = get_feasible_history(T, P, node_to_state, root=root)

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


if __name__ == '__main__':
    run_module_suite()

