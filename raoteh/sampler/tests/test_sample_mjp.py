"""
Test functions that sample Markov jump trajectories on a tree.

These tests currently spam too much and are too slow to be useful,
so they are disabled.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import networkx as nx

from numpy.testing import (run_module_suite, TestCase,
        assert_equal, assert_allclose, assert_, assert_raises,
        decorators)

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)

from raoteh.sampler._sampler import (
        gen_histories,
        get_forward_sample,
        )


class TestShiwenIndel(TestCase):

    @decorators.slow
    def test_shiwen_indel(self):

        # Define the parameters and conditions of the toy model.
        # The state of the process is the sequence length.
        initial_state = 3
        final_state = 7
        elapsed_time = 2.0
        birth_rate = 1.0
        death_rate = 1.5
        max_state = 50
        nhistories = 5000
        initial_node = 0
        final_node = 1

        # Define the interval as a not very interesting tree.
        T = nx.Graph()
        T.add_edge(initial_node, final_node, weight=elapsed_time)

        # Define the rate matrix, throttled to max length.
        Q = nx.DiGraph()
        for state in range(1, max_state+1):
            surplus = state - 1
            if surplus:
                Q.add_edge(state, state-1, weight=surplus*death_rate)
            if state < max_state:
                Q.add_edge(state, state+1, weight=state*birth_rate)

        # Do some forward sampling.
        print('forward samples')
        naccepted = 0
        root_distn = {initial_state : 1}
        while naccepted < nhistories:
            traj = get_forward_sample(T, Q, initial_node, root_distn)
            neighbor = get_first_element(traj[final_node])
            final_edge = traj[final_node][neighbor]
            final_traj_state = final_edge['state']
            if final_traj_state == final_state:
                naccepted += 1
                nnodes = len(traj)
                ntransitions = nnodes - 2
                excess_ntransitions = ntransitions - (
                        final_state - initial_state)
                if excess_ntransitions % 2:
                    raise Exception('the number of excess transitions '
                            'should be even')
                num_of_event_pairs = excess_ntransitions // 2
                print(num_of_event_pairs)

        # Sample some trajectories.
        # Node 0 is the initial node, and node 1 is the final node.
        # Track the number of excess event pairs (num_of_event_pairs)
        # as in figure (5) of shiwen's report.
        print('rao-teh samples')
        num_of_event_pairs_to_count = defaultdict(int)
        high_water_to_count = defaultdict(int)
        node_to_state = {
                initial_node : initial_state,
                final_node : final_state}
        for traj in gen_histories(T, Q, node_to_state,
                root=initial_node, root_distn=None, nhistories=nhistories):
            nnodes = len(traj)
            ntransitions = nnodes - 2
            excess_ntransitions = ntransitions - (final_state - initial_state)
            if excess_ntransitions % 2:
                raise Exception('the number of excess transitions '
                        'should be even')
            num_of_event_pairs = excess_ntransitions // 2
            num_of_event_pairs_to_count[num_of_event_pairs] += 1
            high_water = max(traj[na][nb]['state'] for na, nb in traj.edges())
            high_water_to_count[high_water] += 1
            print(num_of_event_pairs)

        # Report some stuff.
        print()
        print('reproducing fig 5 histogram')
        for n, count in sorted(num_of_event_pairs_to_count.items()):
            print(n, count)
        print()
        print('high water mark state histogram')
        for n, count in sorted(high_water_to_count.items()):
            print(n, count)
        print()


if __name__ == '__main__':
    run_module_suite()

