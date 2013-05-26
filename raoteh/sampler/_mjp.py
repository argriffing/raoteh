"""
Markov jump process likelihood calculations for testing the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip)


__all__ = []


def get_total_rates(Q):
    """
    Get the total rate away from each state.

    Parameters
    ----------
    Q : weighted directed networkx graph
        Instantaneous rate matrix.

    Returns
    -------
    total_rates : dict
        Sparse map from state to total rate out of the state.

    """
    total_rates = defaultdict(float)
    for a, b in Q.edges():
        total_rates[a] += Q[a][b]['weight']
    return dict(total_rates)


def get_conditional_transition_matrix(Q, total_rates=None):
    """
    Construct a transition matrix conditional on a state change.

    Parameters
    ----------
    Q : weighted directed networkx graph
        Instantaneous rate matrix.
    total_rates : dict, optional
        Sparse map from state to total rate out of the state.

    Returns
    -------
    P : weighted directed networkx graph
        Transition probability matrix
        conditional on an instantaneous transition.

    """
    if total_rates is None:
        total_rates = get_total_rates(Q)
    P = nx.DiGraph()
    for a, b in Q.edges():
        weight = Q[a][b]['weight'] / total_rates[a]
        P.add_edge(a, b, weight=weight)
    return P


def get_history_dwell_times(T):
    """

    Parameters
    ----------
    T : undirected weighted networkx tree with edges annotated with states
        A sampled history of states and substitution times on the tree.

    Returns
    -------
    dwell_times : dict
        Map from the state to the total dwell time on the tree.

    """
    dwell_times = defaultdict(float)
    for a, b in T.edges():
        edge = T[a][b]
        state = edge['state']
        weight = edge['weight']
        dwell_times[state] += weight
    return dict(dwell_times)


def get_history_root_state_and_transitions(T, root=None):
    """

    Parameters
    ----------
    T : undirected weighted networkx tree with edges annotated with states
        A sampled history of states and substitution times on the tree.
    root : integer, optional
        The root of the tree.
        If not specified, an arbitrary root will be used.

    Returns
    -------
    root_state : integer
        The state at the root.
    transition_counts : directed weighted networkx graph
        A networkx graph that tracks the number of times
        each transition type appears in the history.

    """

    # Bookkeeping.
    degrees = T.degree()

    # Pick a root with only one neighbor if no root was specified.
    if root is None:
        root = get_arbitrary_tip(T, degrees)

    # The root must have a well defined state.
    # This means that it cannot be adjacent to edges with differing states.
    root_states = [T[root][b]['state'] for b in T[root]]
    if len(set(root_states)) != 1:
        raise ValueError('the root does not have a well defined state')
    root_state = root_states[0]

    # Count the state transitions.
    transition_counts = nx.DiGraph()
    successors = nx.dfs_successors(T, root)
    for a, b in nx.bfs_edges(T, root):
        if degrees[b] == 2:
            c = get_first_element(successors[b])
            sa = T[a][b]['state']
            sb = T[b][c]['state']
            if sa != sb:
                if transition_counts.has_edge(sa, sb):
                    transition_counts[sa][sb]['weight'] += 1
                else:
                    transition_counts.add_edge(sa, sb, weight=1)

    # Return the statistics.
    return root_state, transition_counts


def get_history_statistics(T, root=None):
    """

    Parameters
    ----------
    T : undirected weighted networkx tree with edges annotated with states
        A sampled history of states and substitution times on the tree.
    root : integer, optional
        The root of the tree.
        If not specified, an arbitrary root will be used.

    Returns
    -------
    dwell_times : dict
        Map from the state to the total dwell time on the tree.
        This does not depend on the root.
    root_state : integer
        The state at the root.
    transition_counts : directed weighted networkx graph
        A networkx graph that tracks the number of times
        each transition type appears in the history.
        The counts depend on the root.

    Notes
    -----
    These statistics are sufficient to compute the Markov jump process
    likelihood for the sampled history.
        
    """
    dwell_times = get_history_dwell_times(T)
    root_state, transitions = get_history_root_state_and_transitions(
            T, root=root)
    return dwell_times, root_state, transitions


def construct_node_to_pmap(T, P, node_to_state, root):
    """
    For each node, construct the map from state to subtree likelihood.

    This variant is less general than construct_node_to_restricted pmap.
    It is mainly a helper function for the state resampler,
    and is possibly of not very general interest because of its lack
    of flexibility to change the transition matrix on each branch.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree without required edge annotation.
    P : networkx directed weighted graph
        Sparse transition matrix.
    node_to_state : dict
        A sparse map from a node to its known state.
    root : integer
        The root node.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    # Construct the augmented tree by annotating each edge with P.
    T_aug = nx.Graph()
    for a, b in T.edges():
        T_aug.add_edge(a, b, P=P)

    # Construct the map from node to allowed state set.
    node_to_allowed_states = {}
    all_states = set(P)
    for restricted_node, state in node_to_state.items():
        node_to_allowed_states[restricted_node] = {state}
    for unrestricted_node in set(T) - set(node_to_state):
        node_to_allowed_states[unrestricted_node] = all_states

    # Return the node to pmap dict.
    return construct_node_to_restricted_pmap(
            T_aug, root, node_to_allowed_states)


def construct_node_to_restricted_pmap(T, root, node_to_allowed_states):
    """
    For each node, construct the map from state to subtree likelihood.

    This function allows each node to be restricted to its own
    arbitrary set of allowed states.
    Applications include likelihood calculation,
    calculations of conditional expectations, and conditional state sampling.
    Some care is taken to distinguish between values that are zero
    because of structural reasons as opposed to values that are zero
    for numerical reasons.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices.
        The annotation uses the P attribute,
        and the transition matrices are themselves represented by
        networkx directed graphs with transition probabilities
        as the weight attribute of the edge.
    root : integer
        The root node.
    node_to_allowed_states : dict
        A map from a node to a set of allowed states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """

    # Bookkeeping.
    successors = nx.dfs_successors(T, root)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):
        valid_node_states = node_to_allowed_states[node]
        if node not in successors:
            node_to_pmap[node] = dict((s, 1.0) for s in valid_node_states)
        else:
            pmap = {}
            for node_state in valid_node_states:

                # Check for a structural subtree failure given this node state.
                structural_failure = False
                for n in successors[node]:

                    # Define the transition matrix according to the edge.
                    P = T[node][n]['P']

                    # Check that a transition away from the parent state
                    # is possible along this edge.
                    if node_state not in P:
                        structural_failure = True
                        break

                    # Get the list of possible child node states.
                    # These are limited by sparseness of the matrix of
                    # transitions from the parent state,
                    # and also by the possibility
                    # that the state of the child node is restricted.
                    valid_states = set(P[node_state]) & set(node_to_pmap[n])
                    if not valid_states:
                        structural_failure = True
                        break

                # If there is no structural failure,
                # then add the subtree probability to the node state pmap.
                if not structural_failure:
                    cprob = 1.0
                    for n in successors[node]:
                        P = T[node][n]['P']
                        valid_states = set(P[node_state]) & set(node_to_pmap[n])
                        nprob = 0.0
                        for s in valid_states:
                            a = P[node_state][s]['weight']
                            b = node_to_pmap[n][s]
                            nprob += a * b
                        cprob *= nprob
                    pmap[node_state] = cprob

            # Add the map from state to subtree likelihood.
            node_to_pmap[node] = pmap

    # Return the map from node to the map from state to subtree likelihood.
    return node_to_pmap


def get_restricted_likelihood(T, root, node_to_allowed_states, root_distn):
    """
    Compute a likelihood.

    This is a general likelihood calculator for piecewise
    homegeneous Markov jump processes on tree-structured domains.
    At each node in the tree, the set of possible states may be restricted.
    Lack of state restriction at a node corresponds to missing data;
    a common example of such missing data would be missing states
    at internal nodes in a tree.
    Alternatively, a node could have a completely specified state,
    as could be the case if the state of the process is completely
    known at the tips of the tree.
    More generally, a node could be restricted to an arbitrary set of states.
    The first three args are used to construct a map from each node
    to a map from the state to the subtree likelihood,
    and the last arg defines the initial conditions at the root.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_allowed_states : dict
        A map from a node to a set of allowed states.
    root_distn : dict
        A finite distribution over root states.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    node_to_pmap = construct_node_to_restricted_pmap(
            T, root, node_to_allowed_states)
    root_pmap = node_to_pmap[root]
    feasible_root_states = set(root_distn) & set(root_pmap)
    if not feasible_root_states:
        raise StructuralZeroProb('no root state is feasible')
    return sum(root_distn[s] * root_pmap[s] for s in feasible_root_states)


def get_marginal_state_distributions(T, root, node_to_pmap=None):
    """
    Get marginal state distributions at nodes in a tree.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.

    Returns
    -------
    node_to_distn : dict
        Sparse map from node to sparse map from state to probability.

    """
    # Construct the pmap if it does not yet already exist.

    # Treat the root separately.
    # If only one state is possible at the root, then we do not have to sample.
    # Otherwise consult the map from root states to probabilities.
    root_pmap = node_to_pmap[root]
    if not root_pmap:
        raise StructuralZeroProb('no state is feasible at the root')
    elif len(root_pmap) == 1:
        root_state = get_first_element(root_pmap)
        if not root_pmap[root_state]:
            raise NumericalZeroProb(
                    'the only feasible state at the root '
                    'gives a subtree probability of zero')
    else:
        if root_distn is None:
            raise ValueError(
                    'expected a prior distribution over the '
                    '%d possible states at the root' % len(root_pmap))
        prior_distn = root_distn
        states = list(set(prior_distn) & set(root_pmap))
        if not states:
            raise StructuralZeroProb(
                    'after accounting for the prior distribution at the root, '
                    'no root state is feasible')
        weights = []
        for s in states:
            weights.append(prior_distn[s] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise NumericalZeroProb('numerical problem at the root')
        if len(states) == 1:
            sampled_state = states[0]
        else:
            probs = np.array(weights, dtype=float) / weight_sum
            sampled_state = np.random.choice(states, p=probs)
        root_state = sampled_state
    node_to_sampled_state[root] = root_state

    # Sample the states at the rest of the nodes.
    for node in nx.dfs_preorder_nodes(T, root):

        # The root has already been sampled.
        if node == root:
            continue

        # Get the parent node and its state.
        parent_node = predecessors[node]
        parent_state = node_to_sampled_state[parent_node]
        
        # Check that the parent state has transitions.
        if parent_state not in P:
            raise StructuralZeroProb(
                    'no transition from the parent state is possible')

        # Sample the state of a non-root node.
        # A state is possible if it is reachable in one step from the
        # parent state which has already been sampled
        # and if it gives a subtree probability that is not structurally zero.
        states = list(set(P[parent_state]) & set(node_to_pmap[node]))
        if not states:
            raise StructuralZeroProb('found a non-root infeasibility')
        weights = []
        for s in states:
            weights.append(P[parent_state][s]['weight'] * node_to_pmap[node][s])
        weight_sum = sum(weights)
        if not weight_sum:
            raise NumericalZeroProb('numerical problem at a non-root node')
        if len(states) == 1:
            sampled_state = states[0]
        else:
            probs = np.array(weights, dtype=float) / weight_sum
            sampled_state = np.random.choice(states, p=probs)
        node_to_sampled_state[node] = sampled_state


def get_expected_history_statistics(T, Q, node_to_state, root):
    """
    This is a soft analog of get_history_statistics.

    The input is analogous to the Rao-Teh gen_histories input,
    and the output is analogous to the get_history_statistics output.
    This is not coincidental; the output of this function should
    be the same as the results of averaging the history statistics
    of Rao-Teh history samples, when the number of samples is large.
    A more general version of this function would be able
    to deal with states that have more flexible restrictions.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree.
    Q : directed weighted networkx graph
        A sparse rate matrix.
    node_to_state : dict
        A map from nodes to states.
        Nodes with unknown states do not correspond to keys in this map.
    root : integer
        Required to have a known state.

    Returns
    -------
    dwell_times : dict
        Map from the state to the expected dwell time on the tree.
        This does not depend on the root.
    transitions : directed weighted networkx graph
        A networkx graph that tracks the expected number of times
        each transition type appears in the history.
        The expectation depends on the root.

    """
    # Do some input validation for this restricted variant.
    if root not in node_to_state:
        raise ValueError('the root is required to have a known state')

    # Convert the sparse rate matrix to a dense ndarray rate matrix.
    states = sorted(T)
    nstates = len(states)
    Q_dense = np.zeros((nstates, nstates), dtype=float)
    for a, sa in enumerate(states):
        for b, sb in enumerate(states):
            if Q.has_edge(sa, sb):
                edge = Q[sa][sb]
                Q_dense[a, b] = edge['weight']
    Q_dense = Q_dense - np.diag(np.sum(Q_dense, axis=1))

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = nx.Graph()
    for na, nb in T.edges():
        edge = T[na][nb]
        weight = edge['weight']
        P_dense = scipy.linalg.expm(weight * Q_dense)
        P_nx = nx.DiGraph()
        for a, sa in enumerate(states):
            for b, sb in enumerate(states):
                P_nx.add_edge(sa, sb, weight=P_dense[a, b])
        T_aug.add_edge(a, b, P=P_nx)

    # Construct the map from node to allowed state set.
    node_to_allowed_states = {}
    full_state_set = set(states)
    for restricted_node, state in node_to_state.items():
        node_to_allowed_states[restricted_node] = {state}
    for unrestricted_node in full_state_set - set(node_to_state):
        node_to_allowed_states[unrestricted_node] = full_state_set

    # Construct the node to pmap dict.
    node_to_pmap = construct_node_to_restricted_pmap(
            T_aug, root, node_to_allowed_states)

    # Compute the expectations of the dwell times and the transition counts
    # by iterating over all edges and using the edge-specific
    # joint distribution of the states at the edge endpoints.
    for na, nb in nx.bfs_edges(T_aug, root):
        edge = T_aug[na][nb]
        # check 

