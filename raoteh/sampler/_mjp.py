"""
Markov jump process likelihood calculations for testing the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx
import scipy.linalg
import scipy.stats
from scipy import special

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_dense_rate_matrix, sparse_expm,
        expm_frechet_is_simple, simple_expm_frechet,
        get_first_element, get_arbitrary_tip)

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        get_node_to_distn,
        get_joint_endpoint_distn,
        )


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


#TODO add tests
def get_trajectory_log_likelihood(
        T_aug, root, prior_root_distn, Q_default):
    """

    Parameters
    ----------
    T_aug : undirected weighted networkx graph
        Trajectory with weighted edges annotated with states.
    root : integer
        Root node.
    prior_root_distn : dict
        Prior distribution over states at the root.
    Q_default : directed weighted networkx graph
        Rate matrix which applies to all edges.

    Returns
    -------
    log_likelihood : float
        Logarithm of the trajectory likelihood
        according to the given Markov jump process.

    Notes
    -----
    Regarding the order of the arguments of this function, T_aug is first
    to facilitate functools.partial wrapping for MCMC callback.

    """
    # Compute the total rates.
    total_rates = get_total_rates(Q_default)

    # Compute primary process statistics.
    # These will be used for two purposes.
    # One of the purposes is as the denominator of the
    # importance sampling ratio.
    # The second purpose is to compute contributions
    # to the neg log likelihood estimate.
    info = get_history_statistics(T_aug, root=root)
    dwell_times, root_state, transitions = info

    # contribution of root state to log likelihood
    init_ll = np.log(prior_root_distn[root_state])

    # contribution of dwell times
    ll = 0.0
    for state, dwell in dwell_times.items():
        ll -= dwell * total_rates[state]
    dwell_ll = ll

    # contribution of transitions
    ll = 0.0
    for sa, sb in transitions.edges():
        ntransitions = transitions[sa][sb]['weight']
        rate = Q_default[sa][sb]['weight']
        ll += special.xlogy(ntransitions, rate)
    trans_ll = ll

    # Return the sum of the log likelihood contributions.
    log_likelihood = init_ll + dwell_ll + trans_ll
    return log_likelihood


def get_reversible_differential_entropy(Q, stationary_distn, t):
    """
    Compute differential entropy of a time-reversible Markov jump process.

    This is the differential entropy of the distribution over trajectories.
    The rate matrix Q must be time-reversible
    with the given stationary distribution.

    Parameters
    ----------
    Q : directed weighted networkx graph
        Sparse matrix of transition rates.
    stationary_distn : dict
        Stationary distribution of the process.
    t : float
        Amount of time over which the process is observed.

    Returns
    -------
    differential_entropy : float
        The differential entropy of the distribution over trajectories.
        This is not the Shannon entropy, and may be negative.

    """
    # Initialize the rate of trajectory distribution entropy per time.
    # This accounts for both expected dwell times and expected transitions.
    differential_entropy = 0.0
    for sa in set(Q) & set(stationary_distn):
        prob = stationary_distn[sa]
        differential_entropy -= prob * np.log(prob)
        for sb in Q[sa]:
            rate = Q[sa][sb]['weight']
            differential_entropy += t * prob * rate * (1 - np.log(rate))
    return differential_entropy


def get_expm_augmented_tree(T, root, Q_default=None):
    """
    Add transition probability matrices to edges.

    Construct the augmented tree by annotating each edge
    with the appropriate state transition probability matrix.

    Parameters
    ----------
    T : weighted undirected networkx graph
        This tree is possibly annotated with edge-specific
        rate matrices Q.
    root : integer
        Root node.
    Q_default : weighted directed networkx graph, optional
        Sparse rate matrix.

    Returns
    -------
    T_aug : weighted undirected networkx graph
        Tree annotated with transition probability matrices P.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        weight = edge['weight']
        Q = edge.get('Q', Q_default)
        if Q is None:
            raise ValueError('no rate matrix is available for this edge')
        P = sparse_expm(Q, weight)
        T_aug.add_edge(na, nb, weight=weight, P=P)
    return T_aug


def get_likelihood(T, node_to_allowed_states,
        root, root_distn=None, Q_default=None):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Edges of this tree are annotated with weights and possibly with
        edge-specific Q rate matrices.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    root_distn : dict, optional
        Sparse distribution over states at the root.
        This is optional if only one state is allowed at the root.
    Q_default : directed weighted networkx graph, optional
        A sparse rate matrix.

    Returns
    -------
    likelihood : float
        A marginal likelihood.

    Notes
    -----
    This function is meaningful even when the root_distn is not technically
    a distribution in the sense of summing to 1.
    If the root distribution is not specified,
    then it is treated as all ones; this is different than
    treating it as a uniform distribution on some set of states.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = get_expm_augmented_tree(T, root, Q_default=Q_default)

    # TODO code below here should probably be replaced
    # TODO with a call to an _mc function...

    # Construct the node to pmap dict.
    node_to_pmap = construct_node_to_restricted_pmap(
            T_aug, root, node_to_allowed_states)

    # Get subtree likelihoods conditional on the root state.
    root_pmap = node_to_pmap[root]

    # Get feasible initial states.
    feasible_states = set(root_pmap)
    if root_distn is not None:
        feasible_states.intersection_update(set(root_distn))

    # Return likelihood.
    if root_distn is None:
        likelihood = sum(root_pmap.values())
    else:
        likelihood = sum(root_distn[s] * root_pmap[s] for s in feasible_states)
    return likelihood


def get_expected_history_statistics(T, node_to_allowed_states,
        root, root_distn=None, Q_default=None):
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
        Edges of this tree are annotated with weights and possibly with
        edge-specific Q rate matrices.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    root_distn : dict, optional
        Sparse distribution over states at the root.
        This is optional if only one state is allowed at the root.
    Q_default : directed weighted networkx graph, optional
        A sparse rate matrix.

    Returns
    -------
    dwell_times : dict
        Map from the state to the expected dwell time on the tree.
        This does not depend on the root.
    posterior_root_distn : dict
        Posterior distribution of states at the root.
    transitions : directed weighted networkx graph
        A networkx graph that tracks the expected number of times
        each transition type appears in the history.
        The expectation depends on the root.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # Attempt to define the state space.
    # This will use the default rate matrix if available,
    # and it will try to use all available edge-specific rate matrices.
    full_state_set = set()
    if Q_default is not None:
        full_state_set.update(Q_default)
    for na, nb in nx.bfs_edges(T, root):
        Q = T[na][nb].get('Q', None)
        if Q is not None:
            full_state_set.update(Q)
    states = sorted(full_state_set)
    nstates = len(states)

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = get_expm_augmented_tree(T, root, Q_default=Q_default)

    # Construct the node to pmap dict.
    node_to_pmap = construct_node_to_restricted_pmap(
            T_aug, root, node_to_allowed_states)

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    node_to_distn = get_node_to_distn(
            T_aug, node_to_allowed_states, node_to_pmap, root, root_distn)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    T_joint = get_joint_endpoint_distn(
            T_aug, node_to_pmap, node_to_distn, root)

    # Compute the expectations of the dwell times and the transition counts
    # by iterating over all edges and using the edge-specific
    # joint distribution of the states at the edge endpoints.
    expected_dwell_times = defaultdict(float)
    expected_transitions = nx.DiGraph()
    for na, nb in nx.bfs_edges(T, root):

        # Get the sparse rate matrix to use for this edge.
        Q = T[na][nb].get('Q', Q_default)
        if Q is None:
            raise ValueError('no rate matrix is available for this edge')

        Q_is_simple = expm_frechet_is_simple(Q)
        if not Q_is_simple:
            # Construct the dense local rate matrix
            # for the purposes of frechet derivative.
            Q_dense = np.zeros((nstates, nstates), dtype=float)
            for sa_index, sa in enumerate(states):
                for sb_index, sb in enumerate(states):
                    if Q.has_edge(sa, sb):
                        Q_dense[sa_index, sb_index] = Q[sa][sb]['weight']
            Q_dense = Q_dense - np.diag(np.sum(Q_dense, axis=1))

        # Get the elapsed time along the edge.
        t = T[na][nb]['weight']

        # Get the conditional probability matrix associated with the edge.
        P = T_aug[na][nb]['P']

        # Get the joint probability matrix associated with the edge.
        J = T_joint[na][nb]['J']

        # Compute contributions to dwell time expectations along the path.
        for sc_index, sc in enumerate(states):
            if not Q_is_simple:
                C = np.zeros((nstates, nstates), dtype=float)
                C[sc_index, sc_index] = 1.0
                interact = scipy.linalg.expm_frechet(
                        t*Q_dense, t*C, compute_expm=False)
            for sa_index, sa in enumerate(states):
                for sb_index, sb in enumerate(states):
                    if not J.has_edge(sa, sb):
                        continue
                    cond_prob = P[sa][sb]['weight']
                    joint_prob = J[sa][sb]['weight']
                    if Q_is_simple:
                        x = simple_expm_frechet(
                                Q, sa_index, sb_index, sc_index, sc_index, t)
                    else:
                        x = interact[sa_index, sb_index]
                    # XXX simplify the joint_prob / cond_prob
                    expected_dwell_times[sc] += (joint_prob * x) / cond_prob

        # Compute contributions to transition count expectations.
        for sc_index, sc in enumerate(states):
            for sd_index, sd in enumerate(states):
                if not Q.has_edge(sc, sd):
                    continue
                if not Q_is_simple:
                    C = np.zeros((nstates, nstates), dtype=float)
                    C[sc_index, sd_index] = 1.0
                    interact = scipy.linalg.expm_frechet(
                            t*Q_dense, t*C, compute_expm=False)
                for sa_index, sa in enumerate(states):
                    for sb_index, sb in enumerate(states):
                        if not J.has_edge(sa, sb):
                            continue
                        cond_prob = P[sa][sb]['weight']
                        joint_prob = J[sa][sb]['weight']
                        if Q_is_simple:
                            rate = Q[sc][sd]['weight']
                            x = simple_expm_frechet(Q,
                                    sa_index, sb_index,
                                    sc_index, sd_index, t)
                        else:
                            rate = Q_dense[sc_index, sd_index]
                            x = interact[sa_index, sb_index]
                        # XXX simplify the joint_prob / cond_prob
                        contrib = (joint_prob * rate * x) / cond_prob
                        if not expected_transitions.has_edge(sc, sd):
                            expected_transitions.add_edge(sc, sd, weight=0.0)
                        expected_transitions[sc][sd]['weight'] += contrib

    # Return some expectations.
    dwell_expectation = dict(expected_dwell_times)
    init_expectation = node_to_distn[root]
    trans_expectation = expected_transitions
    return dwell_expectation, init_expectation, trans_expectation


#XXX under construction
def get_expected_history_statistics_for_log_likelihood():
    """

    Returns
    -------
    dwell_statistic : float
        The dot product of the expected dwell times in each state
        with the total rate out of the state.
        This does not depend on the root.
    transition_statistic : float
        The sum, over all transition types, of the product of the
        expected number of such transitions and the log of the transition rate.
        The expectation depends on the root.

    """
    pass
