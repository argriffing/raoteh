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


# Define a few helper functions that treat dictionaries and networkx graphs
# as sparse vectors and matrices with undefined shapes.
# Maybe this should be more object oriented.

def _mul_dd(da, db):
    # dict dict multiplication
    return dict((n, da[n] * db[n]) for n in set(da) & set(db))

def _sum_d(d):
    # dict sum
    return sum(d.values())

def _mul_ds(d, s):
    # dict scalar multiplication
    return dict((k, v * s) for k, v in d.items())

def _div_ds(d, s):
    # dict scalar division
    return dict((k, v / s) for k, v in d.items())



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


def get_zero_step_posterior_distn(prior_distn, pmap):
    """
    Do a kind of sparse dict-dict multiplication and normalize the result.

    Parameters
    ----------
    prior_distn : dict
        A sparse map from a state to a prior probability.
    pmap : dict
        A sparse map from a state to an observation likelihood.
        In the MJP application, this likelihood observation corresponds to 
        a subtree likelihood.

    Returns
    -------
    posterior_distn : dict
        A sparse map from a state to a posterior probability.

    """
    if not prior_distn:
        raise StructuralZeroProb(
                'no state is feasible according to the prior')
    if not pmap:
        raise StructuralZeroProb(
                'no state is feasible according to the observations')
    feasible_states = set(prior_distn) & set(pmap)
    if not feasible_states:
        raise StructuralZeroProb(
                'no state is in the intersection of prior feasible '
                'and observation feasible states')
    d = dict((n, prior_distn[n] * pmap[n]) for n in feasible_states)
    total_weight = sum(d.values())
    if not total_weight:
        raise NumericalZeroProb('numerical zero probability error')
    posterior_distn = dict((k, v / total_weight) for k, v in d.items())
    return posterior_distn


#XXX add tests
def get_node_to_distn(T, node_to_pmap, root, prior_root_distn=None):
    """
    Get marginal state distributions at nodes in a tree.

    This function is similar to the Rao-Teh state sampling function,
    except that instead of sampling a state at each node,
    this function computes marginal distributions over states at each node.
    Also, each edge of the input tree for this function has been
    annotated with its own transition probability matrix,
    whereas the Rao-Teh sampling function uses a single
    uniformized transition probability matrix for all edges.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    node_to_pmap : dict
        Map from node to a map from a state to the subtree likelihood.
        This map incorporates state restrictions.
    root : integer
        Root node.
    prior_root_distn : dict, optional
        A finite distribution over root states.

    Returns
    -------
    node_to_distn : dict
        Sparse map from node to sparse map from state to probability.

    """
    node_to_distn = {}
    for node in nx.dfs_preorder_nodes(T, root):

        # Compute the root prior distribution at the root separately.
        if node == root:
            prior_distn = prior_root_distn
        else:
            parent_node = predecessors[node]
            parent_distn = node_to_distn[parent_node]

            # This is essentially a sparse matrix vector multiplication.
            prior_distn = defaultdict(float)
            for sa, pa in parent_distn.items():
                for sb in P[sa]:
                    edge = P[sa][sb]
                    pab = edge['weight']
                    prior_distn[sb] += pa * pab
            prior_distn = dict(prior_distn)

        # Compute the posterior distribution.
        # This accounts for the marginal distribution at the parent node,
        # the matrix of transition probabilities between the parent node
        # and the current node, and the subtree likelihood conditional
        # on the state of the current node.
        if prior_distn is None:
            if len(set(node_to_pmap[node])) == 1:
                state = get_first_element(node_to_pmap[node])
                if node_to_pmap[node][state]:
                    node_to_distn[node] = {state : 1.0}
                else:
                    raise NumericalZeroProb
            else:
                raise StructuralZeroProb
        else:
            node_to_distn[node] = _mjp.get_zero_step_posterior_distn(
                    prior_distn, node_to_pmap[node])

    # Return the marginal state distributions at nodes.
    return node_to_distn


# XXX add tests
def get_joint_endpoint_state_distn(T, node_to_pmap, node_to_distn, root):
    """

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree with edges annotated with sparse transition probability
        matrices as directed weighted networkx graphs P.
    node_to_pmap : dict
        Map from node to a map from a state to the subtree likelihood.
        This map incorporates state restrictions.
    node_to_distn : dict
        Conditional marginal state distribution at each node.
    root : integer
        Root state.

    Returns
    -------
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as networkx digraphs.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is writeen in
        single letter caps reminiscent of matrix notation.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        pmap = node_to_pmap[nb]
        P = T[na][nb]['P']
        J = nx.DiGraph()
        total_weight = 0.0
        weighted_edges = []
        for sa, pa in node_to_distn[na].items():
            feasible_states = set(P[sa]) & set(node_to_pmap[nb])
            for sb in feasible_states:
                edge = P[sa][sb]
                pab = edge['weight']
                joint_weight = pa * pab * pmap[sb]
                weighted_edges.append((sa, sb, joint_weight))
                total_weight += joint_weight
        for sa, sb, weight in weighted_edges:
            J.add_edge(sa, sb, weight / total_weight)
        T_aug.add_edge(na, nb, J=J)
    return T_aug


# XXX add tests
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
    states = sorted(Q)
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

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    node_to_distn = get_node_to_distn(T_aug, node_to_pmap, root)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    T_joint = get_joint_endpoint_state_distn(
            T_aug, node_to_pmap, node_to_distn, root)

    # Compute the expectations of the dwell times and the transition counts
    # by iterating over all edges and using the edge-specific
    # joint distribution of the states at the edge endpoints.
    expected_dwell_times = defaultdict(float)
    expected_transitions = nx.DiGraph()
    for na, nb in nx.bfs_edges(T, root):

        # Get the elapsed time along the edge.
        t = T[na][nb]['weight']

        # Get the conditional probability matrix associated with the edge.
        P = T_aug[na][nb]['P']

        # Get the joint probability matrix associated with the edge.
        J = T_joint[na][nb]['J']

        # Compute contributions to dwell time expectations along the path.
        for sc_index, sc in enumerate(states):
            C = np.zeros((nstates, nstates), dtype=float)
            C[sc_index, sc_index] = 1.0
            interact = scipy.linalg.expm_frechet(t*Q_dense, t*C)
            for sa_index, sa in enumerate(states):
                for sb_index, sb in enumerate(states):
                    if not J.has_edge(sa, sb):
                        continue
                    cond_prob = P[sa][sb]['weight']
                    joint_prob = J[sa][sb]['weight']
                    # XXX simplify the joint_prob / cond_prob
                    expected_dwell_times[sc] += (
                            joint_prob *
                            interact[sa_index, sb_index] /
                            cond_prob)

        # Compute contributions to transition count expectations.
        for sc_index, sc in enumerate(states):
            for sd_index, sd in enumerate(states):
                if not Q.has_edge(sc, sd):
                    continue
                C = np.zeros((nstates, nstates), dtype=float)
                C[sc_index, sd_index] = 1.0
                interact = scipy.linalg.expm_frechet(t*Q_dense, t*C)
                for sa_index, sa in enumerate(states):
                    for sb_index, sb in enumerate(states):
                        if not J.has_edge(sa, sb):
                            continue
                        cond_prob = P[sa][sb]['weight']
                        joint_prob = J[sa][sb]['weight']
                        contrib = (
                                joint_prob *
                                Q_dense[sc_index, sd_index] *
                                interact[sc_index, sd_index] /
                                cond_prob)
                        if not expected_transitions.has_edge(sc, sd):
                            expected_transitions.add_edge(sc, sd, weight=0.0)
                        expected_transitions[sc][sd]['weight'] += contrib

    return dict(expected_dwell_times), expected_transitions

