"""
Likelihood calculations for testing the Rao-Teh sampler.

MJP means Markov jump process.
Functions associated with a more complicated process are also available,
where the more complicated process multiplexes multiple binary tolerance states
together with a primary process in a way that has a complicated
conditional dependence structure.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx


__all__ = []


def get_first_element(elements):
    """
    Returns the first element of an iterable.
    """
    for x in elements:
        return x


def get_arbitrary_tip(T, degrees=None):
    """

    Parameters
    ----------
    T : undirected networkx graph with integer nodes
        An input graph.
    degrees : dict, optional
        Maps nodes to degree.

    Returns
    -------
    tip : integer
        An arbitrary degree-1 node.

    """
    if degrees is None:
        degrees = T.degree()
    tips = (n for n, d in degrees.items() if d == 1)
    return get_first_element(tips)


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
    root_state : integer
        The state at the root.
    dwell_times : dict
        Map from the state to the total dwell time on the tree.
    transition_counts : directed weighted networkx graph
        A networkx graph that tracks the number of times
        each transition type appears in the history.

    Notes
    -----
    These statistics are sufficient to compute the Markov jump process
    likelihood for the sampled history.
        
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

    # Get the dwell times.
    dwell_times = get_history_dwell_times(T)

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
    return root_state, dwell_times, transition_counts


def get_restricted_likelihood(T, root, root_distn, node_to_allowed_states):
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

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices.
    root : integer
        The root node.
    root_distn : dict
        A finite distribution over root states.
    node_to_allowed_states : dict
        A map from a node to a set of allowed states.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    pass


def get_tolerance_micro_rate_matrix(rate_off, rate_on, rate_absorb):
    """
    Compute a rate matrix.

    The rate matrix can be used to compute transition probabilities
    and conditional expectations of dwell times and transition counts.

    Parameters
    ----------
    rate_off : float
        Rate from the 'on' state to the 'off' state.
    rate_on : float
        Rate from the 'off' state to the 'on' state.
    rate_absorb : float
        Rate from the 'on' state to the 'absorbing' state.

    Returns
    -------
    Q : shape (3, 3) ndarray of floats
        This is a continuous time rate matrix as a numpy ndarray.
        The state order is ('off', 'on', 'absorbed').

    """
    Q = np.array([
        [-rate_on, rate_on, 0],
        [rate_off, -(rate_off + rate_absorb), rate_absorb],
        [0, 0, 0]], dtype=float)
    return Q


def get_tolerance_substrate(
        Q, state_to_part, T, part, node_to_tolerance_in, root=None):
    """
    Get a substrate for a tolerance class of interest.

    This is a helper function for computing likelihoods and expectations
    for a complicated continuous time process with an unmanageably
    large state space which must be handled cleverly according
    to certain conditional independences.

    Parameters
    ----------
    Q : directed networkx graph
        A sparse rate matrix
        for which edge weights are interpreted as instantaneous rates.
    state_to_part : dict
        Maps the primary state to the tolerance class.
    T : weighted undirected networkx tree
        The primary process history on a tree.
        Each edge is annotated with a weight and with a primary process state.
        The edge weight is interpreted as a distance or length.
    part : integer
        The tolerance class of interest.
    node_to_tolerance_in : dict
        Maps some nodes to known tolerance states.
        May be empty if no tolerance states are known.
        Some nodes may be adjacent to edges with known tolerance states;
        the tolerance states of these nodes can be deduced,
        but they are not required to be included in this map.

    Returns
    -------
    T_out : weighted undirected networkx tree
        The annotated history.
        Each edge is annotated with a weight
        and optionally an absorption rate and optionally a tolerance state.
        Many edges are likely to have unknown tolerance states.
    node_to_tolerance_out : dict
        Maps nodes in the annotated history to tolerance states.
        Nodes with unknown tolerance states will not appear in the map.
        This map will include all entries of the tolerance map
        provided by the input dict, and it will also include
        tolerance states for nodes that are adjacent to edges
        whose primary state belongs to the tolerance class of interest.

    """
    # Pick a root with only one neighbor if no root was specified.
    if root is None:
        root = get_arbitrary_tip(T)

    # Build the output tree, edge by edge,
    # and populate the map from node to known tolerance state.
    node_to_tolerance_out = dict(node_to_tolerance_in)
    T_out = nx.Graph()
    for a, b in nx.bfs_edges(T, root):

        # Get the weight and the primary process state associated with the edge.
        state = T[a][b]['state']
        weight = T[a][b]['weight']
        
        # Add the edge, annotated with the weight.
        T_out.add_edge(a, b, weight=weight)

        # If the primary process state of the edge
        # belongs to the tolerance class of interest,
        # then the tolerance state along the edge
        # and at both edge endpoints must be 1.
        if state_to_part[state] == part:

            # Set the edge tolerance state.
            T_out[a][b]['tolerance'] = 1

            # Set the node tolerance states.
            for node in (a, b):
                if node in node_to_tolerance_out:
                    if node_to_tolerance_out[node] != 1:
                        raise ValueError('incompatible tolerance state')
                node_to_tolerance_out[node] = 1

        # The absorption rate along this edge will be the sum of rates
        # from the edge state to states in the target tolerance class.
        # If the absorption rate is structurally zero
        # because no such transitions are allowed,
        # then do not add the absorption rate to the annotation.
        sbs = [sb for sb in Q[state] if state_to_part[state] == part]
        if sbs:
            absorption_rate = sum(Q[state][sb] for sb in sbs)
            T_out[a][b]['absorption'] = absorption_rate

    # Return the annotated networkx tree and the node to tolerance state map.
    return T_out, node_to_tolerance_out


def get_dynamic_blink_thread_log_likelihood(
        part, partition, dg, G_dag,
        rate_on, rate_off):
    """
    This uses more-clever-than-brute force likelihood calculation.
    In particular it uses dynamic programming or memoization or whatever.
    @param part: the part of the partition defining the current blink thred
    @param partition: a map from primary state to part
    @param dg: sparse primary state rate matrix as weighted directed networkx
    @param G_dag: directed phylogenetic tree with blen and state edge values
    @param rate_on: a blink rate
    @param rate_off: a blink rate
    @return: log likelihood
    """

    # Beginning at the leaves and working toward the root,
    # compute subtree likelihoods conditional on each blink state.
    v_to_b_to_lk = defaultdict(dict)

    # Initialize the likelihood map of each leaf vertex.
    leaf_set = set(v for v in G_dag if G_dag.degree(v) == 1)
    for leaf in leaf_set:

        # These likelihoods are allowed to be 1 even when
        # the leaf blink state conflicts with the primary state at the leaf,
        # because the conflicts will be handled at the edge level
        # rather than at the vertex level.
        v_to_b_to_lk[leaf][0] = 1.0
        v_to_b_to_lk[leaf][1] = 1.0

    # Work towards the root.
    for v in reversed(nx.topological_sort(G_dag)):
        if v in leaf_set:
            continue

        # prepare to multiply by likelihoods for each successor branch
        v_to_b_to_lk[v][0] = 1.0
        v_to_b_to_lk[v][1] = 1.0
        for succ in G_dag.successors(v):

            # Get the primary state of this segment,
            # and get its corresponding partition part.
            pri_state = G_dag[v][succ]['state']
            blen = G_dag[v][succ]['blen']
            pri_part = partition[part]

            # Get the conditional rate of turning off the blinking.
            # This is zero if the primary state corresponds to the
            # blink thread state, and otherwise it is rate_off.
            if partition[pri_state] == part:
                conditional_rate_off = 0.0
            else:
                conditional_rate_off = rate_off

            # Get the conditional rate of turning on the blinking.
            # This is always rate_on.
            conditional_rate_on = rate_on

            # Get the absorption rate.
            # This is the sum of primary transition rates
            # into the part that corresponds to the current blink thread state.
            rate_absorb = 0.0
            for sink in dg.successors(pri_state):
                rate = dg[pri_state][sink]['weight']
                if partition[sink] == part:
                    rate_absorb += rate

            # Construct the micro rate matrix and transition matrix.
            P_micro = mmpp.get_mmpp_block(
                    conditional_rate_on,
                    conditional_rate_off,
                    rate_absorb,
                    blen,
                    )
            """
            Q_micro_slow = get_micro_rate_matrix(
                    conditional_rate_off, conditional_rate_on, rate_absorb)
            P_micro_slow = scipy.linalg.expm(Q_micro_slow * blen)[:2, :2]
            if not np.allclose(P_micro, P_micro_slow):
                raise Exception((P_micro, P_micro_slow))
            """

            # Get the likelihood using the v_to_b_to_lk map.
            lk_branch = {}
            lk_branch[0] = 0.0
            lk_branch[1] = 0.0
            for ba, bb in product((0, 1), repeat=2):
                lk_transition = 1.0
                if partition[pri_state] == part:
                    if not (ba and bb):
                        lk_transition *= 0.0
                lk_transition *= P_micro[ba, bb]
                lk_rest = v_to_b_to_lk[succ][bb]
                lk_branch[ba] += lk_transition * lk_rest

            # Multiply by the likelihood associated with this branch.
            v_to_b_to_lk[v][0] *= lk_branch[0]
            v_to_b_to_lk[v][1] *= lk_branch[1]

    # get the previously arbitrarily chosen phylogenetic root vertex
    root = nx.topological_sort(G_dag)[0]

    # get the primary state at the root
    root_successor = G_dag.successors(root)[0]
    initial_primary_state = G_dag[root][root_successor]['state']

    # The initial distribution contributes to the likelihood.
    if partition[initial_primary_state] == part:
        initial_proportion_off = 0.0
        initial_proportion_on = 1.0
    else:
        initial_proportion_off = rate_off / float(rate_on + rate_off)
        initial_proportion_on = rate_on / float(rate_on + rate_off)
    path_likelihood = 0.0
    path_likelihood += initial_proportion_off * v_to_b_to_lk[root][0]
    path_likelihood += initial_proportion_on * v_to_b_to_lk[root][1]

    # Report the log likelihood.
    return math.log(path_likelihood)

