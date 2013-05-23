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

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb, get_first_element)


__all__ = []


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

    # Bookkeeping structures.
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


def get_tolerance_class_likelihood(
        Q, state_to_part, T, part, node_to_tolerance_in,
        rate_off, rate_on, root=None):
    """
    Compute a likelihood associated with a tolerance class.

    This proceeds in several steps.
    The first step is to construct a tolerance substrate
    that encodes properties of the tolerance class with respect
    to the underlying primary process history.

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
    rate_off : float
        Transition rate from tolerance state 1 to tolerance state 0.
    rate_on : float
        Transition rate from tolerance state 0 to tolerance state 1.
    root : integer, optional
        The root of the tree.
        The root should not matter, because the process is time-reversible.
        But it can be optionally specified to facilitate testing.

    Returns
    -------
    likelihood : float
        The likelihood for the tolerance class of interest.

    """
    # Pick an arbitrary root.
    # Because this particular conditional process is time-reversible,
    # there is no constraint on which nodes are allowed to be roots.
    if root is None:
        root = get_first_element(T)

    # Define the prior distribution on the tolerance state at the root.
    # The location of the root does not matter for this purpose,
    # because the binary tolerance process is time-reversible.
    # This is sparsely defined to allow extreme rates.
    root_prior = {}
    total_rate = rate_off + rate_on
    if (rate_off < 0) or (rate_off < 0) or (total_rate <= 0):
        raise ValueError(
                'the tolerance rate_on and rate_off must both be nonzero '
                'and at least one of them must be positive')
    if rate_off:
        root_prior[0] = rate_off / total_rate
    if rate_on:
        root_prior[1] = rate_on / total_rate

    # Get a tree with edges annotated
    # with tolerance state (if known and constant along the edge)
    # and with absorption rate (if any).
    # The tolerance states of the nodes are further specified
    # to be consistent with edges that have known constant tolerance state.
    T_tol, node_to_tolerance = get_tolerance_substrate(
            Q, state_to_part, T, part, node_to_tolerance_in)

    # Construct the map from nodes to allowed states.
    # This implicitly includes the restriction
    # that the absorbing tolerance state 2 is disallowed.
    node_to_allowed_states = {}
    for node, tolerance in node_to_tolerance.items():
        node_to_allowed_states = {tolerance}
    for node in set(T) - set(node_to_tolerance):
        node_to_allowed_states = {0, 1}

    # Construct yet another tree with annotated edges.
    # This one will have a transition matrix on each edge.
    T_aug = nx.Graph()
    for a, b in nx.bfs_edges(T_tol, root):
        edge = T_tol[a][b]
        weight = edge['weight']

        # Define the rates required for the tolerance rate matrix.
        local_rate_on = rate_on
        local_rate_off = rate_off
        local_absorption_rate = 0.0
        if 'tolerance' in edge:
            const_tolerance = edge['tolerance']
            if const_tolerance == 0:
                local_rate_on = 0.0
            elif const_tolerance == 1:
                local_rate_off = 0.0
            else:
                raise ValueError('unknown tolerance state')
        if 'absorption' in edge:
            local_absorption_rate = edge['absorption']

        # Construct the tolerance rate matrix as a dense ndarray.
        # Then convert it to a transition matrix using expm.
        # Then convert the transition matrix to a networkx digraph.
        R_local = get_tolerance_micro_rate_matrix(
                local_rate_off, local_rate_on, local_absorption_rate)
        P_local_ndarray = scipy.linalg.expm(weight * R_local)
        P_local_nx = nx.DiGraph()
        for i in range(3):
            for j in range(3):
                P_local_nx.add_edge(i, j, weight=P_local_ndarray[i, j])
        T_aug.add_edge(a, b, P=P_local)

    # Get the likelihood from the augmented tree and the root distribution.
    likelihood = get_restricted_likelihood(
            T_aug, root, node_to_allowed_states, root_distn)

    # Return the likelihood.
    return likelihood


def get_tolerance_process_log_likelihood(Q, T, root_distn, root=None):
    """

    The direct contribution of the primary process is through its
    state distribution at the root, and its transitions.

    Parameters
    ----------
    Q : networkx graph
        Primary process state transition rate matrix.
    T : networkx tree
        Primary process history.
    root_distn : dict
        A prior distribution over the primary process root states.
    root : integer, optional
        A node that does not represent a primary state transition.
        If a root is not provided, a root will be picked arbitrarily
        among the tips of the tree.

    Returns
    -------
    likelihood : float
        The likelihood of the compound primary and tolerance process.

    """
    # Initialize the log likelihood.
    log_likelihood = 0.0

    # Get the root state and the transitions of the primary process.
    root_state, transitions = get_history_root_state_and_transitions(
            T, root=root)

    # For the primary process, get the total rate away from each state.
    total_rates = {}
    for a in Q:
        rate_out = 0.0
        for b in Q[a]:
            rate_out += Q[a][b]['weight']
        total_rates[a] = rate_out

    # Construct a transition matrix conditional on a state change.
    P = nx.DiGraph()
    for a in Q:
        for b in Q[a]:
            weight = Q[a][b]['weight'] / rate_out[a]
            P.add_edge(a, b, weight=weight)

    # Add the log likelihood contribution of the primary thread.
    log_likelihood += np.log(root_distn[root_state])
    for a, b in transitions.edges():
        ntrans = transitions[a][b]['weight']
        ptrans = P[a][b]['weight']
        log_likelihood += scipy.special.xlogy(ntrans, ptrans)

    # Add the log likelihood contribution of the process
    # associated with each tolerance class.
    #XXX under construction
    for tolerance_class in foo:
        ll = get_tolerance_class_likelihood(
                Q, state_to_part, T, part, node_to_tolerance_in,
                rate_off, rate_on, root=root)
        log_likelihood += np.log(ll)

    # Return the log likelihood for the entire process.
    return log_likelihood


#XXX under construction (or destruction)
def get_blink_thread_likelihood(
        Q, state_to_part, T, part, node_to_tolerance_in, root=None):

    #def get_dynamic_blink_thread_log_likelihood(
            #part, partition, dg, G_dag,
            #rate_on, rate_off):

    T_aug = get_tolerance_substrate(
            Q, state_to_part, T, part, node_to_tolerance_in, root=None)

    #XXX old content after this...
    #def get_dynamic_blink_thread_log_likelihood(
            #part, partition, dg, G_dag,
            #rate_on, rate_off):
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

