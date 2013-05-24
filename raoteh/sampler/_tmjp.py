"""
Tolerance class Markov jump process functions for testing the Rao-Teh sampler.

This more complicated process multiplexes multiple binary tolerance states
together with a primary Markov jump process in a way that has a complicated
conditional dependence structure.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip)

from raoteh.sampler._mjp import (
        get_total_rates, get_conditional_transition_matrix)


__all__ = []


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


def get_tolerance_process_log_likelihood(Q, state_to_part, T, node_to_tmap,
        rate_off, rate_on, root_distn, root=None):
    """

    The direct contribution of the primary process is through its
    state distribution at the root, and its transitions.
    Each tolerance class also contributes.

    Parameters
    ----------
    Q : networkx graph
        Primary process state transition rate matrix.
    state_to_part : dict
        Maps the primary state to the tolerance class.
    T : networkx tree
        Primary process history,
        with edges annotated with primary state and with weights.
    node_to_tmap : dict
        A map from a node to a known tolerance map.
        The map of known tolerances maps tolerance classes to sets
        of allowed tolerance states.
        This map is sparse and is meant to represent known or partially
        known data at the tips of the tree.
    rate_off : float
        Transition rate from tolerance state 1 to tolerance state 0.
    rate_on : float
        Transition rate from tolerance state 0 to tolerance state 1.
    root_distn : dict
        A prior distribution over the primary process root states.
    root : integer, optional
        A node that does not represent a primary state transition.
        If a root is not provided, a root will be picked arbitrarily
        among the tips of the tree.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the compound primary and tolerance process.

    Notes
    -----
    This function returns a log likelihood instead of a likelihood
    because this likelihood is most naturally expressed as a product.
    On the other hand, functions for which the likelihood is most
    naturally expressed as a sum will prefer to return a likelihood rather than
    a log likelihood.

    """
    # Initialize the log likelihood.
    log_likelihood = 0.0

    # Get the root state and the transitions of the primary process.
    root_state, transitions = get_history_root_state_and_transitions(
            T, root=root)

    # For the primary process, get the total rate away from each state.
    total_rates = get_total_rates(Q)

    # Construct a transition matrix conditional on a state change.
    P = get_conditional_transition_matrix(Q, total_rates)

    # Add the log likelihood contribution of the primary thread.
    log_likelihood += np.log(root_distn[root_state])
    for a, b in transitions.edges():
        ntrans = transitions[a][b]['weight']
        ptrans = P[a][b]['weight']
        log_likelihood += scipy.special.xlogy(ntrans, ptrans)

    # Add the log likelihood contribution of the process
    # associated with each tolerance class.
    tolerance_classes = set(state_to_part.values())
    for part in tolerance_classes:
        node_to_tolerance_in = {}
        for node in T:
            if node in node_to_tmap:
                tmap = node_to_tmap[node]
                if part in tmap:
                    node_to_tolerance_in[node] = tmap[part]
        tolerance_class_likelihood = get_tolerance_class_likelihood(
                Q, state_to_part, T, part, node_to_tolerance_in,
                rate_off, rate_on, root=root)
        log_likelihood += np.log(tolerance_class_likelihood)

    # Return the log likelihood for the entire process.
    return log_likelihood

