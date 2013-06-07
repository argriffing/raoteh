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
import scipy.linalg
from scipy import special

from raoteh.sampler import _mc, _mjp

from raoteh.sampler._util import (
        StructuralZeroProb,
        NumericalZeroProb,
        get_first_element,
        get_arbitrary_tip,
        sparse_expm,
        expm_frechet_is_simple,
        simple_expm_frechet,
        )

from raoteh.sampler._mc import (
        construct_node_to_restricted_pmap,
        get_node_to_distn,
        get_joint_endpoint_distn,
        )

from raoteh.sampler._mjp import (
        get_history_root_state_and_transitions,
        get_total_rates,
        get_conditional_transition_matrix,
        get_expected_history_statistics,
        get_expm_augmented_tree,
        )


__all__ = []


def get_tolerance_substrate(
        Q, primary_to_part, T, part, root):
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
    primary_to_part : dict
        Maps the primary state to the tolerance class.
    T : weighted undirected networkx tree
        The primary process history on a tree.
        Each edge is annotated with a weight and with a primary process state.
        The edge weight is interpreted as a distance or length.
    part : integer
        The tolerance class of interest.

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
    # Check the root.
    if root is None:
        raise ValueError('unspecified root')
    if root not in T:
        raise ValueError('the specified root is not a node in the tree')

    # Build the output tree, edge by edge,
    # and populate the map from node to known tolerance state.
    node_to_tolerance_out = {}
    T_out = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):

        # Get the weight and the primary process state associated with the edge.
        primary_state = T[na][nb]['state']
        weight = T[na][nb]['weight']
        
        # Add the edge, annotated with the weight.
        T_out.add_edge(na, nb, weight=weight)

        # If the primary process state of the edge
        # belongs to the tolerance class of interest,
        # then the tolerance state along the edge
        # and at both edge endpoints must be 1.
        if primary_to_part[primary_state] == part:

            # Set the edge tolerance state.
            T_out[na][nb]['tolerance'] = 1

            # Set the node tolerance states.
            for node in (na, nb):
                if node in node_to_tolerance_out:
                    if node_to_tolerance_out[node] != 1:
                        raise ValueError('incompatible tolerance state')
                node_to_tolerance_out[node] = 1

        # The absorption rate along this edge will be the sum of rates
        # from the edge state to states in the target tolerance class.
        # If the absorption rate is structurally zero
        # because no such transitions are allowed,
        # then do not add the absorption rate to the annotation.
        sbs = [sb for sb in Q[primary_state] if primary_to_part[sb] == part]
        if sbs:
            absorption_rate = sum(Q[primary_state][sb]['weight'] for sb in sbs)
            T_out[na][nb]['absorption'] = absorption_rate

    # Return the annotated networkx tree and the node to tolerance state map.
    return T_out, node_to_tolerance_out


def get_tolerance_class_likelihood(
        Q, primary_to_part, T, part,
        rate_off, rate_on, root):
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
    primary_to_part : dict
        Maps the primary state to the tolerance class.
    T : weighted undirected networkx tree
        The primary process history on a tree.
        Each edge is annotated with a weight and with a primary process state.
        The edge weight is interpreted as a distance or length.
    part : integer
        The tolerance class of interest.
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
    # Check the root.
    if root is None:
        raise ValueError('unspecified root')
    if root not in T:
        raise ValueError('the specified root is not a node in the tree')

    # Get the root state.
    root_neighbor = get_first_element(T[root])
    root_primary_state = T[root][root_neighbor]['state']

    # Define the distribution over tolerance states.
    tolerance_distn = {}
    total_rate = rate_off + rate_on
    if (rate_off < 0) or (rate_off < 0) or (total_rate <= 0):
        raise ValueError(
                'the tolerance rate_on and rate_off must both be nonzero '
                'and at least one of them must be positive')
    if rate_off:
        tolerance_distn[0] = rate_off / total_rate
    if rate_on:
        tolerance_distn[1] = rate_on / total_rate

    # If the tolerance class of the primary state of the root
    # is equal to the current tolerance class,
    # then the root tolerance state does not have an interesting
    # prior distribution.
    # Otherwise, the root tolerance state prior distribution is
    # given by the equilibrium tolerance state distribution.
    if primary_to_part[root_primary_state] == part:
        root_prior = {1 : 1}
    else:
        root_prior = tolerance_distn

    # Get a tree with edges annotated
    # with tolerance state (if known and constant along the edge)
    # and with absorption rate (if any).
    # The tolerance states of the nodes are further specified
    # to be consistent with edges that have known constant tolerance state.
    T_tol, node_to_tolerance = get_tolerance_substrate(
            Q, primary_to_part, T, part, root)

    # Construct the map from nodes to allowed states.
    # This implicitly includes the restriction
    # that the absorbing tolerance state 2 is disallowed.
    node_to_allowed_states = {}
    for node, tolerance in node_to_tolerance.items():
        node_to_allowed_states[node] = {tolerance}
    for node in set(T) - set(node_to_tolerance):
        node_to_allowed_states[node] = {0, 1}

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

        # Construct the local tolerance rate matrix.
        Q_tol = nx.DiGraph()
        if local_rate_on:
            Q_tol.add_edge(0, 1, weight=local_rate_on)
        if local_rate_off:
            Q_tol.add_edge(1, 0, weight=local_rate_off)
        if local_absorption_rate:
            Q_tol.add_edge(1, 2, weight=local_absorption_rate)

        # Construct the local transition matrix.
        P_local_nx = sparse_expm(Q_tol, weight)

        # Add the transition matrix.
        T_aug.add_edge(a, b, P=P_local_nx)

    # Get the likelihood from the augmented tree and the root distribution.
    likelihood = _mc.get_restricted_likelihood(
            T_aug, root, node_to_allowed_states,
            root_distn=root_prior, P_default=None)

    # Return the likelihood.
    return likelihood


def get_tolerance_process_log_likelihood(
        Q_primary, primary_to_part, T_primary,
        rate_off, rate_on, primary_root_distn, root):
    """

    The direct contribution of the primary process is through its
    state distribution at the root, and its transitions.
    Each tolerance class also contributes.

    Parameters
    ----------
    Q_primary : networkx graph
        Primary process state transition rate matrix.
    primary_to_part : dict
        Maps the primary state to the tolerance class.
    T_primary : networkx tree
        Primary process history,
        with edges annotated with primary state and with weights.
    rate_off : float
        Transition rate from tolerance state 1 to tolerance state 0.
    rate_on : float
        Transition rate from tolerance state 0 to tolerance state 1.
    primary_root_distn : dict
        A prior distribution over the primary process root states.
    root : integer
        A node that does not represent a primary state transition.

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
    # Check the root.
    if root is None:
        raise ValueError('unspecified root')
    if root not in T_primary:
        raise ValueError('the specified root is not a node in the tree')

    # Define the distribution over tolerance states.
    tolerance_distn = {}
    total_rate = rate_off + rate_on
    if (rate_off < 0) or (rate_off < 0) or (total_rate <= 0):
        raise ValueError(
                'the tolerance rate_on and rate_off must both be nonzero '
                'and at least one of them must be positive')
    if rate_off:
        tolerance_distn[0] = rate_off / total_rate
    if rate_on:
        tolerance_distn[1] = rate_on / total_rate

    # Get the root state and the transitions of the primary process.
    info = get_history_root_state_and_transitions(T_primary, root=root)
    primary_root_state, primary_transitions = info

    # Initialize the log likelihood.
    log_likelihood = 0.0

    # Add the log likelihood contribution of the primary thread.
    log_likelihood += np.log(primary_root_distn[primary_root_state])
    for sa in set(primary_transitions) & set(Q_primary):
        for sb in set(primary_transitions[sa]) & set(Q_primary[sa]):
            ntransitions = primary_transitions[sa][sb]['weight']
            rate = Q_primary[sa][sb]['weight']
            log_likelihood += scipy.special.xlogy(ntransitions, rate)

    # Add the log likelihood contribution of the process
    # associated with each tolerance class.
    tolerance_classes = set(primary_to_part.values())
    for tolerance_class in tolerance_classes:

        # If the tolerance class of the primary state of the root
        # is equal to the current tolerance class,
        # then the root tolerance state does not have an interesting
        # prior distribution.
        # Otherwise, the root tolerance state prior distribution is
        # given by the equilibrium tolerance state distribution.
        if primary_to_part[primary_root_state] == tolerance_class:
            root_tolerance_prior = {1 : 1}
        else:
            root_tolerance_prior = tolerance_distn

        # Construct the piecewise homogeneous  Markov jump process.
        T_tol, node_to_allowed_tolerances = get_inhomogeneous_mjp(
                primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
                tolerance_class)

        # Get the likelihood from the augmented tree and the root distribution.
        likelihood = _mjp.get_likelihood(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=root_tolerance_prior, Q_default=None)

        # Contribute to the log likelihod.
        log_likelihood += np.log(likelihood)

        #tolerance_class_likelihood = get_tolerance_class_likelihood(
                #Q_primary, primary_to_part, T_primary, part,
                #rate_off, rate_on, root=root)
        #log_likelihood += np.log(tolerance_class_likelihood)

    # Return the log likelihood for the entire process.
    return log_likelihood


def get_absorption_integral(T, node_to_allowed_states,
        root, root_distn=None, Q_default=None):
    """
    Compute a certain integral.

    This function is copypasted from get_expected_history_statistics in _mjp.
    Its purpose is to compute a single weird thing -- the expectation
    over all tolerance processes and over all edges,
    of the "absorption rate" multiplied by the expected amount of time
    spent in the "on" tolerance state.
    So this is similar to the on-dwell-time expectation calculation,
    except it has a weird per-branch weighting.

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
    x : x
        expectation

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
    absorption_expectation = 0.0
    for na, nb in nx.bfs_edges(T, root):

        # Get the sparse rate matrix to use for this edge.
        Q = T[na][nb].get('Q', Q_default)
        if Q is None:
            raise ValueError('no rate matrix is available for this edge')

        Q_is_simple = expm_frechet_is_simple(Q)
        if not Q_is_simple:
            raise ValueError(
                    'this function requires a rate matrix '
                    'of a certain form')

        # Get the elapsed time along the edge.
        t = T[na][nb]['weight']

        # Get the conditional probability matrix associated with the edge.
        P = T_aug[na][nb]['P']

        # Get the joint probability matrix associated with the edge.
        J = T_joint[na][nb]['J']

        # Compute contributions to dwell time expectations along the path.
        # The sc_index = 1 signifies the tolerance "on" state.
        branch_absorption_rate = 0
        if Q.has_edge(1, 2):
            branch_absorption_rate = Q[1][2]['weight']
        branch_absorption_expectation = 0.0
        sc_index = 1
        for sa_index, sa in enumerate(states):
            for sb_index, sb in enumerate(states):
                if not J.has_edge(sa, sb):
                    continue
                cond_prob = P[sa][sb]['weight']
                joint_prob = J[sa][sb]['weight']
                x = simple_expm_frechet(
                        Q, sa_index, sb_index, sc_index, sc_index, t)
                # XXX simplify the joint_prob / cond_prob
                branch_absorption_expectation += (joint_prob * x) / cond_prob
        absorption_expectation += (
                branch_absorption_rate *
                branch_absorption_expectation)

    # Return the expectation.
    return absorption_expectation


def get_tolerance_expectations(
        primary_to_part, rate_on, rate_off, Q_primary, T_primary, root):
    """
    Get tolerance process expectations conditional on a primary trajectory.

    Given a primary process trajectory,
    compute some log likelihood contributions
    related to the tolerance process.

    Parameters
    ----------
    primary_to_part : x
        x
    T_primary : x
        x
    Q_primary : x
        x
    root : x
        x
    rate_on : x
        x
    rate_off : x
        x

    Returns
    -------
    dwell_ll_contrib : float
        Log likelihood contribution of the conditional
        tolerance state dwell times.
    init_ll_contrib : float
        Log likelihood contribution of the conditional
        initial distribution of tolerance states.
    trans_ll_contrib : float
        Log likelihood contribution of the conditional
        tolerance process transition counts and types.

    Notes
    -----
    This function assumes that no tolerance process data is observed
    at the leaves, other than what could be inferred through
    the primary process observations at the leaves.
    The ordering of the arguments of this function was chosen haphazardly.
    The ordering of the return values corresponds to the ordering of
    the return values of the _mjp function that returns
    statistics of a trajectory.

    """
    # Summarize the tolerance process.
    total_weight = T_primary.size(weight='weight')
    nparts = len(set(primary_to_part.values()))
    total_tolerance_rate = rate_on + rate_off
    tolerance_distn = {
            0 : rate_off / total_tolerance_rate,
            1 : rate_on / total_tolerance_rate}

    # Compute conditional expectations of statistics
    # of the tolerance process.
    # This requires constructing independent piecewise homogeneous
    # Markov jump processes for each tolerance class.
    expected_ngains = 0.0
    expected_nlosses = 0.0
    expected_dwell_on = 0.0
    expected_initial_on = 0.0
    expected_nabsorptions = 0.0
    for tolerance_class in range(nparts):

        # Get the restricted inhomogeneous Markov jump process
        # associated with the tolerance class,
        # conditional on the trajectory of the primary state.
        T_tol, node_to_allowed_tolerances =  get_inhomogeneous_mjp(
                primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
                tolerance_class)

        # Compute conditional expectations of dwell times
        # and transitions for this tolerance class.
        expectation_info = get_expected_history_statistics(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=tolerance_distn)
        dwell_times, post_root_distn, transitions = expectation_info

        # Get the dwell time log likelihood contribution.
        if 1 in dwell_times:
            expected_dwell_on += dwell_times[1]

        # Get the transition log likelihood contribution.
        if transitions.has_edge(0, 1):
            expected_ngains += transitions[0][1]['weight']
        if transitions.has_edge(1, 0):
            expected_nlosses += transitions[1][0]['weight']

        # Get the initial state log likelihood contribution.
        if 1 in post_root_distn:
            expected_initial_on += post_root_distn[1]

        # Get an expectation that connects
        # the blinking process to the primary process.
        # This is the expected number of times that
        # a non-forbidden primary process "absorption"
        # into the current blinking state
        # would have been expected to occur.
        expected_nabsorptions += get_absorption_integral(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=tolerance_distn)

    # Summarize expectations.
    expected_initial_off = nparts - expected_initial_on
    expected_dwell_off = total_weight * nparts - expected_dwell_on

    # Return the log likelihood contributions.
    dwell_ll_contrib = -(
            expected_dwell_off * rate_on +
            (expected_dwell_on - total_weight) * rate_off +
            expected_nabsorptions)
    init_ll_contrib = (
            special.xlogy(expected_initial_on - 1, tolerance_distn[1]) +
            special.xlogy(expected_initial_off, tolerance_distn[0]))
    trans_ll_contrib = (
            special.xlogy(expected_ngains, rate_on) +
            special.xlogy(expected_nlosses, rate_off))
    return dwell_ll_contrib, init_ll_contrib, trans_ll_contrib


def get_inhomogeneous_mjp(
        primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
        tolerance_class):
    """
    Get a restricted inhomogeneous Markov jump process.

    Returns
    -------
    T_tol : weighted undirected networkx graph
        The inhomogenous Markov jump process on the tree.
        Edges are annotated with a local 3-state rate matrix,
        where state 0 is untolerated, state 1 is tolerated,
        and state 2 is an absorbing state.
    node_to_allowed_tolerances : dict
        Maps each node to a set of allowed tolerances.
        Each set is either {1} or {0, 1}.

    """
    # Define the set of allowed tolerances at each node.
    # These may be further constrained
    # by the sampled primary process trajectory.
    node_to_allowed_tolerances = dict(
            (n, {0, 1}) for n in T_primary)

    # Construct the tree whose edges are in correspondence
    # to the edges of the sampled primary trajectory,
    # and whose edges are annotated with weights
    # and with edge-specific 3-state transition rate matrices.
    # The third state of each edge-specific rate matrix is an
    # absorbing state which will never be entered.
    T_tol = nx.Graph()
    for na, nb in nx.bfs_edges(T_primary, root):
        edge = T_primary[na][nb]
        primary_state = edge['state']
        local_tolerance_class = primary_to_part[primary_state]
        weight = edge['weight']

        # Define the local on->off rate, off->on rate,
        # and absorption rate.
        local_rate_on = rate_on
        if tolerance_class == local_tolerance_class:
            local_rate_off = 0
        else:
            local_rate_off = rate_off
        absorption_rate = 0
        if primary_state in Q_primary:
            for sb in Q_primary[primary_state]:
                if primary_to_part[sb] == tolerance_class:
                    rate = Q_primary[primary_state][sb]['weight']
                    absorption_rate += rate

        # Construct the local tolerance rate matrix.
        Q_tol = nx.DiGraph()
        if local_rate_on:
            Q_tol.add_edge(0, 1, weight=local_rate_on)
        if local_rate_off:
            Q_tol.add_edge(1, 0, weight=local_rate_off)
        if absorption_rate:
            Q_tol.add_edge(1, 2, weight=absorption_rate)

        # Add the edge.
        T_tol.add_edge(na, nb, weight=weight, Q=Q_tol)

        # Possibly restrict the set of allowed tolerances
        # at the endpoints of the edge.
        if tolerance_class == local_tolerance_class:
            for n in (na, nb):
                node_to_allowed_tolerances[n].discard(0)

    # Return the info.
    return T_tol, node_to_allowed_tolerances

