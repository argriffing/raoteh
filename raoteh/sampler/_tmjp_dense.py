"""
Tolerance class Markov jump process functions for testing the Rao-Teh sampler.

This more complicated process multiplexes multiple binary tolerance states
together with a primary Markov jump process in a way that has a complicated
conditional dependence structure.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools

import numpy as np
import networkx as nx
import scipy.linalg
from scipy import special

from raoteh.sampler import _mc0, _mc, _mcy, _mjp

from raoteh.sampler._util import (
        StructuralZeroProb,
        NumericalZeroProb,
        get_first_element,
        get_arbitrary_tip,
        )

from raoteh.sampler._linalg import (
        sparse_expm,
        expm_frechet_is_simple,
        simple_expm_frechet,
        )

from raoteh.sampler._mjp import (
        get_history_root_state_and_transitions,
        get_total_rates,
        get_conditional_transition_matrix,
        get_expected_history_statistics,
        get_expm_augmented_tree,
        )


__all__ = []


def get_tolerance_rate_matrix(rate_off, rate_on):
    """

    Parameters
    ----------
    rate_off : float
        Rate of tolerance transition from on to off.
    rate_on : float
        Rate of tolerance transition from off to on.

    Returns
    -------
    Q_tolerance : 2d ndarray
        Tolerance state transition rate matrix.

    """
    Q_tolerance = np.array([
        [-rate_on, rate_on],
        [rate_off, -rate_off]], dtype=float)
    return Q_tolerance


def get_tolerance_distn(rate_off, rate_on):
    """

    Parameters
    ----------
    rate_off : float
        Rate of tolerance transition from on to off.
    rate_on : float
        Rate of tolerance transition from off to on.

    Returns
    -------
    tolerance_distn : 1d ndarray
        Distribution over the tolerance states 0 and 1.
        Tolerance state 0 is off, and tolerance state 1 is on.

    """
    if (rate_off < 0) or (rate_on < 0):
        raise ValueError('rates must be non-negative')
    total_tolerance_rate = rate_off + rate_on
    if total_tolerance_rate <= 0:
        raise ValueError('the total tolerance rate must be positive')
    unnormal_tolerance_distn = np.array([rate_off, rate_on], dtype=float)
    tolerance_distn = unnormal_tolerance_distn / total_tolerance_rate
    return tolerance_distn


def get_tolerance_process_log_likelihood(
        Q_primary, primary_to_part, T_primary,
        rate_off, rate_on, primary_root_distn, root):
    """

    The direct contribution of the primary process is through its
    state distribution at the root, and its transitions.
    Each tolerance class also contributes.

    Parameters
    ----------
    Q_primary : 2d ndarray
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
    primary_root_distn : ndarray
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
    tolerance_distn = get_tolerance_distn(rate_off, rate_on)

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

        # Construct the piecewise homogeneous Markov jump process.
        T_tol, node_to_allowed_tolerances = get_inhomogeneous_mjp(
                primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
                tolerance_class)

        # Get the likelihood from the augmented tree and the root distribution.
        likelihood = _mjp.get_likelihood(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=root_tolerance_prior, Q_default=None)

        # Contribute to the log likelihod.
        log_likelihood += np.log(likelihood)

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
    absorption_expectation : float
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
    node_to_pmap = _mcy.get_node_to_pmap(T_aug, root,
            node_to_allowed_states=node_to_allowed_states)

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    node_to_distn = _mc0.get_node_to_distn(T_aug, root, node_to_pmap,
            root_distn=root_distn)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    T_joint = _mc0.get_joint_endpoint_distn(
            T_aug, root, node_to_pmap, node_to_distn)

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
    rate_on : x
        x
    rate_off : x
        x
    Q_primary : x
        x
    T_primary : x
        x
    root : x
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
    tolerance_distn = get_tolerance_distn(rate_off, rate_on)

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
    Get a restricted piecewise homogeneous Markov jump process.

    Parameters
    ----------
    primary_to_part : dict
        Map from primary state to tolerance class.
    rate_on : float
        Transition rate from tolerance state off to on.
    rate_off : float
        Transition rate from tolerance state on to off.
    Q_primary : directed weighted networkx graph
        Primary process rate matrix.
    T_primary : weighted undirected networkx graph
        Primary process trajectory.
    root : integer
        Root node
    tolerance_class : integer
        The tolerance class under consideration.

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


def get_primary_proposal_rate_matrix(
        Q_primary, primary_to_part, tolerance_distn):
    """
    Get the rate matrix that approximates the primary process.

    The primary process is not a Markov process because it is
    dependent on the simultaneously evolving tolerance processes.
    But it can be approximated by a Markov process.
    The approximation becomes exact in the limit as the
    tolerance process rates go to infinity.

    Parameters
    ----------
    Q_primary : directed weighted networkx graph
        A sparse transition rate matrix.
        The diagonal entries are assumed to be missing.
    primary_to_part : dict
        Maps the primary state to its tolerance class.
    tolerance_distn : dict
        The sparse distribution over tolerance states 0 and 1.

    Returns
    -------
    Q_primary_proposal : directed weighted networkx graph
        A sparse transition rate matrix.
        The diagonal entries are assumed to be missing.
        It is related to the primary process transition rate matrix,
        but the between-tolerance-class rates may be reduced.

    Notes
    -----
    Define a rate matrix for a primary process proposal distribution.
    This is intended to define a Markov jump process for primary states
    which approximates the non-Markov jump process for primary states
    defined by the marginal primary component of the compound process.
    This biased proposal primary process can be used for either
    importance sampling or for a Metropolis-Hastings step
    within the Rao-Teh sampling.
    It is also used to help construct an initial feasible trajectory.

    """
    Q_proposal = nx.DiGraph()
    for sa, sb in Q_primary.edges():
        primary_rate = Q_primary[sa][sb]['weight']
        if primary_to_part[sa] == primary_to_part[sb]:
            Q_proposal.add_edge(sa, sb, weight=primary_rate)
        elif 1 in tolerance_distn:
            proposal_rate = primary_rate * tolerance_distn[1]
            Q_proposal.add_edge(sa, sb, weight=proposal_rate)
    return Q_proposal


def get_example_tolerance_process_info(
        tolerance_rate_on, tolerance_rate_off):
    """
    Construct a toy model intended for testing.

    Parameters
    ----------
    tolerance_rate_on : float
        Transition rate from tolerance state off to tolerance state on.
    tolerance_rate_off : float
        Transition rate from tolerance state on to tolerance state off.

    Returns
    -------
    primary_distn : dict
        Primary process state distribution.
    Q : weighted directed networkx graph
        Sparse primary process transition rate matrix.
    primary_to_part : dict
        Maps primary state to tolerance class.
    compound_to_primary : list
        Ordered list of primary states for the compound states.
    compound_to_tolerances : list
        Ordered list of tolerance classes for the compound states.
    compound_distn : dict
        Map from the compound state to its probability.
        This will be the stationary distribution of a time-reversible process
        if the primary process is time-reversible.
    Q_compound : weighted directed networkx graph
        Sparse compound process transition rate matrix.

    """
    # Define a distribution over some primary states.
    nprimary = 6
    primary_distn = {
            0 : 0.05,
            1 : 0.1,
            2 : 0.15,
            3 : 0.2,
            4 : 0.25,
            5 : 0.25}

    # Define the transition rates.
    primary_transition_rates = [
            (0, 1, 2 * primary_distn[1]),
            (1, 0, 2 * primary_distn[0]),
            (1, 2, primary_distn[2]),
            (2, 1, primary_distn[1]),
            (2, 3, 3 * primary_distn[3]),
            (3, 2, 3 * primary_distn[2]),
            (3, 4, primary_distn[4]),
            (4, 3, primary_distn[3]),
            (4, 5, primary_distn[5]),
            (5, 4, primary_distn[4]),
            (5, 0, primary_distn[0]),
            (0, 5, primary_distn[5]),
            ]

    # Define the primary process through its transition rate matrix.
    Q = nx.DiGraph()
    Q.add_weighted_edges_from(primary_transition_rates)

    # Define some tolerance process stuff.
    tolerance_distn = get_tolerance_distn(tolerance_rate_off, tolerance_rate_on)

    # Define a couple of tolerance classes.
    nparts = 3
    primary_to_part = {
            0 : 0,
            1 : 0,
            2 : 1,
            3 : 1,
            4 : 2,
            5 : 2}

    # Define a compound state space.
    compound_to_primary = []
    compound_to_tolerances = []
    for primary, tolerances in itertools.product(
            range(nprimary),
            itertools.product((0, 1), repeat=nparts)):
        compound_to_primary.append(primary)
        compound_to_tolerances.append(tolerances)

    # Define the sparse distribution over compound states.
    compound_distn = {}
    for i, (primary, tolerances) in enumerate(
            zip(compound_to_primary, compound_to_tolerances)):
        part = primary_to_part[primary]
        if tolerances[part] == 1:
            p_primary = primary_distn[primary]
            p_tolerances = 1.0
            for tolerance_class, tolerance_state in enumerate(tolerances):
                if tolerance_class != part:
                    p_tolerances *= tolerance_distn[tolerance_state]
            compound_distn[i] = p_primary * p_tolerances

    # Check the number of entries in the compound state distribution.
    if len(compound_distn) != nprimary * (1 << (nparts - 1)):
        raise Exception('internal error')

    # Check that the distributions have the correct normalization.
    # The loop is unrolled to better isolate errors.
    if not np.allclose(sum(primary_distn.values()), 1):
        raise Exception('internal error')
    if not np.allclose(sum(tolerance_distn.values()), 1):
        raise Exception('internal error')
    if not np.allclose(sum(compound_distn.values()), 1):
        raise Exception('internal error')

    # Define the compound transition rate matrix.
    # Use compound_distn to avoid formal states with zero probability.
    # This is slow, but we do not need to be fast.
    Q_compound = nx.DiGraph()
    for i in compound_distn:
        for j in compound_distn:
            if i == j:
                continue
            i_prim = compound_to_primary[i]
            j_prim = compound_to_primary[j]
            i_tols = compound_to_tolerances[i]
            j_tols = compound_to_tolerances[j]
            tol_pairs = list(enumerate(zip(i_tols, j_tols)))
            tol_diffs = [(k, x, y) for k, (x, y) in tol_pairs if x != y]
            tol_hdist = len(tol_diffs)

            # Look for a tolerance state change.
            # Do not allow simultaneous primary and tolerance changes.
            # Do not allow more than one simultaneous tolerance change.
            # Do not allow changes to the primary tolerance class.
            if tol_hdist > 0:
                if i_prim != j_prim:
                    continue
                if tol_hdist > 1:
                    continue
                part, i_tol, j_tol = tol_diffs[0]
                if part == primary_to_part[i_prim]:
                    continue

                # Add the transition rate.
                if j_tol:
                    weight = tolerance_rate_on
                else:
                    weight = tolerance_rate_off
                Q_compound.add_edge(i, j, weight=weight)

            # Look for a primary state change.
            # Do not allow simultaneous primary and tolerance changes.
            # Do not allow a change to a non-tolerated primary class.
            # Do not allow transitions that have zero rate
            # in the primary process.
            if i_prim != j_prim:
                if tol_hdist > 0:
                    continue
                if not i_tols[primary_to_part[j_prim]]:
                    continue
                if not Q.has_edge(i_prim, j_prim):
                    continue
                
                # Add the primary state transition rate.
                weight = Q[i_prim][j_prim]['weight']
                Q_compound.add_edge(i, j, weight=weight)

    return (primary_distn, Q, primary_to_part,
            compound_to_primary, compound_to_tolerances, compound_distn,
            Q_compound)

