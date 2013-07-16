"""
Tolerance class Markov jump process functions for testing the Rao-Teh sampler.

This more complicated process multiplexes multiple binary tolerance states
together with a primary Markov jump process in a way that has a complicated
conditional dependence structure.

This module uses dense numpy ndarrays to represent state matrices and vectors.

"""
from __future__ import division, print_function, absolute_import

import itertools

import numpy as np
import networkx as nx
import scipy.linalg
from scipy import special

import pyfelscore

from raoteh.sampler import (
        _density,
        _mc0_dense,
        _mcy_dense,
        _mjp_dense,
        _tmjp_util,
        )


__all__ = []



class CompoundToleranceModel(object):
    """
    A read-only aggregation of information describing an evolutionary process.

    """
    def __init__(self,
            Q_primary, primary_distn, primary_to_part,
            rate_on, rate_off):
        """

        Parameters
        ----------
        Q_primary : 2d ndarray
            x
        primary_distn : 1d ndarray
            x
        primary_to_part : dict
            x
        rate_on : float
            x
        rate_off : float
            x

        """
        # Store the inputs.
        self.Q_primary = Q_primary
        self.primary_distn = primary_distn
        self.primary_to_part = primary_to_part
        self.rate_on = rate_on
        self.rate_off = rate_off

        # Precompute some summaries which can be computed quickly
        # and do not use much memory.
        # Summaries that are slow or complicated to compute or which may use
        # too much memory are computed on demand
        # through an explicit function call.
        self.nprimary = len(primary_to_part)
        self.nparts = len(set(primary_to_part.values()))
        self.ncompound = int(np.ldexp(self.nprimary, self.nparts))
        self.tolerance_distn = get_two_state_tolerance_distn(rate_off, rate_on)

        # Mark some attributes as un-initialized.
        # These attributes are related to the compound distribution,
        # and may be initialized later using init_compound().
        self.Q_compound = None
        self.compound_distn = None
        self.compound_to_primary = None
        self.compound_to_tolerances = None

    def init_compound(self):
        """

        """
        if self.Q_compound is not None:
            raise Exception(
                    'compound attributes should be initialized only once')
        if self.ncompound > 1e6:
            raise Exception(
                    'the compound state space is too big')

        # Define a compound state space.
        self.compound_to_primary = []
        self.compound_to_tolerances = []
        for primary, tolerances in itertools.product(
                range(self.nprimary),
                itertools.product((0, 1), repeat=self.nparts)):
            self.compound_to_primary.append(primary)
            self.compound_to_tolerances.append(tolerances)

        # Define the distribution over compound states.
        self.compound_distn = np.zeros(self.ncompound)
        for i, (primary, tolerances) in enumerate(
                zip(self.compound_to_primary, self.compound_to_tolerances)):
            part = self.primary_to_part[primary]
            if tolerances[part] == 1:
                p_primary = self.primary_distn[primary]
                p_tolerances = 1.0
                for tolerance_class, tolerance_state in enumerate(tolerances):
                    if tolerance_class != part:
                        p_tolerances *= self.tolerance_distn[tolerance_state]
                self.compound_distn[i] = p_primary * p_tolerances

        # Check that the distributions have the correct normalization.
        # The loop is unrolled to better isolate errors.
        if not np.allclose(self.primary_distn.sum(), 1):
            raise Exception('internal error')
        if not np.allclose(self.tolerance_distn.sum(), 1):
            raise Exception('internal error')
        if not np.allclose(self.compound_distn.sum(), 1):
            raise Exception('internal error')

        # Define the compound transition rate matrix.
        # This is slow, but we do not need to be fast.
        shape = (self.ncompound, self.ncompound)
        self.Q_compound = np.zeros(shape, dtype=float)
        for i in range(self.ncompound):
            for j in range(self.ncompound):
                if i == j:
                    continue
                i_prim = self.compound_to_primary[i]
                j_prim = self.compound_to_primary[j]
                i_tols = self.compound_to_tolerances[i]
                j_tols = self.compound_to_tolerances[j]
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
                    if part == self.primary_to_part[i_prim]:
                        continue

                    # Add the transition rate.
                    if j_tol:
                        rate = self.rate_on
                    else:
                        rate = self.rate_off
                    self.Q_compound[i, j] = rate

                # Look for a primary state change.
                # Do not allow simultaneous primary and tolerance changes.
                # Do not allow a change to a non-tolerated primary class.
                # Do not allow transitions that have zero rate
                # in the primary process.
                if i_prim != j_prim:
                    if tol_hdist > 0:
                        continue
                    if not i_tols[self.primary_to_part[j_prim]]:
                        continue
                    if not self.Q_primary[i_prim, j_prim]:
                        continue
                    
                    # Add the primary state transition rate.
                    rate = self.Q_primary[i_prim, j_prim]
                    self.Q_compound[i, j] = rate

        self.Q_compound -= np.diag(self.Q_compound.sum(axis=1))


def get_tolerance_rate_matrix(rate_off, rate_on):
    """
    Get the two-state tolerance rate matrix.

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


def get_tolerance_expm_augmented_tree(T, root, Q_default=None):
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
    Q_default : 2d ndarray, optional
        Default rate matrix.

    Returns
    -------
    T_aug : weighted undirected networkx graph
        Tree annotated with transition probability matrices P.

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):

        # Get info about the edge.
        edge = T[na][nb]
        weight = edge['weight']
        Q = edge.get('Q', Q_default)
        _density.check_square_dense(Q)

        # Construct the transition probability matrix.
        P = np.empty_like(Q)
        pyfelscore.get_tolerance_rate_matrix(weight, Q, P)

        # Add the edge.
        T_aug.add_edge(na, nb, weight=weight, P=P)
    return T_aug


def get_expected_tolerance_history_statistics(
        T, node_to_allowed_states, root,
        root_distn=None):
    """
    This is copypasted from _mjp_dense.get_expected_history_statistics().

    It is specialized to 3-state tolerance rate matrices.

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Edges of this tree are annotated with weights
        and with 3-state tolerance rate matrices.
    node_to_allowed_states : dict
        Maps each node to a set of allowed states.
    root : integer
        Root node.
    root_distn : 1d ndarray, optional
        Distribution over states at the root.

    Returns
    -------
    dwell_times : 1d ndarray
        Expected dwell time on the tree, for each state.
        This does not depend on the root.
    posterior_root_distn : 1d ndarray
        Posterior distribution of states at the root.
    transitions : 2d ndarray
        A 2d ndarray that tracks the expected number of times
        each transition type appears in the history.
        The expectation depends on the root.
    absorption_expectation : float
        The total absorption expectation on the tree.

    """
    # Do some input validation for this restricted variant.
    if root not in T:
        raise ValueError('the specified root is not in the tree')

    # Construct the augmented tree by annotating each edge
    # with the appropriate state transition probability matrix.
    T_aug = get_tolerance_expm_augmented_tree(T, root)
    ntolerance_states = 3

    # Construct the node to pmap dict.
    #node_to_pmap = _mcy_dense.get_node_to_pmap(T_aug, root, ntolerance_states,
            #node_to_allowed_states=node_to_allowed_states)

    # Get the marginal state distribution for each node in the tree,
    # conditional on the known states.
    #node_to_distn = _mc0_dense.get_node_to_distn(
            #T_aug, root, node_to_pmap, ntolerance_states,
            #root_distn=root_distn)

    # For each edge in the tree, get the joint distribution
    # over the states at the endpoints of the edge.
    #T_joint = _mc0_dense.get_joint_endpoint_distn(
            #T_aug, root, node_to_pmap, node_to_distn, ntolerance_states)

    node_to_pmap, node_to_distn, edge_to_joint_distn = _mcy_dense.kitchen_sink(
            T_aug, root, ntolerance_states,
            node_to_allowed_states=node_to_allowed_states,
            root_distn=root_distn)

    T_joint = nx.Graph()
    for (na, nb), J in edge_to_joint_distn.items():
        edge_obj = T_aug[na][nb]
        T_joint.add_edge(na, nb, J=J)

    # Compute the expectations of the dwell times and the transition counts
    # by iterating over all edges and using the edge-specific
    # joint distribution of the states at the edge endpoints.
    expected_dwell_times = np.zeros(2, dtype=float)
    expected_transitions = np.zeros((2, 2), dtype=float)
    absorption_expectation = 0.0
    for na, nb in nx.bfs_edges(T, root):

        # Get the rate matrix to use for this edge.
        Q = T[na][nb].get('Q')
        _density.check_square_dense(Q)

        # Get the elapsed time along the edge.
        t = T[na][nb]['weight']

        # Get the conditional probability matrix associated with the edge.
        P = T_aug[na][nb]['P']
        _density.check_square_dense(P)

        # Get the joint probability matrix associated with the edge.
        J = T_joint[na][nb]['J']
        _density.check_square_dense(J)

        # Use Cython code to accumulate expectations along the branch.
        absorption_expectation += pyfelscore.get_tolerance_expectations(
                t, Q, P, J, expected_dwell_times, expected_transitions)

    # Return some expectations.
    dwell_expectation = expected_dwell_times
    init_expectation = node_to_distn[root]
    trans_expectation = expected_transitions
    return (
            dwell_expectation,
            init_expectation,
            trans_expectation,
            absorption_expectation)


def get_two_state_tolerance_distn(rate_off, rate_on):
    """

    Parameters
    ----------
    rate_off : float
        Rate of tolerance transition from on to off.
    rate_on : float
        Rate of tolerance transition from off to on.

    Returns
    -------
    tolerance_distn : dict
        Sparse distribution over the tolerance states 0 and 1.
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


def get_three_state_tolerance_distn(rate_off, rate_on):
    """

    Parameters
    ----------
    rate_off : float
        Rate of tolerance transition from on to off.
    rate_on : float
        Rate of tolerance transition from off to on.

    Returns
    -------
    tolerance_distn : dict
        Sparse distribution over the tolerance states 0 and 1.
        Tolerance state 0 is off, and tolerance state 1 is on.

    """
    if (rate_off < 0) or (rate_on < 0):
        raise ValueError('rates must be non-negative')
    total_tolerance_rate = rate_off + rate_on
    if total_tolerance_rate <= 0:
        raise ValueError('the total tolerance rate must be positive')
    unnormal_tolerance_distn = np.array([rate_off, rate_on, 0], dtype=float)
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
    tolerance_distn = get_tolerance_distn(rate_off, rate_on)

    # Get the root state and the transitions of the primary process.
    info = _mjp.get_history_root_state_and_transitions(T_primary, root=root)
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

    # Precompute edges.
    T_primary_edges = list(nx.bfs_edges(T_primary, root))

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
                T_primary_edges, tolerance_class)

        # Get the likelihood from the augmented tree and the root distribution.
        likelihood = _mjp.get_likelihood(
                T_tol, node_to_allowed_tolerances,
                root, root_distn=root_tolerance_prior, Q_default=None)

        # Contribute to the log likelihod.
        log_likelihood += np.log(likelihood)

    # Return the log likelihood for the entire process.
    return log_likelihood


def ll_expectation_helper(
        primary_to_part, rate_on, rate_off,
        Q_primary, primary_distn, T_primary_aug, root,
        disease_data=None):
    """
    Get contributions to the expected log likelihood of the compound process.

    The primary process trajectory is fully observed,
    but the binary tolerance states are unobserved.

    Parameters
    ----------
    primary_to_part : dict
        Maps the primary state to the tolerance class.
    rate_on : float
        Transition rate from tolerance state off to on.
    rate_off : float
        Transition rate from tolerance state on to off.
    Q_primary : 2d ndarray
        Primary rate matrix.
    primary_distn : 1d ndarray
        Primary process state distribution.
    T_primary_aug : x
        x
    root : integer
        The root node.
    disease_data : x
        x

    Returns
    -------
    cnll : instance of CompoundNegLL
        x

    """
    _density.check_square_dense(Q_primary)
    nprimary = Q_primary.shape[0]
    if primary_distn.shape[0] != nprimary:
        raise ValueError('inconsistency in the number of primary states')

    total_tree_length = T_primary_aug.size(weight='weight')
    primary_info = _mjp_dense.get_history_statistics(
            T_primary_aug, nprimary, root=root)
    dwell_times, root_state, transitions = primary_info
    post_root_distn = np.zeros(nprimary, dtype=float)
    post_root_distn[root_state] = 1

    neg_ll_info = _mjp_dense.differential_entropy_helper(
            Q_primary, primary_distn,
            post_root_distn, dwell_times, transitions)
    neg_init_prim_ll, neg_dwell_prim_ll, neg_trans_prim_ll = neg_ll_info

    tol_summary = get_tolerance_summary(
            primary_to_part, rate_on, rate_off,
            Q_primary, T_primary_aug, root,
            disease_data=disease_data)

    tol_info = get_tolerance_ll_contribs(
            rate_on, rate_off, total_tree_length, *tol_summary)
    init_tol_ll, dwell_tol_ll, trans_tol_ll = tol_info

    cnll = _tmjp_util.CompoundNegLL(
            neg_init_prim_ll, -init_tol_ll, -dwell_tol_ll,
            neg_trans_prim_ll, -trans_tol_ll)
    return cnll


#TODO copypasted from _tmjp
def differential_entropy_helper(
        ctm, post_root_distn, post_dwell_times, post_transitions):
    """
    More sophisticated than _mjp.differential_entropy_helper().

    The primary process is assumed to be time-reversible,
    and the prior distribution at the root is assumed to be
    the stationary distribution of the compound process.

    Parameters
    ----------
    ctm : instance of _tmjp_dense.CompoundToleranceModel
        Compound tolerance model.
    post_root_distn : 1d ndarray
        Compound process posterior state distribution at the root.
    post_dwell_times : 1d ndarray
        Compound process posterior expected dwell time for each state.
    post_transitions : 2d ndarray
        Compound process posterior expected count of each transition type.

    Returns
    -------
    cnll : instance of CompoundNegLL
        Negative log likelihood of the trajectory, additively separated.

    """
    # Get the non-separated differential entropy components for comparison.
    # The dwell time component will be used directly,
    # without further separation.
    info = _mjp_dense.differential_entropy_helper(
            ctm.Q_compound, ctm.compound_distn,
            post_root_distn, post_dwell_times, post_transitions)
    diff_ent_init, diff_ent_dwell, diff_ent_trans = info

    # Contribution of the initial primary state.
    diff_ent_init_prim = 0.0
    for compound_state, post_prob in enumerate(post_root_distn):
        if not post_prob:
            continue
        primary_state = ctm.compound_to_primary[compound_state]
        prior_prob = ctm.primary_distn[primary_state]
        diff_ent_init_prim -= special.xlogy(post_prob, prior_prob)

    # Contribution of the initial tolerance states.
    # This is careful about incompatibility between primary state
    # and tolerance states; it gives an infinity negative log likelihood
    # to the case where the tolerance class of the primary state
    # is untolerated.
    diff_ent_init_tol = 0.0
    for compound_state, post_prob in enumerate(post_root_distn):
        if not post_prob:
            continue
        primary_state = ctm.compound_to_primary[compound_state]
        primary_part = ctm.primary_to_part[primary_state]
        tol_states = ctm.compound_to_tolerances[compound_state]
        if not tol_states[primary_part]:
            print('primary fail')
            diff_ent_init_tol = np.inf
        tol_count = sum(1 for tol_state in tol_states if tol_state)
        untol_count = sum(1 for tol_state in tol_states if not tol_state)
        if tol_count:
            prior_prob = ctm.tolerance_distn[1]
            if prior_prob:
                diff_ent_init_tol -= special.xlogy(
                        post_prob * (tol_count-1), prior_prob)
            else:
                print('tol_count fail')
                diff_ent_init_tol = np.inf
        if untol_count:
            prior_prob = ctm.tolerance_distn[0]
            if prior_prob:
                diff_ent_init_tol -= special.xlogy(
                        post_prob * untol_count, prior_prob)
            else:
                print('untol_count fail')
                diff_ent_init_tol = np.inf

    # Check the initial state contribution to the differential entropy
    if not np.allclose(
            diff_ent_init_prim + diff_ent_init_tol,
            diff_ent_init):
        raise Exception(
                'internal differential entropy calculation error: '
                '%s + %s = %s but expected %s' % (
                    diff_ent_init_prim, diff_ent_init_tol,
                    diff_ent_init_prim + diff_ent_init_tol,
                    diff_ent_init))

    # Transition contribution to differential entropy.
    diff_ent_trans_prim = 0.0
    diff_ent_trans_tol = 0.0
    for sa in range(ctm.ncompound):
        for sb in range(ctm.ncompound):
            rate = ctm.Q_compound[sa, sb]
            ntrans_expected = post_transitions[sa, sb]
            sa_primary = ctm.compound_to_primary[sa]
            sb_primary = ctm.compound_to_primary[sb]
            if sa_primary != sb_primary:
                diff_ent_trans_prim -= special.xlogy(ntrans_expected, rate)
            else:
                diff_ent_trans_tol -= special.xlogy(ntrans_expected, rate)

    # Check the transition contribution to the differential entropy
    if not np.allclose(
            diff_ent_trans_prim + diff_ent_trans_tol,
            diff_ent_trans):
        raise Exception('internal error')

    # Aggregate and return the contributions to the differential entropy.
    cnll = _tmjp_util.CompoundNegLL(
            diff_ent_init_prim, diff_ent_init_tol, diff_ent_dwell,
            diff_ent_trans_prim, diff_ent_trans_tol)
    return cnll


#TODO disease_data argument is untested
def get_tolerance_summary(
        primary_to_part, rate_on, rate_off,
        Q_primary, T_primary, root,
        disease_data=None):
    """
    Get tolerance process expectations conditional on a primary trajectory.

    Given a primary process trajectory,
    compute some log likelihood contributions
    related to the tolerance process.

    Parameters
    ----------
    primary_to_part : dict
        Map from the primary state to the tolerance class.
    rate_on : float
        Transition rate from tolerance state off to on.
    rate_off : float
        Transition rate from tolerance state on to off.
    Q_primary : 2d ndarray
        Primary rate matrix.
    T_primary : x
        x
    root : integer
        The root node.
    disease_data : dict, optional
        For each tolerance class,
        a map from a node to a set of allowed tolerance states.

    Returns
    -------
    expected_initial_on : float
        x
    expected_initial_off : float
        x
    expected_dwell_on : float
        x
    expected_dwell_off : float
        x
    expected_nabsorptions : float
        x
    expected_ngains : float
        x
    expected_nlosses : float
        x

    Notes
    -----
    This function assumes that no tolerance process data is observed
    at the leaves, other than what could be inferred through
    the primary process observations at the leaves.
    The ordering of the arguments of this function was chosen haphazardly.

    """
    # Summarize the tolerance process.
    total_weight = T_primary.size(weight='weight')
    nparts = len(set(primary_to_part.values()))
    tolerance_distn = get_three_state_tolerance_distn(rate_off, rate_on)

    # Precompute edges.
    T_primary_edges = list(nx.bfs_edges(T_primary, root))

    # Compute conditional expectations of statistics
    # of the tolerance process.
    # This requires constructing independent piecewise homogeneous
    # Markov jump processes for each tolerance class.
    #
    # expected_nabsorptions is an expectation that connects
    # the blinking process to the primary process.
    # This is the expected number of times that
    # a non-forbidden primary process "absorption"
    # into the current blinking state
    # would have been expected to occur.
    expected_ngains = 0.0
    expected_nlosses = 0.0
    expected_dwell_on = 0.0
    expected_initial_on = 0.0
    expected_nabsorptions = 0.0
    for tolerance_class in range(nparts):

        # Get the restricted inhomogeneous Markov jump process
        # associated with the tolerance class,
        # conditional on the trajectory of the primary state.
        T_tol, node_to_allowed_tolerances = get_inhomogeneous_mjp(
                primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
                T_primary_edges, tolerance_class)

        # Further restrict the node_to_allowed_tolerances using disease data.
        if disease_data is not None:

            # Get the tolerance-class-specific map from a node to
            # a set of allowed tolerance states.
            # Then update the node_to_allowed_tolerances according
            # to the intersection of this tolerance state set.
            node_to_tol_set = disease_data[tolerance_class]
            for node, tol_set in node_to_tol_set.items():
                node_to_allowed_tolerances[node].intersection_update(tol_set)

        # Compute conditional expectations of dwell times
        # and transitions for this tolerance class.
        expectation_info = get_expected_tolerance_history_statistics(
                T_tol, node_to_allowed_tolerances, root,
                root_distn=tolerance_distn)
        dwell_times = expectation_info[0]
        post_root_distn = expectation_info[1]
        transitions = expectation_info[2]
        nabsorptions = expectation_info[3]

        # Get the dwell time expectation contribution.
        expected_dwell_on += dwell_times[1]

        # Get the transition expectation contribution.
        expected_ngains += transitions[0, 1]
        expected_nlosses += transitions[1, 0]

        # Get the initial state expectation contribution.
        expected_initial_on += post_root_distn[1]

        # Get the absorption expectation contribution.
        expected_nabsorptions += nabsorptions

    # Summarize expectations.
    expected_initial_off = nparts - expected_initial_on
    expected_dwell_off = total_weight * nparts - expected_dwell_on

    # Return expectations.
    ret = (
            expected_initial_on, expected_initial_off,
            expected_dwell_on, expected_dwell_off,
            expected_nabsorptions,
            expected_ngains, expected_nlosses)
    return ret


def get_tolerance_ll_contribs(
        rate_on, rate_off, total_tree_length,
        expected_initial_on, expected_initial_off,
        expected_dwell_on, expected_dwell_off,
        expected_nabsorptions,
        expected_ngains, expected_nlosses,
        ):
    """
    Tolerance process log likelihood contributions.

    Note that the contributions associated with dwell times
    subsume the primary process dwell time log likelihood contributions.
    The first group of args defines parameters of the process.
    The second group defines the posterior tolerance distribution at the root.
    The third group defines the posterior tolerance dwell times.
    The fourth group is just a virtual posterior nabsorptions count which
    is related to dwell times of the primary process.
    The fifth group defines posterior transition expectations.

    Parameters
    ----------
    rate_on : float
        Transition rate from tolerance state off to on.
    rate_off : float
        Transition rate from tolerance state on to off.
    total_tree_length : float
        x
    expected_initial_on : float
        x
    expected_initial_off : float
        x
    expected_dwell_on : float
        x
    expected_dwell_off : float
        x
    expected_nabsorptions : float
        x
    expected_ngains : float
        x
    expected_nlosses : float
        x

    Returns
    -------
    init_ll_contrib : float
        x
    dwell_ll_contrib : float
        x
    trans_ll_contrib : float
        x

    """
    tolerance_distn = get_three_state_tolerance_distn(rate_off, rate_on)
    init_ll_contrib = (
            special.xlogy(expected_initial_on - 1, tolerance_distn[1]) +
            special.xlogy(expected_initial_off, tolerance_distn[0]))
    dwell_ll_contrib = -(
            expected_dwell_off * rate_on +
            (expected_dwell_on - total_tree_length) * rate_off +
            expected_nabsorptions)
    trans_ll_contrib = (
            special.xlogy(expected_ngains, rate_on) +
            special.xlogy(expected_nlosses, rate_off))
    return init_ll_contrib, dwell_ll_contrib, trans_ll_contrib


def get_primary_state_to_absorption_rate(
        Q_primary, primary_to_part, tolerance_class):
    """
    Define the absorption rate for each primary state.

    This is a helper function for speed profiling get_inhomogeneous_mjp().

    Parameters
    ----------
    Q_primary : 2d ndarray
        Primary process rate matrix.
    primary_to_part : dict
        Map from primary state to tolerance class.
    tolerance_class : integer
        The tolerance class under consideration.

    Returns
    -------
    primary_state_to_absorption_rate : dict
        Map from primary state to absorption rate.

    """
    _density.check_square_dense(Q_primary)
    nprimary = len(primary_to_part)
    primary_state_to_absorption_rate = {}
    for sa in range(nprimary):
        absorption_rate = 0
        for sb in range(nprimary):
            if sb != sa:
                if primary_to_part[sb] == tolerance_class:
                    rate = Q_primary[sa, sb]
                    absorption_rate += rate
        primary_state_to_absorption_rate[sa] = absorption_rate
    return primary_state_to_absorption_rate


def get_inhomogeneous_mjp(
        primary_to_part, rate_on, rate_off, Q_primary, T_primary, root,
        T_primary_edges, tolerance_class):
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
    Q_primary : 2d ndarray
        Primary process rate matrix.
    T_primary : weighted undirected networkx graph
        Primary process trajectory.
    root : integer
        Root node
    T_primary_edges : sequence of directed edges as node pairs
        Edges directed away from the root.
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
    _density.check_square_dense(Q_primary)
    nprimary = len(primary_to_part)

    # Define the absorption rate for each primary state.
    primary_state_to_absorption_rate = get_primary_state_to_absorption_rate(
        Q_primary, primary_to_part, tolerance_class)

    # Define the set of allowed tolerances at each node.
    # These may be further constrained
    # by the sampled primary process trajectory.
    node_to_allowed_tolerances = dict((n, {0, 1}) for n in T_primary)

    # Construct the tree whose edges are in correspondence
    # to the edges of the sampled primary trajectory,
    # and whose edges are annotated with weights
    # and with edge-specific 3-state transition rate matrices.
    # The third state of each edge-specific rate matrix is an
    # absorbing state which will never be entered.
    T_tol = nx.Graph()
    for na, nb in T_primary_edges:
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
        absorption_rate = primary_state_to_absorption_rate[primary_state]

        # Construct the local tolerance rate matrix.
        Q_tol = np.zeros((3, 3), dtype=float)
        Q_tol[0, 1] = local_rate_on
        Q_tol[1, 0] = local_rate_off
        Q_tol[1, 2] = absorption_rate
        Q_tol = Q_tol - np.diag(Q_tol.sum(axis=1))

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
    Q_primary : 2d ndarray.
        x
    primary_to_part : dict
        Maps the primary state to its tolerance class.
    tolerance_distn : 1d ndarray
        The dense distribution over tolerance states 0 and 1.

    Returns
    -------
    Q_primary_proposal : 2d ndarray
        A transition rate matrix.
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
    _density.check_square_dense(Q_primary)
    nprimary = len(primary_to_part)

    # Adjust the proposal rate matrix
    # by reducing rates that cross tolerance classes.
    Q_proposal = Q_primary.copy()
    for sa in range(nprimary):
        for sb in range(nprimary):
            if primary_to_part[sa] != primary_to_part[sb]:
                Q_proposal[sa, sb] *= tolerance_distn[1]
    Q_proposal -= np.diag(Q_proposal.sum(axis=1))
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

