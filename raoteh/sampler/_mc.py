"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
For continuous time processes (as opposed to discrete time processes)
use the Markov jump process module instead.
Everything related to hold times, dwell times, instantaneous rates,
total rates, and edges or lengths or expectations associated with edges
is out of the scope of this module.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn)

from raoteh.sampler import _mc0, _mcx


__all__ = []


def construct_node_to_restricted_pmap(
        T, root, node_to_allowed_states=None,
        P_default=None, states_default=None):
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
    node_to_allowed_states : dict, optional
        A map from a node to a set of allowed states.
    P_default : directed weighted acyclic networkx graph, optional
        A default transition matrix.
    states_default : dict, optional
        Nodes that are not in the node_to_allowed_states map
        will use this default set of allowed states.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    # Input validation.
    if (node_to_allowed_states is None) and (states_default is None):
        raise ValueError('expected either a map from nodes to allowed states '
                'or a set of default allowed states or both')

    # Bookkeeping.
    if root not in T:
        raise ValueError('the specified root node is not in the tree')
    successors = nx.dfs_successors(T, root)

    # For each node, get a sparse map from state to subtree probability.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):

        # Get the set of valid states at this node.
        if node_to_allowed_states is None:
            valid_node_states = states_default
        else:
            valid_node_states = node_to_allowed_states.get(node, states_default)
        if valid_node_states is None:
            raise ValueError(
                    'the set of valid states is undefined at this node '
                    '(note that this is not the same as a defined but '
                    'empty set of valid states)')

        if node not in successors:
            node_to_pmap[node] = dict((s, 1.0) for s in valid_node_states)
        else:
            pmap = {}
            for node_state in valid_node_states:

                # Check for a structural subtree failure given this node state.
                structural_failure = False
                for n in successors[node]:

                    # Define the transition matrix according to the edge.
                    P = T[node][n].get('P', P_default)
                    if P is None:
                        raise ValueError('one of the edges is not annotated '
                                'with a transition matrix, and no default '
                                'transition matrix has been provided')

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

                # If there is no structural failure or error,
                # then add the subtree probability to the node state pmap.
                if not structural_failure:
                    cprob = 1.0
                    for n in successors[node]:
                        P = T[node][n].get('P', P_default)
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


def get_restricted_likelihood(T, root, node_to_allowed_states,
        root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    This is a general likelihood calculator for
    a Markov chain on tree-structured domains.
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
    root_distn : dict, optional
        A sparse finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.
    P_default : directed weighted networkx graph, optional
        If an edge is not annotated with a transition matrix P,
        then this default transition matrix will be used.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    # Get likelihoods conditional on the root state.
    node_to_pmap = construct_node_to_restricted_pmap(
            T, root, node_to_allowed_states, P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Return the likelihood.
    return _mc0.get_likelihood(root_pmap, root_distn=root_distn)


#XXX this is kind of obsolete by the new get_normalized_dict_distn interface
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
    d = dict((s, prior_distn[s] * pmap[s]) for s in feasible_states)
    return get_normalized_dict_distn(d)


def get_node_to_distn_naive(T, node_to_allowed_states,
        root, prior_root_distn, P_default=None):
    """
    Get marginal state distributions at nodes in a tree.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    node_to_allowed_states : dict
        Map from node to collection of allowed states.
    root : integer
        Root node.
    prior_root_distn : dict
        A finite distribution over root states.
    P_default : weighted directed networkx graph, optional
        A default universal probability transition matrix.

    Returns
    -------
    node_to_distn : dict
        Maps each node to a distribution over states.

    See also
    --------
    get_node_to_distn

    """
    nodes, allowed_states = zip(*node_to_allowed_states.items())
    node_to_state_to_weight = dict((n, defaultdict(float)) for n in nodes)
    for assignment in itertools.product(*allowed_states):

        # Get the map corresponding to the assignment.
        node_to_state = dict(zip(nodes, assignment))

        # Compute the log likelihood for the assignment.
        # If the log likelihood cannot be computed,
        # then skip to the next state assignment.
        try:
            ll = _mc0.get_history_log_likelihood(T, root, node_to_state,
                    root_distn=prior_root_distn, P_default=P_default)
        except StructuralZeroProb as e:
            continue

        # Add the likelihood to weights of states assigned to nodes.
        likelihood = np.exp(ll)
        for node, state in zip(nodes, assignment):
            d = node_to_state_to_weight[node]
            d[state] += likelihood

    # For each node, normalize the distribution over states.
    node_to_distn = {}
    for node in nodes:
        d = node_to_state_to_weight[node]
        if not d:
            raise ValueError('for one of the nodes, no state was observed')
        total_weight = sum(d.values())
        if not total_weight:
            raise NumericalZeroProb(
                    'for one of the nodes, '
                    'the sum of weights of observed states is zero')
        distn = dict((n, w / total_weight) for n, w in d.items())
        node_to_distn[node] = distn

    # Return the map from node to distribution.
    return node_to_distn


#TODO under destruction
def xxx_get_node_to_distn(T, node_to_allowed_states, node_to_pmap,
        root, prior_root_distn=None, P_default=None):
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
    node_to_allowed_states : dict
        A map from a node to a set of allowed states.
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
    # Bookkeeping.
    predecessors = nx.dfs_predecessors(T, root)

    # Get the distributions.
    node_to_distn = {}
    for node in nx.dfs_preorder_nodes(T, root):

        # Get the map from state to subtree likelihood.
        pmap = node_to_pmap[node]

        # Compute the prior distribution at the root separately.
        # If the prior distribution is not provided,
        # then treat it as uninformative.
        if node == root:
            if prior_root_distn is None:
                if not pmap:
                    raise StructuralZeroProb('no root state is feasible')
                total_weight = sum(pmap.values())
                if not total_weight:
                    raise NumericalZeroProb('numerical zero probability error')
                distn = dict((k, v / total_weight) for k, v in pmap.items())
            else:
                distn = get_zero_step_posterior_distn(prior_root_distn, pmap)
        else:
            parent_node = predecessors[node]
            parent_distn = node_to_distn[parent_node]

            # Get the transition matrix associated with this edge.
            P = T[parent_node][node].get('P', P_default)
            if P is None:
                raise ValueError('no transition matrix is available')

            # For each parent state,
            # get the distribution over child states;
            # this distribution will include both the P matrix
            # and the pmap of the child node.
            distn = defaultdict(float)
            for sa, pa in parent_distn.items():

                # Construct the conditional transition probabilities.
                feasible_sb = set(P[sa]) & set(node_to_pmap[node])
                sb_weights = {}
                for sb in feasible_sb:
                    a = P[sa][sb]['weight']
                    b = node_to_pmap[node][sb]
                    sb_weights[sb] = a*b
                tot = np.sum(sb_weights.values())
                sb_distn = dict((sb, w / tot) for sb, w in sb_weights.items())

                # Add to the marginal distn.
                for sb, pb in sb_distn.items():
                    distn[sb] += pa * pb

        # Set the node_to_distn.
        node_to_distn[node] = distn

    # Return the marginal state distributions at nodes.
    return node_to_distn

