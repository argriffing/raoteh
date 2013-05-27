"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
For continuous time processes (as opposed to discrete time processes)
see the Markov jump process module.
Everything related to hold times, dwell times, instantaneous rates,
total rates, and edges or lengths or expectations associated with edges
is out of the scope of this module.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict

import numpy as np
import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip)


__all__ = []


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

