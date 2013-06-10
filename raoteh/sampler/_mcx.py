"""
Functions related to a Markov process in discrete time and space.

This module assumes a rooted tree-shaped dependence structure.
Transition probability matrices may be provided per edge,
and a default transition matrix may be provided which will be used
if an edge-specific transition matrix is not available.
The name of this module
is derived from "m"arkov "c"hain observation type "x",
where the observation type x refers to the least abstract of three
ways to deal with partial observations of the process state.
Type x uses a sparse map from nodes in the network
to their corresponding observed state if any.
Nodes missing from the map are assumed to have completely unobserved state,
while nodes in the map are assumed to have completely observed state.

Here are some more notes regarding the argument node_to_state.
Nodes in this map are assumed to have completely known state.
Nodes not in this map are assumed to have completely missing state.
If this map is not provided,
all states information will be assumed to be completely missing.
Entries of this dict that correspond to nodes not in the tree
will be silently ignored.

"""
from __future__ import division, print_function, absolute_import

from collections import defaultdict
import itertools

import numpy as np
import networkx as nx

from raoteh.sampler import _mc0

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn,
        )


__all__ = []


def get_node_to_pset(T, root, node_to_state=None, P_default=None):
    """
    For each node, get the set of states that give positive subtree likelihood.

    This function is analogous to get_node_to_pmap.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.

    Returns
    -------
    node_to_pset : dict
        A map from a node to the set of states with positive subtree likelihood.

    """
    # Input validation.
    if len(set(T)) < 2:
        raise ValueError('expected at least two nodes in the tree')

    # Bookkeeping.
    successors = nx.dfs_successors(T, root)
    predecessors = nx.dfs_predecessors(T, root)

    # Compute the map from node to set.
    node_to_pset = {}
    for nb in nx.dfs_postorder_nodes(T, root):

        # If a parent node is available, get a set of states
        # involved in the transition matrix associated with the parent edge.
        # A more complicated implementation would use only the sink
        # states of that transition matrix.
        na_set = None
        if nb in predecessors:
            na = predecessors[nb]
            P = T[na][nb].get('P', P_default)
            na_set = set(P)

        # If the state of the current state is known,
        # define the set containing only that state.
        nb_set = None
        if nb in node_to_state:
            nb_set = {node_to_state[nb]}

        # If a child node is available, get the set of states
        # that have transition to child states
        # for which the child subtree likelihoods are positive.
        nc_set = None
        if nb in successors:
            for nc in successors[nb]:
                allowed_set = set()
                P = T[nb][nc].get('P', P_default)
                for sb, sc in P.edges():
                    if sc in node_to_pset[nc]:
                        allowed_set.add(sb)
                if nc_set is None:
                    nc_set = allowed_set
                else:
                    nc_set.intersection_update(allowed_set)

        # Take the intersection of informative constraints due to
        # possible parent transitions,
        # possible direct constraints on the node state,
        # and possible child node state constraints.
        pset = None
        for constraint_set in (na_set, nb_set, nc_set):
            if constraint_set is not None:
                if pset is None:
                    pset = constraint_set
                else:
                    pset.intersection_update(constraint_set)

        # This value should not be None unless there has been some problem.
        if pset is None:
            raise ValueError('internal error')

        # Define the pset for the node.
        node_to_pset[nb] = pset

    # Return the node_to_pset map.
    return node_to_pset


def get_node_to_pmap(T, root, node_to_state=None, P_default=None):
    """
    For each node, construct the map from state to subtree likelihood.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.

    Returns
    -------
    node_to_pmap : dict
        A map from a node to a map from a state to a subtree likelihood.

    """
    # Get the possible states for each node,
    # after accounting for the rooted tree shape
    # and the edge-specific transition matrix sparsity patterns
    # and the observed states.
    #node_to_smap = get_node_to_smap(T, root,
            #node_to_state=node_to_state, P_default=P_default)
    #node_to_state_set = get_node_to_state_set(T, root,
            #node_to_smap, node_to_state=node_to_state)
    node_to_pset = get_node_to_pset(T, root,
            node_to_state=node_to_state, P_default=P_default)
    node_to_state_set = _mc0.get_node_to_set(T, root,
            node_to_pset, P_default=P_default)

    # Check the node to state set for consistency.
    # Either the likelihood is structurally positive
    # in which case all nodes should have possible states,
    # or likelihood is structurally zero
    # in which case all nodes should have no possible states.
    npos = sum(1 for k, v in node_to_state_set.items() if v)
    nneg = sum(1 for k, v in node_to_state_set.items() if not v)
    if npos and nneg:
        print()
        print(node_to_pset)
        print(node_to_state_set)
        raise ValueError('internal error')

    # Bookkeeping.
    successors = nx.dfs_successors(T, root)

    # For each node, get a sparse map from state to subtree likelihood.
    node_to_pmap = {}
    for node in nx.dfs_postorder_nodes(T, root):

        # Build the pmap.
        pmap = {}
        for node_state in node_to_state_set[node]:

            # Add the subtree likelihood to the node state pmap.
            cprob = 1.0
            for n in successors.get(node, []):
                P = T[node][n].get('P', P_default)
                nprob = 0.0
                allowed_states = set(P[node_state]) & set(node_to_pmap[n])
                if not allowed_states:
                    print()
                    print(node_to_pset)
                    print(node_to_state_set)
                    raise ValueError('internal error')
                for s in allowed_states:
                    a = P[node_state][s]['weight']
                    b = node_to_pmap[n][s]
                    nprob += a * b
                cprob *= nprob
            pmap[node_state] = cprob

        # Add the map from state to subtree likelihood.
        node_to_pmap[node] = pmap

    # Return the map from node to the map from state to subtree likelihood.
    return node_to_pmap


def get_likelihood(T, root,
        node_to_state=None, root_distn=None, P_default=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        A tree whose edges are annotated with transition matrices P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state if any.
        Nodes in this map are assumed to have completely known state.
        Nodes not in this map are assumed to have completely missing state.
        If this map is not provided,
        all states information will be assumed to be completely missing.
        Entries of this dict that correspond to nodes not in the tree
        will be silently ignored.
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
    node_to_pmap = get_node_to_pmap(T, root,
            node_to_state=node_to_state, P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Return the likelihood.
    return _mc0.get_likelihood(root_pmap, root_distn=root_distn)


#TODO under construction
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

