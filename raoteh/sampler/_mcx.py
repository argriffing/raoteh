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

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn,
        )

from raoteh.sampler import _mc0


__all__ = []


# TODO maybe delete this function
def validate_root(T, root):
    """
    Assert that the root is a node in the tree.

    Parameters
    ----------
    T : undirected networkx graph
        The tree.
    root : integer
        The root node.

    """
    if root is None:
        raise ValueError('a root must be provided')
    if root not in T:
        raise ValueError('the root must be a node in the tree')


def get_node_to_state_set(T, root, node_to_smap, node_to_state=None):
    """
    Get a map from each node to a set of valid states.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree.
    root : integer
        The root node.
    node_to_smap : dict
        A map from each non-root node to a sparse map from a parent state
        to a nonempty set of valid states.
    node_to_state : dict, optional
        A sparse map from a node to its known state.

    Returns
    -------
    node_to_state_set : dict
        A map from each node to a set of valid states.

    Notes
    -----
    The node_to_state argument is used only for the root node.

    """
    # Build the map from node to state set.
    node_to_state_set = {}
    predecessors = nx.dfs_predecessors(T, root)
    successors = nx.dfs_successors(T, root)
    for nb in nx.dfs_preorder_nodes(T, root):

        # Initialize the set of allowed states.
        allowed_states = None

        # Get the state constraint induced by the parent state set.
        if nb in predecessors:
            smap = node_to_smap[nb]
            allowed_parent_states = node_to_state_set[predecessors[nb]]
            allowed_states = set()
            for sa in allowed_parent_states:
                if sa in smap:
                    allowed_states.update(smap[sa])

        # Apply the root constraint if available.
        if nb == root:
            if (node_to_state is not None) and (root in node_to_state):
                root_state = node_to_state[root]
                if allowed_states is None:
                    allowed_states = {root_state}
                else:
                    allowed_states.intersection_update({root_state})

        # Get the state constraint induced by child state set constraints.
        if nb in successors:
            for nc in successors[nb]:
                constraint = set(node_to_smap[nc])
                if allowed_states is None:
                    allowed_states = constraint
                else:
                    allowed_states.intersection_update(constraint)

        # Define the allowed state set.
        if allowed_states is None:
            raise ValueError('this node has no predecessors or successors')
        node_to_state_set[nb] = allowed_states

    # Return the map from node to state set.
    return node_to_state_set


def get_node_to_smap(T, root, node_to_state=None, P_default=None):
    """
    Get a map from each non-root node to a map from parent state to a state set.

    Transition matrices are used only through their sparsity pattern.

    Parameters
    ----------
    T : undirected unweighted acyclic networkx graph
        A tree whose edges are optionally annotated
        with edge-specific state transition probability matrix P.
    root : integer
        The root node.
    node_to_state : dict, optional
        A sparse map from a node to its known state.
    P_default : networkx directed weighted graph, optional
        Sparse transition matrix to be used for edges
        which are not annotated with an edge-specific transition matrix.

    Returns
    -------
    node_to_smap : dict
        A map from each non-root node to a sparse map from a parent state
        to a nonempty set of valid states.

    """
    # Precompute the successors of each node in the tree.
    successors = nx.dfs_successors(T, root)

    # Compute the map from node to smap.
    node_to_smap = {}
    for na, nb in reversed(list(nx.bfs_edges(T, root))):
        edge = T[na][nb]
        P = edge.get('P', P_default)
        if P is None:
            raise ValueError('no transition matrix is available for this edge')
        smap = {}
        for sa in set(P):

            # Define the prior set of states.
            prior_sb_set = set()
            if (node_to_state is not None) and (nb in node_to_state):
                sb = node_to_state[nb]
                if P.has_edge(sa, sb):
                    prior_sb_set.add(sb)
            else:
                prior_sb_set.update(P[sa])

            # The posterior state set uses subtree information.
            posterior_sb_set = set()
            for sb in prior_sb_set:
                if all(sb in node_to_smap[nc] for nc in successors.get(nb, [])):
                    posterior_sb_set.add(sb)

            # The map from the parent state will be sparse.
            # If a parent state is not included in the map,
            # then it means that no states are possible
            # given that parent state.
            if posterior_sb_set:
                smap[sa] = posterior_sb_set
        node_to_smap[nb] = smap
    return node_to_smap


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
    # Do some input validation.
    validate_root(T, root)

    # Get the possible states for each node,
    # after accounting for the rooted tree shape
    # and the edge-specific transition matrix sparsity patterns
    # and the observed states.
    node_to_smap = get_node_to_smap(T, root,
            node_to_state=node_to_state, P_default=P_default)
    node_to_state_set = get_node_to_state_set(T, root,
            node_to_smap, node_to_state=node_to_state)

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
                    print('fail xxx')
                    print('node to state set', node_to_state_set)
                    print('transition states', set(P[node_state]))
                    print('pmap states', set(node_to_pmap[n]))
                    print()
                    raise ValueError(
                            'internal error: ' + str((node, node_state)) +
                            ' -> ' + str(n))
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


# XXX depends on node_to_state and P_default only through root_pmap
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
    # Check whether the prior by itself causes the likelihood to be zero.
    if (root_distn is not None) and not root_distn:
        raise StructuralZeroProb('no root state has nonzero prior likelihood')

    # Get likelihoods conditional on the root state.
    node_to_pmap = get_node_to_pmap(T, root,
            node_to_state=node_to_state, P_default=P_default)
    root_pmap = node_to_pmap[root]

    # Check whether the likelihoods at the root, by themselves,
    # cause the likelihood to be zero.
    if not root_pmap:
        raise StructuralZeroProb(
                'all root states give a subtree likelihood of zero')

    # Construct the set of possible root states.
    # If no root state is possible raise the exception indicating
    # that the likelihood is zero by sparsity.
    feasible_rstates = set(root_pmap)
    if root_distn is not None:
        feasible_rstates.intersection_update(set(root_distn))
    if not feasible_rstates:
        raise StructuralZeroProb(
                'all root states have either zero prior likelihood '
                'or give a subtree likelihood of zero')

    # Compute the likelihood.
    if root_distn is not None:
        likelihood = sum(root_distn[s] * root_pmap[s] for s in feasible_rstates)
    else:
        likelihood = sum(root_pmap[s].values())

    # Return the likelihood.
    return likelihood


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


def get_history_log_likelihood(T, node_to_state, root, root_distn,
        P_default=None):
    """
    Compute the log likelihood for a fully augmented history.

    Parameters
    ----------
    T : undirected acyclic networkx graph
        Tree annotated with transition matrices.
    node_to_state : dict
        Each node in the tree is mapped to an integer state.
    root : integer
        Root node.
    root_distn : dict
        Sparse prior distribution over states at the root.
    P_default : weighted directed networkx graph, optional
        A default universal probability transition matrix.

    Returns
    -------
    log_likelihood : float
        The log likelihood of the fully augmented history.

    """
    # Check that the set of nodes for which the state is available
    # exactly matches the set of nodes in the tree.
    if set(T) != set(node_to_state):
        raise ValueError(
                'the set of nodes with known states in the history '
                'should exactly match the set of known states on the tree')

    # Check the root state.
    root_state = node_to_state[root]
    if root_state not in root_distn:
        raise StructuralZeroProb(
                'the prior state distribution at the root '
                'does not include the root state in this history')

    # Initialize the log likelihood.
    log_likelihood = 0.0

    # Add the log likelihood contribution from the root.
    log_likelihood += np.log(root_distn[root_state])

    # Add the log likelihood contribution from state transitions.
    for na, nb in nx.bfs_edges(T, root):
        edge = T[na][nb]
        P = edge.get('P', P_default)
        if P is None:
            raise ValueError('undefined transition matrix on this edge')
        sa = node_to_state[na]
        sb = node_to_state[nb]
        if not P.has_edge(sa, sb):
            raise StructuralZeroProb(
                    'the states of the endpoints of an edge '
                    'are incompatible with the transition matrix on the edge')
        p = P[sa][sb]['weight']
        log_likelihood += np.log(p)

    # Return the log likelihood.
    return log_likelihood


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
            ll = get_history_log_likelihood(
                    T, node_to_state, root, prior_root_distn, P_default)
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


def get_node_to_distn(T, node_to_allowed_states, node_to_pmap,
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


def get_joint_endpoint_distn_naive(T, node_to_allowed_states,
        root, prior_root_distn, P_default=None):
    """

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
    T_aug : undirected networkx graph
        A tree whose edges are annotated with sparse joint endpoint
        state distributions as networkx digraphs.
        These annotations use the attribute 'J' which
        is supposed to mean 'joint' and which is writeen in
        single letter caps reminiscent of matrix notation.

    See also
    --------
    get_joint_endpoint_distn

    """
    T_aug = nx.Graph()
    for na, nb in nx.bfs_edges(T, root):
        T_aug.add_edge(na, nb, J=nx.DiGraph())
    nodes, allowed_states = zip(*node_to_allowed_states.items())
    for assignment in itertools.product(*allowed_states):

        # Get the map corresponding to the assignment.
        node_to_state = dict(zip(nodes, assignment))

        # Compute the log likelihood for the assignment.
        # If the log likelihood cannot be computed,
        # then skip to the next state assignment.
        try:
            ll = get_history_log_likelihood(
                    T, node_to_state, root, prior_root_distn, P_default)
        except StructuralZeroProb as e:
            continue

        # Add the likelihood to weights of ordered node pairs on edges.
        likelihood = np.exp(ll)
        for na, nb in nx.bfs_edges(T, root):
            J = T_aug[na][nb]['J']
            sa = node_to_state[na]
            sb = node_to_state[nb]
            if not J.has_edge(sa, sb):
                J.add_edge(sa, sb, weight=0.0)
            J[sa][sb]['weight'] += likelihood

    # For each edge, normalize the distribution over ordered state pairs.
    for na, nb in nx.bfs_edges(T, root):
        J = T_aug[na][nb]['J']
        weights = []
        for sa, sb in J.edges():
            weights.append(J[sa][sb]['weight'])
        if not weights:
            raise Exception('internal error')
        total_weight = np.sum(weights)
        for sa, sb in J.edges():
            J[sa][sb]['weight'] /= total_weight
    
    # Return the tree with the sparse joint distributions on edges.
    return T_aug


def get_joint_endpoint_distn(T, node_to_pmap, node_to_distn, root):
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
        Root node.

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
        weighted_edges = []
        for sa, pa in node_to_distn[na].items():

            # Construct the conditional transition probabilities.
            feasible_sb = set(P[sa]) & set(pmap)
            sb_weights = {}
            for sb in feasible_sb:
                a = P[sa][sb]['weight']
                b = pmap[sb]
                sb_weights[sb] = a*b
            tot = np.sum(sb_weights.values())
            sb_distn = dict((sb, w / tot) for sb, w in sb_weights.items())

            # Add to the joint distn.
            for sb, pb in sb_distn.items():
                weighted_edges.append((sa, sb, pa * pb))

        # Add the joint distribution.
        J = nx.DiGraph()
        J.add_weighted_edges_from(weighted_edges)
        T_aug.add_edge(na, nb, J=J)

    # Return the augmented tree.
    return T_aug

