"""
Generic functions relevant to algorithms involving Markov chains.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from raoteh.sampler._util import (
        StructuralZeroProb, NumericalZeroProb,
        get_first_element, get_arbitrary_tip,
        get_normalized_dict_distn,
        )


__all__ = []


#TODO under destruction
def xxx_get_node_to_state_set(T, root, node_to_smap):
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

    Returns
    -------
    node_to_state_set : dict
        A map from each node to a set of valid states.

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
            raise ValueError('this a node has no predecessors or successors')
        node_to_state_set[nb] = allowed_states

    # Return the map from node to state set.
    return node_to_state_set


def get_likelihood(root_pmap, root_distn=None):
    """
    Compute a likelihood or raise an exception if the likelihood is zero.

    Parameters
    ----------
    root_pmap : dict
        A map from states at the root to conditional likelihoods.
    root_distn : dict, optional
        A sparse finite distribution or weights over root states.
        Values should be positive but are not required to sum to 1.
        If the distribution is not provided,
        then it will be assumed to have values of 1 for each possible state.

    Returns
    -------
    likelihood : float
        The likelihood.

    """
    # Check whether the prior by itself causes the likelihood to be zero.
    if (root_distn is not None) and not root_distn:
        raise StructuralZeroProb('no root state has nonzero prior likelihood')

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
        likelihood = sum(root_pmap[s] * root_distn[s] for s in feasible_rstates)
    else:
        likelihood = sum(root_pmap.values())

    # Return the likelihood.
    return likelihood

