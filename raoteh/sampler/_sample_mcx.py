"""
Sample Markov chain trajectories on trees.

This module uses a particularly simple observation type.
This is a thin wrapper that uses _mcx to get the node_to_pmap from
the observation or constraint data and uses _sample_mc0 to sample
the joint states using the node_to_pmap.

"""
from __future__ import division, print_function, absolute_import

from raoteh.sampler import _mcx, _sample_mc0


__all__ = []


def resample_states(T, root,
        node_to_state=None, root_distn=None, P_default=None):
    """
    This function applies to a tree for which nodes will be assigned states.

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
    node_to_sampled_state : dict
        A map from each node of T to its state.
        If the state was not defined by the node_to_state argument,
        then the state will have been sampled.

    """
    # Get the map from each node to a sparse map
    # from each feasible state to the subtree likelihood.
    node_to_pmap = _mcx.get_node_to_pmap(T, root,
            node_to_state=node_to_state, P_default=P_default)

    # Use the generic sampler.
    return _sample_mc0.resample_states(T, root, node_to_pmap,
            root_distn=root_distn, P_default=P_default)
