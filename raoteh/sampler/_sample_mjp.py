"""
Rao-Teh samples of MJP trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import networkx as nx

from raoteh.sampler._util import get_first_element

from raoteh.sampler._mjp import get_total_rates


__all__ = []


#TODO move more stuff from _sampler.py into this module


def resample_poisson(T, state_to_rate, root=None):
    """

    Parameters
    ----------
    T : weighted undirected acyclic networkx graph
        Weighted tree whose edges are annotated with states.
        In other words, this is an MJP trajectory.
    state_to_rate : dict
        Map the state to the expected number of poisson events
        per edge weight.
    root : integer, optional
        Root of the tree.

    Returns
    -------
    T_out : weighted undirected acyclic networkx graph
        Weighted tree without state annotation.

    """

    # If no root was specified then pick one arbitrarily.
    if root is None:
        root = get_first_element(T)

    # Define the next node.
    next_node = max(T) + 1

    # Build the list of weighted edges.
    weighted_edges = []
    for a, b in nx.bfs_edges(T, root):
        weight = T[a][b]['weight']
        state = T[a][b]['state']
        rate = state_to_rate[state]
        prev_node = a
        total_dwell = 0.0
        while True:
            dwell = np.random.exponential(scale = 1/rate)
            if total_dwell + dwell > weight:
                break
            total_dwell += dwell
            mid_node = next_node
            next_node += 1
            weighted_edges.append((prev_node, mid_node, dwell))
            prev_node = mid_node
        weighted_edges.append((prev_node, b, weight - total_dwell))

    # Return the resampled tree with poisson events on the edges.
    T_out = nx.Graph()
    T_out.add_weighted_edges_from(weighted_edges)
    return T_out


def get_uniformized_transition_matrix(Q,
        uniformization_factor=None, omega=None):
    """

    Parameters
    ----------
    Q : directed weighted networkx graph
        Rate matrix.
    uniformization_factor : float, optional
        A value greater than 1.
    omega : float, optional
        The uniformization rate.

    Returns
    -------
    P : directed weighted networkx graph
        Transition probability matrix.

    """

    if (uniformization_factor is not None) and (omega is not None):
        raise ValueError('the uniformization factor and omega '
                'should not both be provided')

    # Compute the total rates.
    total_rates = get_total_rates(Q)

    # Compute omega if necessary.
    if omega is None:
        if uniformization_factor is None:
            uniformization_factor = 2
        omega = uniformization_factor * max(total_rates.values())

    # Construct a uniformized transition matrix from the rate matrix
    # and the uniformization rate.
    P = nx.DiGraph()
    for a in Q:
        if Q[a]:
            weight = 1.0 - total_rates[a] / omega
            P.add_edge(a, a, weight=weight)
            for b in Q[a]:
                weight = Q[a][b]['weight'] / omega
                P.add_edge(a, b, weight=weight)

    # Return the uniformized transition matrix.
    return P

