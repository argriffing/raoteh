"""
Rao-Teh samples of MJP trajectories on trees.

"""
from __future__ import division, print_function, absolute_import

import networkx as nx

from raoteh.sampler._mjp import get_total_rates


__all__ = []


#TODO move stuff from _sampler.py into this module


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

