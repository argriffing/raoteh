"""
Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

__all__ = ['mysum']

# This is for checking the plumbing.
def mysum(*args):
    return sum(args)

def get_feasible_history(rate_matrix, tip_states, tree):
    """
    Find an arbitrary feasible history.

    Parameters
    ----------
    rate_matrix : weighted directed networkx graph
        This is a potentially sparse rate matrix.
    tip_states : dict
        Maps tip vertex ids to states.
    tree : weighted undirected networkx graph
        A representation of a tree.  Following networks convention,
        the distance along an edge is its 'weight'.

    Returns
    -------
    feasible_history : weighted undirected networkx graph
        A feasible history as a networkx graph.
        The format is similar to that of the input tree,
        except for a couple of differences.
        Additional degree-two vertices have been added at the points
        at which the state has changed along a branch.
        Each edge is annotated not only by the 'weight'
        that defines its length, but also by the 'state'
        which is constant along each edge.

    Notes
    -----
    The returned history is not sampled according to any particularly
    meaningful distribution.

    """

