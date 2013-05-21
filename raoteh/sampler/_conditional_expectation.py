"""
Functions related to conditional expectations for testing the Rao-Teh sampler.

"""
from __future__ import division, print_function, absolute_import

import math

import networkx as nx

__all__ = []


def get_jukes_cantor_rate_matrix(n=4):
    Q = nx.DiGraph()
    weight = 1 / float(n - 1)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            Q.add_edge(i, j, weight=weight)
    return Q

def get_jukes_cantor_probability(i, j, t, n=4):
    """
    p_{i, j}(t)
    """
    p = math.exp(-(n*t) / (n-1))
    if i == j:
        return (1 + p*(n-1))/n
    else:
        return (1-p)/n

def get_jukes_cantor_interaction(a, b, c, d, t, n=4):
    """
    I_{c, d}^{a, b}(t)
    """
    p = math.exp(-(n*t) / (n-1))
    pm1 = math.expm1(-(n*t) / (n-1))
    if a != c and d != b:
        x = t*p + pm1*2*(n-1)/n
    elif a == c and d == b:
        x = (n-1)*(n-1)*t*p - pm1*2*(n-1)*(n-1)/n
    else:
        x = -(n-1)*t*p - pm1*(n-2)*(n-1)/n
    return (t + x) / (n*n)

