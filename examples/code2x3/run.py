"""
This is for testing inference in a toy model.

Px is primary process state x.
Tx is tolerance class x, with states in {0, 1}.
Nx is node x in the tree.


Primary process rate matrix schematic:

P0 --- P2 --- P4
 |      |      |
 |      |      |
P1 --- P3 --- P5


The primary states are partitioned into tolerance classes as follows:

T0 : {P0, P1}
T1 : {P2, P3}
T2 : {P4, P5}


Rooted tree schematic, annotated with known primary process states at leaves:

        P0
        N0
         |
         |
        N1
        / \
       /   \
     N2    N5
     /\    P1
    /  \   
  N3   N4
  P4   P5


Available process state data:

N0 : primary process state is P0
N1 : primary process state is unknown
N2 : primary process state is unknown
N3 : primary process state is P4
N4 : primary process state is P5
N5 : primary process state is P1


Tolerance class data (disease data):

     T0       T1       T2
    ---------------------
N0 : on       (off)    (on)
N1 : unknown  unknown  unknown
N2 : unknown  unknown  unknown
N3 : unknown  unknown  on
N4 : unknown  unknown  on
N5 : on       unknown  unknown

"""

import itertools

import numpy as np


#TODO for now only get the total expected number of primary transitions
def do_pure_primary_process(
        Q_primary, primary_distn,
        preorder_nodes, preorder_edges, branch_length,
        node_to_allowed_primary_states,
        ):
    """
    """
    nnodes = len(preorder_nodes)
    nprimary = len(primary_distn)
    pass


def do_switching_process():
    """
    Define the switching model with the default and reference processes.

    In this model, the N0 reference node is associated with the reference
    process which disallows tolerance class T1, but which
    allows the other two tolerance classes T0 and T2.
    In contrast to the more restrictive reference model,
    the default model tolerates all tolerance classes.

    """
    primary_parts_allowed_in_reference = {0, 2}
    switch_rate = 1.0
    Q_default = Q_primary.copy()
    Q_reference = Q_primary.copy()
    for pa, pa_part in primary_to_part.items():
        for pb, pb_part in primary_to_part.items():
            for part in (pa_part, pb_part):
                if part not in primary_parts_allowed_in_reference:
                    Q_reference[pa, pb] = 0
    nswitching = 2*nprimary
    switch_diag = np.zero(nswitching, dtype=float)
    for primary_state, part in primary_to_part.items():
        if part in primary_parts_allowed_in_reference:
            switch_diag[primary_state] = switch_rate
    Q_switching = np.zeros((nswitching, nswitching), dtype=float)
    Q_switching[:nprimary, :nprimary] = Q_reference
    Q_switching[nprimary:, nprimary:] = Q_default
    Q_switching[:nprimary, nprimary:] = np.diag(switch_diag)
    Q_switching = Q_switching - np.diag(Q_switching.sum(axis=1))


def do_compound_process():
    """
    """
    compound_to_primary = []
    compound_to_tolerances = []
    for primary_state in range(nprimary):
        for tolerance_states in itertools.product((0, 1), repeat=nparts):
            compound_to_primary.append(primary_state)
            compound_to_tolerances.append(tolerance_states)
    ncompound = len(compound_to_primary)
    node_to_allowed_compound_states = {}
    for node in range(nnodes):
        allowed_compound = set()
        allowed_primary_states = node_to_allowed_primary_states[node]
        part_to_allowed_states = node_to_part_to_allowed_states[node]
        for compound in range(ncompound):
            if compound_state_is_allowed(
                    allowed_primary_states, part_to_allowed_states,
                    compound_to_primary, compound_to_tolerances, compound):
                allowed_compound.add(compound)
        node_to_allowed_compound_states[node] = allowed_compound


def main():

    # Define the primary process and the tolerance processes.
    nprimary = 6
    nparts = 3
    primary_to_part = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
    pre_Q_primary = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        ], dtype=float)
    Q_primary = pre_Q_primary - np.diag(pre_Q_primary.sum(axis=1))
    rate_on = 1.0
    rate_off = 1.0

    # Define the stationary distribution of the primary process
    # and the stationary distribution common to all tolerance processes.
    primary_weights = np.ones(nprimary, dtype=float)
    tolerance_weights = np.array([rate_off, rate_on], dtype=float)
    primary_distn = primary_weights / primary_weights.sum()
    tolerance_distn = tolerance_weights / tolerance_weights.sum()

    # Define the tree.
    nnodes = 6
    preorder_nodes = (0, 1, 2, 3, 4, 5)
    preorder_edges = ((0, 1), (1, 2), (2, 3), (2, 4), (1, 5))

    # Define the single branch length, assumed to be common to all branches.
    # This is given in units of expected number of primary state transitions
    # when all primary states are tolerated.
    branch_length = 0.5

    # L0, L1, L2 define three levels of data.
    # L0 provides no data.
    # L1 provides alignment data (primary states) only.
    # L2 provides alignment data and disease data (tolerance class states).

    # No primary states are known.
    L0_node_to_allowed_primary_states = dict(
            (n, {0, 1, 2, 3, 4, 5}) for n in range(nnodes))

    # No tolerance states are known.
    L0_node_part_to_allowed_states = dict(
            ((n, p), {0, 1}) for n in range(nnodes) for p in range(nparts))

    # Primary states are known at the degree-1 nodes.
    L1_node_to_allowed_primary_states = {
            0 : {0},
            1 : {0, 1, 2, 3, 4, 5},
            2 : {0, 1, 2, 3, 4, 5},
            3 : {4},
            4 : {5},
            5 : {1},
            }

    # At the degree-1 nodes,
    # the tolerance class of the known primary state is known to be tolerated.
    L1_node_part_to_allowed_states = {
            (0,0):{1}, (0,1):{0,1}, (0,2):{0,1},
            (1,0):{0,1}, (1,1):{0,1}, (1,2):{0,1},
            (2,0):{0,1}, (2,1):{0,1}, (2,2):{0,1},
            (3,0):{0,1}, (3,1):{0,1}, (3,2):{1},
            (4,0):{0,1}, (4,1):{0,1}, (4,2):{1},
            (5,0):{1}, (5,1):{0,1}, (5,2):{0,1},
            }

    # Primary states are known at the degree-1 nodes.
    L2_node_to_allowed_primary_states = L1_node_to_allowed_primary_states

    # At the degree-1 nodes,
    # the tolerance class of the known primary state is known to be tolerated.
    # Additionally all tolerance classes are known at the root;
    # this represents access to full disease data for a reference taxon.
    L2_node_part_to_allowed_states = {
            (0,0):{1}, (0,1):{0}, (0,2):{1},
            (1,0):{0,1}, (1,1):{0,1}, (1,2):{0,1},
            (2,0):{0,1}, (2,1):{0,1}, (2,2):{0,1},
            (3,0):{0,1}, (3,1):{0,1}, (3,2):{1},
            (4,0):{0,1}, (4,1):{0,1}, (4,2):{1},
            (5,0):{1}, (5,1):{0,1}, (5,2):{0,1},
            }


def compound_state_is_allowed(
        allowed_primary_states, part_to_allowed_states,
        compound_to_primary, compound_to_tolerances,
        candidate_compound_state):

    # check the primary state
    primary = compound_to_primary[candidate_compound_state]
    if primary not in allowed_primary_states:
        return False

    # check the tolerances
    tolerances = compound_to_tolerances[candidate_compound_state]
    for part, tolerance_state in enumerate(tolerances):
        if tolerance_state not in part_to_allowed_states[part]:
            return False

    return True


if __name__ == '__main__':
    main()

