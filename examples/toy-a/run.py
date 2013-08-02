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


The partition of primary states into tolerance classes is as follows:

T0 : {P0, P1}
T1 : {P2, P3}
T2 : {P4, P5}


Tree schematic:

    N0
     |
     |
    N1
    / \
   /   \
 N2    N3


Primary process data:

N0 : primary process state is P0
N1 : primary process state is unknown
N2 : primary process state is P4
N3 : primary process state is P5


Tolerance class data (disease data):

     T0       T1       T2
    ---------------------
N0 : on       off      unknown
N1 : unknown  unknown  unknown
N2 : unknown  unknown  on
N3 : unknown  unknown  on

"""

def main():

    # A summary of the tree.
    nnodes = 4
    
    # Define the primary process and the tolerance processes.
    nprimary = 6
    nparts = 3
    primary_to_part = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
    node_to_allowed_primary_states = {
            0 : {0},
            1 : {0, 1, 2, 3, 4, 5},
            2 : {4},
            3 : {5},
            }
    preorder_nodes = (0, 1, 2, 3)
    preorder_edges = ((0, 1), (1, 2), (1, 3))
    node_to_part_to_allowed_states = {
            0 : {0:{1}, 1:{0}, 2:{0,1}},
            1 : {0:{0,1}, 1:{0,1}, 2:{0,1}},
            2 : {0:{0,1}, 1:{0,1}, 2:{1}},
            2 : {0:{0,1}, 1:{0,1}, 2:{1}},
            }
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

    # Define the compound process.
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

    # Define the switching model with the default and reference processes.
    # In this model, the N0 reference node is associated with the reference
    # process which disallows tolerance class T1, but which
    # allows the other two tolerance classes T0 and T2.
    # In contrast to the more restrictive reference model,
    # the default model tolerates all tolerance classes.
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

