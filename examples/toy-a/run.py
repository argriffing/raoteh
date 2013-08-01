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
N3 : primary process state is P4


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
            3 : {4},
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
        for compound in range(ncompound):
            primary = compound_to_primary[compound]
            tolerances = compound_to_tolerances[compound]
            if primary in node_to_allowed_primary_states[node]:
            allowed_compound.add(compound)
        node_to_allowed_compound_states[node] = allowed_compound


def foo(allowed_primary_states, part_to_allowed_states,
        compound_to_primary, compound_to_tolerances,
        candidate_compound_state):
    primary = compound_to_primary[candidate_compound_state]
    tolerances = compound_to_tolerances[candidate_compound_state]


if __name__ == '__main__':
    main()

