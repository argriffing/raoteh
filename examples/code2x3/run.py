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
from __future__ import division, print_function, absolute_import

import itertools

import networkx as nx
import numpy as np

import extras


def do_pure_primary_process(
        Q_primary, primary_distn,
        preorder_nodes, preorder_edges, branch_length,
        node_to_allowed_primary_states,
        ):
    """
    """
    nnodes = len(preorder_nodes)
    nprimary = len(primary_distn)

    # Construct a rooted tree with branch lengths.
    root = preorder_nodes[0]
    T = nx.Graph()
    for na, nb in preorder_edges:
        T.add_edge(na, nb, weight=branch_length)

    # Get the expected number of transitions on each branch.
    # Count all transitions equally.
    edge_to_expectations = extras.get_expected_ntransitions(
            T, node_to_allowed_primary_states, root, nprimary,
            root_distn=primary_distn, Q_default=Q_primary)

    # Report the expectations.
    print('edge expectations:')
    for edge, expectation in edge_to_expectations.items():
        print(edge, expectation)
    print()


def do_switching_process(
            Q_primary, primary_distn,
            preorder_nodes, preorder_edges, branch_length,
            primary_to_part,
            tolerance_distn,
            switching_rate,
            node_to_allowed_primary_states,
            node_part_to_allowed_states,
            ):
    """
    Define the switching model with the default and reference processes.

    In this model, the N0 reference node is associated with the reference
    process which disallows tolerance class T1, but which
    allows the other two tolerance classes T0 and T2.
    In contrast to the more restrictive reference model,
    the default model tolerates all tolerance classes.

    """
    nnodes = len(preorder_nodes)
    nprimary = len(primary_distn)
    nparts = len(set(primary_to_part.values()))

    # Define the number of blocks in the state-switching rate matrix.
    nblocks = 2**nparts + 1

    # Define the number of states in the state-switching rate matrix.
    nswitch = nprimary * nblocks

    # Define the first index of the 'sink' block
    # which represents the default process
    # where every primary state is tolerated.
    sink_block_offset = 2**nparts * nprimary

    # Define the rate matrix itself.
    # Also construct a binary mask
    # that specifies the within-block transitions
    # and a separate mask that specifies the between-block transitions.
    Q_switching = np.zeros((nswitch, nswitch), dtype=float)
    tol_tuples = list(itertools.product((0, 1), repeat=nparts))
    for block_index, tol_tuple in enumerate(tol_tuples):

        # Construct a within-block mask
        # that defines which within-block transitions are allowed.
        within_block_mask = np.zeros((nprimary, nprimary), dtype=float)
        for c, c_part in primary_to_part.items():
            for d, d_part in primary_to_part.items():
                if c == d:
                    continue
                if tol_tuple[c_part] and tol_tuple[d_part]:
                    within_block_mask = 1

        # Define the diagonal block of the rate matrix.
        # The elements of the block that are on the diagonal
        # will not yet be meaningfully defined.
        a = block_index * nprimary
        b = (block_index + 1) * nprimary
        Q_switching[a:b, a:b] = Q_primary * within_block_mask

        # Add the transition rates that are allowed out of the current block.
        for c, c_part in primary_to_part.items():
            if tol_tuple[c_part]:
                Q_switching[a+c, sink_block_offset+c] = switching_rate

    # Add the within-block transition rates
    # for the block that defines the default process.
    Q_switching[sink_block_offset:, sink_block_offset:] = Q_primary

    # Define the diagonal of the switching process rate matrix.
    Q_switching = Q_switching - np.diag(Q_switching.sum(axis=1))

    # Get the initial state distribution.
    # This requires a probability parameter that is greater
    # when more primary states in the reference process
    # are likely to be tolerated.
    # We assume that the process cannot begin in the default process;
    # it must begin in some tolerated primary state in some
    # reference process, but the tolerance characteristics
    # of this initial reference process have some uncertainty.
    switching_distn = np.zeros(nswitch, dtype=float)
    for block_index, tol_tuple in enumerate(tol_tuples):
        n_untol = sum(1 for x in tol_tuple if not x)
        n_tol = sum(1 for x in tol_tuple if x)
        a = block_index * nprimary
        b = (block_index + 1) * nprimary
        for c, c_part in primary_to_part.items():
            if tol_tuple[c_part]:
                p_untol = tolerance_distn[0] ** n_untol
                p_tol = tolerance_distn[1] ** (n_tol - 1)
                switching_distn[a+c] = primary_distn[c] * p_untol * p_tol
    switching_distn_sum = switching_distn.sum()
    if not np.allclose(switching_distn_sum, 1):
        raise Exception(
                'expected initial probabilities to sum to 1'
                'but found ' + str(switching_distn_sum))

    # Distinguish within-block from between-block transitions
    # by constructing some indicator matrices.
    E_within_block = np.zeros_like(Q_switching)
    E_between_blocks = np.ones_like(Q_switching)
    for block_index in range(nblocks):
        a = block_index * nprimary
        b = (block_index + 1) * nprimary
        E_within_block[a:b, a:b] = 1
        E_between_blocks[a:b, a:b] = 0
    np.fill_diagonal(E_within_block, 0)
    np.fill_diagonal(E_between_blocks, 0)

    # For each node in the tree,
    # determine the set of allowed compound switching states.
    # This uses both alignment-like (primary) and disease-like (tolerance) data.
    node_to_allowed_switching_states = {}
    for na in range(nnodes):
        allowed_switching_states = set()
        part_to_allowed_states = dict(
                (p, node_part_to_allowed_states[na, p]) for p in range(nparts))
        for block_index, tol_tuple in enumerate(tol_tuples):
            a = block_index * nprimary
            b = (block_index + 1) * nprimary
            for primary, c_part in primary_to_part.items():
                switching_state = a + primary
                allowed_primary_in = node_to_allowed_primary_states[primary]

                # If the compound state is allowed,
                # then add it to the set.
                if _compound_state_is_allowed(
                        allowed_primary_in, part_to_allowed_states,
                        primary, tol_tuple):
                    allowed_switching_states.add(switching_state)

        # The non-root nodes may possibly have switched to the default process.
        if na > 0:
            for primary in range(nprimary):
                default_compound_state = sink_block_offset + primary
                if primary in allowed_primary_in:
                    allowed_switching_states.add(default_compound_state)

        # Associate the set of allowed switching states
        # with the appropriate node of the tree.
        node_to_allowed_switching_states[na] = allowed_switching_states

    # Report some expectations.
    # The expectations should be divided into two parts.
    # One part defines the expected number
    # of primary transitions on each branch.
    # The other part defines the expected number of reference-to-default
    # switches on each branch.
    # Because the reference-to-default switch can only happen once
    # on each path directed away from the reference root,
    # the expected number of reference-to-default switches
    # on a branch is the same as the probability of a reference-to-default
    # switch on the branch.

    # Construct a rooted tree with branch lengths.
    root = preorder_nodes[0]
    T = nx.Graph()
    for na, nb in preorder_edges:
        T.add_edge(na, nb, weight=branch_length)

    # Get the expected number of within-block transitions on each branch.
    edge_to_expectations = extras.get_expected_ntransitions(
            T, node_to_allowed_switching_states, root, nswitch,
            root_distn=switching_distn, Q_default=Q_switching,
            E=E_within_block)
    print('primary state transition edge expectations:')
    for edge, expectation in edge_to_expectations.items():
        print(edge, expectation)
    print()

    # Get the expected number of between-block transitions on each branch.
    edge_to_expectations = extras.get_expected_ntransitions(
            T, node_to_allowed_switching_states, root, nswitch,
            root_distn=switching_distn, Q_default=Q_switching,
            E=E_between_blocks)
    print('reference-to-default edge expectations:')
    for edge, expectation in edge_to_expectations.items():
        print(edge, expectation)
    print()


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

    # Define the primary process.
    nprimary = 6
    pre_Q_primary = np.array([
        [0, 1, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [0, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 1, 0],
        ], dtype=float)
    Q_primary = pre_Q_primary - np.diag(pre_Q_primary.sum(axis=1))

    # Compute the stationary distribution over the primary states.
    primary_weights = np.ones(nprimary, dtype=float)
    primary_distn = primary_weights / primary_weights.sum()
    
    # Rescale the primary rate matrix so that the expected rate is 1.0.
    expected_rate = -np.dot(primary_distn, np.diag(Q_primary))
    Q_primary /= expected_rate

    # Define the tolerance process.
    nparts = 3
    primary_to_part = {0:0, 1:0, 2:1, 3:1, 4:2, 5:2}
    rate_on = 1.0
    rate_off = 1.0

    # Define the switching rate for the rare-reference process.
    switching_rate = 1.0

    # Define the stationary distribution of the primary process
    # and the stationary distribution common to all tolerance processes.
    tolerance_weights = np.array([rate_off, rate_on], dtype=float)
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

    # For the pure primary process,
    # compute expectations with and without the alignment data.
    # For these calculations the tolerance process and the disease data
    # do not come into play.
    print('L0 pure primary process expectations:')
    do_pure_primary_process(
            Q_primary, primary_distn,
            preorder_nodes, preorder_edges, branch_length,
            L0_node_to_allowed_primary_states,
            )
    print()
    print('L1 pure primary process expectations:')
    do_pure_primary_process(
            Q_primary, primary_distn,
            preorder_nodes, preorder_edges, branch_length,
            L1_node_to_allowed_primary_states,
            )
    print()

    # Try the switching model with three levels of increasing amounts
    # of observed data (no data, primary only, primary plus tolerance).
    print('L0 rare-reference model process expectations:')
    print()
    do_switching_process(
            Q_primary, primary_distn,
            preorder_nodes, preorder_edges, branch_length,
            primary_to_part,
            tolerance_distn,
            switching_rate,
            L0_node_to_allowed_primary_states,
            L0_node_part_to_allowed_states,
            )
    print()



def compound_state_is_allowed(
        allowed_primary_states, part_to_allowed_states,
        compound_to_primary, compound_to_tolerances,
        candidate_compound_state):
    """
    This older interface is for compatibility with the switching model.

    """
    return _compound_state_is_allowed(
            allowed_primary_states, part_to_allowed_states,
            compound_to_primary[candidate_compound_state],
            compound_to_tolerances[candidate_compound_state])


def _compound_state_is_allowed(
        allowed_primary_states, part_to_allowed_states,
        candidate_primary, candidate_tolerances,
        ):
    """
    This helper function can be used for both blinking and switching models.

    """
    # check the primary state
    if candidate_primary not in allowed_primary_states:
        return False

    # check the tolerances
    for part, tolerance_state in enumerate(candidate_tolerances):
        if tolerance_state not in part_to_allowed_states[part]:
            return False

    return True


if __name__ == '__main__':
    main()

