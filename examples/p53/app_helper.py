"""
Application-specific helper functions.

"""

import networkx as nx
import dendropy


__all__ = [
        'gen_paragraphs',
        'read_phylip',
        'read_newick',
        'print_codon_distn',
        'read_genetic_code',
        ]


def gen_paragraphs(lines):
    para = []
    for line in lines:
        line = line.strip()
        if not line:
            if para:
                yield para
                para = []
        else:
            para.append(line)
    if para:
        yield para


def read_phylip(fin):
    """
    Yield (taxon name, codons) pairs.
    @param fin: file open for reading
    """

    # Get the paragraphs in the most inefficient way possible.
    # Ignore the first line which is also the first paragraph.
    paras = list(gen_paragraphs(fin))[1:]
    if len(paras) != 25:
        raise Exception('expected p53 alignment of 25 taxa')

    # Each paragraph defines a p53 coding sequence of some taxon.
    # The first line gives the taxon name.
    # The rest of the lines are codons.
    for para in paras:
        taxon_name = para[0]
        codons = ' '.join(para[1:]).split()
        if len(codons) != 393:
            raise Exception('expected 393 codons')
        yield taxon_name, codons


def read_newick(fin):
    """

    Returns
    -------
    T : undirected weighted networkx tree
        Tree with edge weights.
    root_index : integer
        The root node.
    leaf_name_pairs : sequence
        Sequence of (node, name) pairs.

    """
    # use dendropy to read this newick file
    t = dendropy.Tree(stream=fin, schema='newick')
    leaves = t.leaf_nodes()
    nodes = list(t.postorder_node_iter())
    non_leaves = [n for n in nodes if n not in leaves]
    ordered_nodes = leaves + non_leaves
    root_index = len(ordered_nodes) - 1

    # node index lookup
    node_id_to_index = dict((id(n), i) for i, n in enumerate(ordered_nodes))

    # build the networkx tree
    T = nx.Graph()
    edges = list(t.postorder_edge_iter())
    for i, edge in enumerate(edges):
        if edge.head_node and edge.tail_node:
            na = node_id_to_index[id(edge.head_node)]
            nb = node_id_to_index[id(edge.tail_node)]
            T.add_edge(na, nb, weight=edge.length)

    # get a list of (leaf, name) pairs for the table
    leaf_name_pairs = [(i, str(n.taxon)) for i, n in enumerate(leaves)]

    return T, root_index, leaf_name_pairs


def print_codon_distn(codon_to_state, state_to_prob):
    evolver_nts = 'TCAG'
    for first_nt in evolver_nts:
        for third_nt in evolver_nts:
            arr = []
            for second_nt in evolver_nts:
                codon = first_nt + second_nt + third_nt
                if codon not in codon_to_state:
                    p = 0.0
                else:
                    state = codon_to_state[codon]
                    p = state_to_prob[state]
                arr.append(p)
            print('\t'.join(('%1.6f' % p) for p in arr))


def read_genetic_code(fin):
    """

    Parameters
    ----------
    fin : input stream
        Open text stream for reading the genetic code.

    Returns
    -------
    genetic_code : list
        List of (state, residue, codon) triples.

    """
    genetic_code = []
    for line in fin:
        line = line.strip()
        if line:
            state, residue, codon = line.split()
            state = int(state)
            residue = residue.upper()
            codon = codon.upper()
            if residue != 'STOP':
                triple = (state, residue, codon)
                genetic_code.append(triple)
    return genetic_code
