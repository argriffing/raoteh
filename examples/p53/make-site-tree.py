"""
This is a utility script for visualization.

It helps to visualize a codon site of an alignment
together with the tree relating the taxa.

"""

from StringIO import StringIO
import argparse

import dendropy

from app_helper import gen_paragraphs, read_phylip

def pos_int(x):
    x = int(x)
    if x < 1:
        raise ValueError('expected a positive integer')
    return x


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


def main(args):

    #tree_string = """((((((Has:  0.0156250000,Ptr:  0.0156250000):  0.0156250000,Ppy:  0.0312500000):  0.0312500000,(((Mmu:  0.0078125000,Mfu:  0.0078125000):  0.0078125000,Mfa:  0.0156250000):  0.0156250000,Cae:  0.0312500000):  0.0312500000):  0.0625000000,(Mim:  0.0625000000,Tgl:  0.0625000000):  0.0625000000):  0.1250000000,((((((Mum:  0.0039062500,Rno:  0.0039062500):  0.0039062500,Mun:  0.0078125000):  0.0078125000,(Cgr:  0.0078125000,Mau:  0.0078125000):  0.0078125000):  0.0156250000,Sju:  0.0312500000):  0.0312500000,(Cpo:  0.0312500000,Mmo:  0.0312500000):  0.0312500000):  0.0625000000,(Ocu:  0.0625000000,Opr:  0.0625000000):  0.0625000000):  0.1250000000):  0.2500000000,(Sar:  0.2500000000,((Fca:  0.0625000000,Cfa:  0.0625000000):  0.0625000000,((Bta:  0.0312500000,Oar:  0.0312500000):  0.0312500000,Dle:  0.0625000000):  0.0625000000):  0.1250000000):  0.2500000000);"""

    with open('testseq') as fin:
        name_sequence_pairs = list(read_phylip(fin))

    #names, sequences = zip(*name_sequence_pairs)

    for name, sequence in name_sequence_pairs:
        print name, sequence[args.site-1]

    #fin = StringIO(tree_string)

    # use dendropy to read this newick file
    #t = dendropy.Tree(stream=fin, schema='newick')
    #leaves = t.leaf_nodes()
    #nodes = list(t.postorder_node_iter())
    #non_leaves = [n for n in nodes if n not in leaves]
    #ordered_nodes = leaves + non_leaves
    #root_index = len(ordered_nodes) - 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', type=pos_int, required=True,
            help='codon site (first position is 1)')
    main(parser.parse_args())

