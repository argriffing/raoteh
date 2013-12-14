"""
For each p53 amino acid position, compare disease amino acids to wild types.

For each position in the p53 codon alignment,
count the number of taxa whose wild-type amino acid at that position
has been found in a human tumor.

Per-leaf information is also written per site.

Write the output in json format for javascript visualization compatibility.

"""
from __future__ import print_function, division

import argparse
from StringIO import StringIO
from collections import defaultdict

import json
import networkx as nx

import layout
import app_helper
from app_helper import read_interpreted_disease_data

LETHAL = 'LETHAL'
BENIGN = 'BENIGN'
UNKNOWN = 'UNKNOWN'

# Hardcode some filenames.
g_interpreted_disease_filename = 'int1.out'
g_alignment_filename = 'testseq'
g_genetic_code_filename = 'universal.code.txt'


#TODO copypasted from summarize-disease-data.py
def get_disease_info(fin):
    """
    Read some filtered disease data.

    The interpretation filters the p53 disease data
    by assigning a disease state to each of the 20 amino acids,
    for each codon position in the reference (human) p53 sequence.
    The possible disease states are BENIGN, LETHAL, or UNKNOWN.

    """
    interpreted_disease_data = read_interpreted_disease_data(fin)

    # Define the set of codon positions.
    pos_set = set()
    for pos, residue, status in interpreted_disease_data:
        pos_set.add(pos)

    pos_to_benign_residues = defaultdict(set)
    pos_to_lethal_residues = defaultdict(set)
    for pos, residue, status in interpreted_disease_data:
        if status == BENIGN:
            pos_to_benign_residues[pos].add(residue)
        elif status == LETHAL:
            pos_to_lethal_residues[pos].add(residue)
        elif status == UNKNOWN:
            raise NotImplementedError(
                    'unknown amino acid status in the reference process '
                    'requires integrating over too many things')
        else:
            raise Exception('invalid disease status: ' + str(status))
    pos_to_benign_residues = dict(
            (pos, frozenset(s)) for pos, s in pos_to_benign_residues.items())
    pos_to_lethal_residues = dict(
            (pos, frozenset(s)) for pos, s in pos_to_lethal_residues.items())

    return pos_set, pos_to_lethal_residues


def translate_tree(nx_tree, nx_root):
    nx_parent = None
    return rtranslate_tree(nx_tree, nx_parent, nx_root)


def rtranslate_tree(nx_tree, nx_parent, nx_node):
    """
    The tree is a networkx tree.
    The root is an index.
    All nodes in this tree are represented by integers.
    """
    if nx_parent is None:
        blen = None
    else:
        blen = nx_tree[nx_parent][nx_node]['weight']
    child_nodes = []
    for node in nx_tree[nx_node]:
        if node != nx_parent:
            child_nodes.append(rtranslate_tree(nx_tree, nx_node, node))
    return (nx_node, blen, child_nodes)


def main(args):

    # Read the interpreted disease data.
    # It returns a map from codon position to lethal residue set.
    if args.verbose:
        print('reading interpreted disease data...')
    with open(g_interpreted_disease_filename) as fin:
        pos_set, pos_to_lethal_residues = get_disease_info(fin)

    # Read the codon alignment.
    # It is a list of (taxon_name, codon_list) pairs.
    if args.verbose:
        print('reading codon alignment...')
    with open(g_alignment_filename) as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # Read the genetic code.
    # It is a list of (state, residue, codon) triples.
    if args.verbose:
        print('reading genetic code...')
    with open(g_genetic_code_filename) as fin:
        genetic_code = app_helper.read_genetic_code(fin)
    codon_to_residue = dict((c, r) for s, r, c in genetic_code)

    # Read the newick string.
    # NOTE This is 'jeff data e' from liwen-branch-expecatations.py
    tree_string = """((((((Has:  0.0073385245,Ptr:  0.0073385245):  0.0640509640,Ppy:  0.0713894884):  0.0542000118,(((Mmu:  0.0025462071,Mfu:  0.0025462071):  0.0000000000,Mfa:  0.0025462071):  0.0318638454,Cae:  0.0344100525):  0.0911794477):  0.1983006745,(Mim:  0.3238901743,Tgl:  0.3238901743):  0.0000000004):  0.2277808059,((((((Mum:  0.1797319785,Rno:  0.1797319785):  0.1566592047,Mun:  0.3363911832):  0.0192333544,(Cgr:  0.1074213106,Mau:  0.1074213106):  0.2482032271):  0.0447054051,Sju:  0.4003299428):  0.1000000288,(Cpo:  0.4170856630,Mmo:  0.4170856630):  0.0832443086):  0.0250358682,(Ocu:  0.4149196099,Opr:  0.4149196099):  0.1104462299):  0.0263051408):  0.0000000147,(Sar:  0.4524627987,((Fca:  0.2801000848,Cfa:  0.2801000848):  0.1338023902,((Bta:  0.0880000138,Oar:  0.0880000138):  0.1543496707,Dle:  0.2423496845):  0.1715527905):  0.0385603236):  0.0992081966);"""
    fin = StringIO(tree_string)
    nx_tree, nx_root, leaf_name_pairs = app_helper.read_newick(fin)
    leaf_to_name = dict(leaf_name_pairs)
    name_to_leaf = dict((v, k) for k, v in leaf_name_pairs)
    #print(tree)
    #print(root)
    #print(leaf_name_pairs)

    # Translate the tree into a more layout friendly form.
    # Then compute the layout.
    root = translate_tree(nx_tree, nx_root)
    vert, horz, nodes = layout.layout_tree(root)
    #The vert_out is a sequence of (x, y1, y2) lines.
    #The horz_out is a sequence of (y, x1, x2, handle1, handle2) lines.
    #The nodes_out is a sequence of (x, y, handle) lines.
    #print(vert)
    #print(horz)
    #print(nodes)
    #for x, y, handle in nodes:
        #if handle in leaf_to_name:
            #print(x, y, leaf_to_name[handle])
    #return

    # Construct the list of dicts to convert to json.
    json_dicts = []
    for i in range(393):
        codon_pos = i + 1
        nconflicts = 0
        lethal_residues = pos_to_lethal_residues.get(codon_pos, None)

        # Get the leaf info and count the number of wild-type disease residues.
        leaf_info = []
        for name, codons in name_codons_list:
            idx = name_to_leaf[name]
            codon = codons[i]
            residue = codon_to_residue[codon]
            disease = 0
            if lethal_residues and residue in lethal_residues:
                disease = 1
            leaf_info.append({
                'idx' : idx,
                'name' : name,
                'codon' : codon,
                'residue' : residue,
                'disease' : disease,
                })
            nconflicts += disease

        # Get vertical lines.
        json_vert_lines = []
        for x, y1, y2 in vert:
            json_vert_lines.append(dict(x=x, y1=y1, y2=y2))

        # Get horizontal lines.
        json_horz_lines = []
        for y, x1, x2, idx1, idx2 in horz:
            json_horz_lines.append(dict(y=y, x1=x1, x2=x2))

        # Get nodes.
        json_nodes = []
        for x, y, handle in nodes:
            json_nodes.append(dict(x=x, y=y))

        # Append the site info dictionary.
        json_dicts.append(dict(pos=codon_pos, nconflicts=nconflicts,
            leaf_info=leaf_info, nodes=json_nodes,
            vert_lines=json_vert_lines,
            horz_lines=json_horz_lines))

    # Write the json data.
    print(json.dumps(json_dicts, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
