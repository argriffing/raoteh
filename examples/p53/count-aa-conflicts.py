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
g_pos_handle_p_filename = 'node.data.tsv'


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
    Recursively translate a networkx tree into a more elementary format.

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


def get_layout_json_string(vert, horz, nodes):
    """
    Write a tree layout to a json string.

    """
    # Get vertical lines.
    vert_lines = []
    for x, y1, y2, handle in vert:
        vert_lines.append(dict(x=x, y1=y1, y2=y2))

    # Get horizontal lines.
    horz_lines = []
    for y, x1, x2, idx1, idx2 in horz:
        horz_lines.append(dict(y=y, x1=x1, x2=x2))

    # Get nodes.
    json_nodes = []
    for x, y, handle in nodes:
        json_nodes.append(dict(x=x, y=y))

    # Construct the top level dict.
    toplevel = dict(
            vert_lines=vert_lines, horz_lines=horz_lines, nodes=json_nodes)

    # Return the json string.
    return json.dumps(toplevel, indent=2)


def get_static_leaf_json_string(name_codons_list, name_to_leaf):
    """

    """
    leaf_info = []
    for name, codons in name_codons_list:
        idx = name_to_leaf[name]
        leaf_info.append(dict(name=name, idx=idx))
    return json.dumps(leaf_info, indent=2)


def get_layout_element_data(vert, horz, nodes, handle_to_p):
    """
    Return lists for json layout with data on branches.

    """
    # Get vertical lines.
    vert_lines = []
    for x, y1, y2, handle in vert:
        p = handle_to_p[handle]
        vert_lines.append(p)

    # Get horizontal lines.
    horz_lines = []
    for y, x1, x2, idx1, idx2 in horz:
        p1 = handle_to_p[idx1]
        p2 = handle_to_p[idx2]
        p = max(0, (p1 + p2) / 2)
        horz_lines.append(p)

    # Get nodes.
    json_nodes = []
    for x, y, handle in nodes:
        p = max(0, handle_to_p[handle])
        json_nodes.append(p)

    return vert_lines, horz_lines, json_nodes



def main(args):

    # Read the interpreted disease data.
    # It returns a map from codon position to lethal residue set.
    print('reading interpreted disease data...')
    with open(g_interpreted_disease_filename) as fin:
        pos_set, pos_to_lethal_residues = get_disease_info(fin)

    # Read the codon alignment.
    # It is a list of (taxon_name, codon_list) pairs.
    print('reading codon alignment...')
    with open(g_alignment_filename) as fin:
        name_codons_list = list(app_helper.read_phylip(fin))

    # Read the genetic code.
    # It is a list of (state, residue, codon) triples.
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
    # Make sure the order is the same.
    root = translate_tree(nx_tree, nx_root)
    leaves = [leaf for leaf, name in leaf_name_pairs]
    leaf_to_y = dict((x, x) for x in leaves)
    vert, horz, nodes = layout.layout_tree(root, leaf_to_y)
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

    # Write the layout to a json file.
    json_tree_layout = get_layout_json_string(vert, horz, nodes)
    with open(args.json_tree_out, 'w') as fout:
        fout.write(json_tree_layout)

    # Write the static (not codon-site-specific) leaf info to a json file.
    s = get_static_leaf_json_string(name_codons_list, name_to_leaf)
    with open(args.json_static_leaf_out, 'w') as fout:
        fout.write(s)

    # Read the tsv file that has the node probability data.
    pos_to_handle_to_p = dict((pos, {}) for pos in range(1, 393+1))
    with open(g_pos_handle_p_filename) as fin:
        for line in fin.readlines():
            line = line.strip()
            if line:
                s_pos, s_handle, s_p = line.split()
                pos = int(s_pos)
                handle = int(s_handle)
                p = float(s_p)
                handle_to_p = pos_to_handle_to_p[pos]
                handle_to_p[handle] = p

    # Construct the list of dicts to convert to json.
    json_dicts = []
    for i in range(393):
        codon_pos = i + 1
        handle_to_p = pos_to_handle_to_p[codon_pos]

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

        # Get per-layout-element data.
        vert_data, horz_data, node_data = get_layout_element_data(
                vert, horz, nodes, handle_to_p)

        # Append the site info dictionary.
        json_dicts.append(dict(
            pos=codon_pos,
            nconflicts=nconflicts,
            leaf_info=leaf_info,
            vert_data=vert_data,
            horz_data=horz_data,
            node_data=node_data,
            ))

    # Write the per-site layout element data to a json file.
    s = json.dumps(json_dicts, indent=2)
    with open(args.json_sites_out, 'w') as fout:
        fout.write(s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--json-tree-out',
            default='layout.data.json',
            help='write the json tree data to this file')
    parser.add_argument('--json-sites-out',
            default='site.data.json',
            help='write the per-site json data to this file')
    parser.add_argument('--json-static-leaf-out',
            default='static.leaf.data.json',
            help='write the static per-leaf json data to this file')
    main(parser.parse_args())

