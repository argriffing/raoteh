"""
This script uses tsv output files from liwen-branch-expectations.py.

It uses branch ordering conformant with that of count-aa-conflicts.py.

"""
from __future__ import division, print_function

from StringIO import StringIO
import itertools
import json
from collections import defaultdict

import numpy as np

import app_helper
import layout

g_prior_switch_filename = 'prior.switch.data.tsv'
g_posterior_switch_filename = 'posterior.switch.data.tsv'
g_json_out_filename = 'horz.prob.data.json'


def sorted_pair(a, b):
    return tuple(sorted((a, b)))

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


def get_element_data(horz, handle_pair_to_prior, handle_pair_to_post):
    """
    Deal with handle pairs.

    """
    # Get horizontal lines.
    toplevel = []
    for y, x1, x2, idx1, idx2 in horz:
        k = sorted_pair(idx1, idx2)
        prior = handle_pair_to_prior[k]
        post = handle_pair_to_post[k]
        d = dict(prior=prior, cumulative_posterior=post)
        toplevel.append(d)
    return toplevel


def get_posterior_cumulative_probs(fin):

    k_to_probs = defaultdict(list)

    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        s_pos, s_na, s_nb, s_p = line.split()
        pos = int(s_pos)
        na = int(s_na)
        nb = int(s_nb)
        k = sorted_pair(na, nb)
        p = max(0, float(s_p))
        k_to_probs[k].append(p)

    k_to_cumulative_probs = dict(
            (k, np.cumsum([0] + v).tolist()) for k, v in k_to_probs.items())

    return k_to_cumulative_probs


def get_prior_probs(fin):

    k_to_prob = dict()

    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        s_na, s_nb, s_p = line.split()
        na = int(s_na)
        nb = int(s_nb)
        k = sorted_pair(na, nb)
        p = max(0, float(s_p))
        k_to_prob[k] = p

    return k_to_prob



def main():

    tree_string = """((((((Has:  0.0073385245,Ptr:  0.0073385245):  0.0640509640,Ppy:  0.0713894884):  0.0542000118,(((Mmu:  0.0025462071,Mfu:  0.0025462071):  0.0000000000,Mfa:  0.0025462071):  0.0318638454,Cae:  0.0344100525):  0.0911794477):  0.1983006745,(Mim:  0.3238901743,Tgl:  0.3238901743):  0.0000000004):  0.2277808059,((((((Mum:  0.1797319785,Rno:  0.1797319785):  0.1566592047,Mun:  0.3363911832):  0.0192333544,(Cgr:  0.1074213106,Mau:  0.1074213106):  0.2482032271):  0.0447054051,Sju:  0.4003299428):  0.1000000288,(Cpo:  0.4170856630,Mmo:  0.4170856630):  0.0832443086):  0.0250358682,(Ocu:  0.4149196099,Opr:  0.4149196099):  0.1104462299):  0.0263051408):  0.0000000147,(Sar:  0.4524627987,((Fca:  0.2801000848,Cfa:  0.2801000848):  0.1338023902,((Bta:  0.0880000138,Oar:  0.0880000138):  0.1543496707,Dle:  0.2423496845):  0.1715527905):  0.0385603236):  0.0992081966);"""
    fin = StringIO(tree_string)
    nx_tree, nx_root, leaf_name_pairs = app_helper.read_newick(fin)
    leaf_to_name = dict(leaf_name_pairs)
    name_to_leaf = dict((v, k) for k, v in leaf_name_pairs)
    root = translate_tree(nx_tree, nx_root)
    leaves = [leaf for leaf, name in leaf_name_pairs]
    leaf_to_y = dict((x, x) for x in leaves)
    vert, horz, nodes = layout.layout_tree(root, leaf_to_y)

    with open(g_prior_switch_filename) as fin:
        prior_probs = get_prior_probs(fin)
    with open(g_posterior_switch_filename) as fin:
        posterior_cumulative_probs = get_posterior_cumulative_probs(fin)
    toplevel = get_element_data(
            horz, prior_probs, posterior_cumulative_probs)
    with open(g_json_out_filename, 'w') as fout:
        print(json.dumps(toplevel, indent=2), file=fout)


if __name__ == '__main__':
    main()

