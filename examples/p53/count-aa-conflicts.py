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
from collections import defaultdict
import json

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

    # Construct the list of dicts to convert to json.
    json_dicts = []
    for i in range(393):
        codon_pos = i + 1
        nconflicts = 0
        lethal_residues = pos_to_lethal_residues.get(codon_pos, None)

        # Get the leaf info and count the number of wild-type disease residues.
        leaf_info = []
        for idx, (name, codons) in enumerate(name_codons_list):
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

        json_dicts.append({
            'pos' : codon_pos,
            'nconflicts' : nconflicts,
            'leaf_info' : leaf_info,
            })

    # Write the json data.
    print(json.dumps(json_dicts, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    main(parser.parse_args())
