"""
"""

import argparse
from collections import defaultdict

from app_helper import read_interpreted_disease_data

LETHAL = 'LETHAL'
BENIGN = 'BENIGN'
UNKNOWN = 'UNKNOWN'

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
    with open(args.disease) as fin:
        pos_set, pos_to_lethal_residues = get_disease_info(fin)
    dpat_multiset = defaultdict(int)
    disease_patterns = set()
    for pos in pos_set:
        disease_pattern = pos_to_lethal_residues.get(pos, frozenset())
        disease_patterns.add(disease_pattern)
        dpat_multiset[disease_pattern] += 1
    rows = [(n, pat) for pat, n in dpat_multiset.items()]
    print 'count\tpattern'
    for n, pat in sorted(rows):
        print n, '\t', ', '.join(pat)
    print
    print 'number of unique disease patterns'
    print len(dpat_multiset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('disease', help='tsv disease data input file')
    main(parser.parse_args())

