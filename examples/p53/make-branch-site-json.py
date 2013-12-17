"""
This script uses tsv output files from liwen-branch-expectations.py.

It uses branch ordering conformant with that of count-aa-conflicts.py.

"""
from __future__ import division, print_function

import itertools
import json

import numpy as np

g_prior_switch_filename = 'prior.switch.data.tsv'
g_posterior_switch_filename = 'posterior.switch.data.tsv'
g_json_out_filename = 'horz.prob.data.json'


def get_posterior_cumulative_probs(fin):

    # Get the data rows.
    pairs = []
    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        s_pos, s_na, s_nb, s_p = line.split()
        pos = int(s_pos)
        na = int(s_na)
        nb = int(s_nb)
        p = max(0, float(s_p))
        pairs.append((pos, p))

    site_groups = []
    for key, group in itertools.groupby(pairs, lambda x : x[0]):
        site_groups.append(list(group))

    #print('site groups:')
    #print(site_groups)
    #print()

    #print('zip(*site_groups):')
    #print(zip(*site_groups))
    #print()

    branch_probs = []
    for branch in zip(*site_groups):
        branch_probs.append([p for pos, p in branch])

    branch_cumulative_probs = []
    for probs in branch_probs:
        cumulative_probs = np.cumsum([0] + probs).tolist()
        branch_cumulative_probs.append(cumulative_probs)

    return branch_cumulative_probs


def get_prior_probs(fin):

    # Get the data rows.
    probs = []
    for line in fin.readlines():
        line = line.strip()
        if not line:
            continue
        s_na, s_nb, s_p = line.split()
        na = int(s_na)
        nb = int(s_nb)
        p = max(0, float(s_p))
        probs.append(p)

    return probs



def main():
    with open(g_prior_switch_filename) as fin:
        prior_probs = get_prior_probs(fin)
    with open(g_posterior_switch_filename) as fin:
        posterior_cumulative_probs = get_posterior_cumulative_probs(fin)
    toplevel = []
    for prior, posterior in zip(prior_probs, posterior_cumulative_probs):
        d = dict(prior=prior, cumulative_posterior=posterior)
        toplevel.append(d)
    with open(g_json_out_filename, 'w') as fout:
        print(json.dumps(toplevel, indent=2), file=fout)


if __name__ == '__main__':
    main()

