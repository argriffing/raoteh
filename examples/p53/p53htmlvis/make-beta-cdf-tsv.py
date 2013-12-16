from __future__ import division, print_function

from scipy import stats
import numpy as np

def main(alpha, beta):
    X = np.linspace(0, 1, num=393, endpoint=True)
    # print the header
    print('pos', 'cumulative', sep='\t')
    for i, cumulative in enumerate(stats.beta.cdf(X, alpha, beta)):
        pos = i + 1
        print(pos, cumulative, sep='\t')

if __name__ == '__main__':
    alpha = 2
    beta = 3
    main(alpha, beta)
