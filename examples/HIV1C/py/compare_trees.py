import logging
import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from Bio.Phylo import parse
from ete3 import Tree
from scipy.misc import comb

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str)
    parser.add_argument('--labels', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--qt', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    data, indices = [], []
    trees = [Tree(_.format('newick')) for _ in parse(params.trees, 'newick')]
    with open(params.labels, 'r') as f:
        labels = f.readline().strip('\n').strip(' ').split('\t')
    n = len(trees)
    tri = np.zeros((n, n))
    tri[np.tril_indices(n, 0)] = np.fromfile(params.qt, dtype=float, sep='\t')
    tri /= comb(len(trees[0]), 4)

    def compare(args):
        i, j = args
        nqt = tri[j, i]
        logging.info('Comparing {} to {}: quartet dist is {}'.format(i, j, nqt))
        nrf = trees[i].compare(trees[j], unrooted=True)['norm_rf']
        logging.info('Comparing {} to {}: RF dist is {}'.format(i, j, nrf))
        return '{} vs {}'.format(labels[i], labels[j]), nrf, nqt

    with ThreadPool() as pool:
        dists = pool.map(func=compare, iterable=((i, j) for i in range(n - 1)
                                                 for j in range(i + 1, n)))
    for label, nrf, nqt in dists:
        data.append([nrf, nqt])
        indices.append(label)
    df = pd.DataFrame(data=data, columns=['norm_rf', 'norm_qt'], index=indices)
    mrf = df['norm_rf'].mean()
    mqt = df['norm_qt'].mean()
    df.loc['mean', ['norm_rf', 'norm_qt']] = mrf, mqt
    df.to_csv(params.output, sep='\t')

