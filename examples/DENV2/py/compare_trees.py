import logging
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
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
    with open(params.labels, 'r') as f:
        labels = f.readline().strip('\n').strip(' ').split('\t')
    with open(params.trees, 'r') as f:
        trees = [Tree('{};'.format(_)) for _ in f.read().replace('\n', '').strip().split(';')[:-1]]
    n = len(trees)
    logging.info('Read {} trees'.format(n))
    tri = np.zeros((n, n))
    tri[np.tril_indices(n, 0)] = np.fromfile(params.qt, dtype=float, sep='\t')
    total_quartets = 2 * np.ones(tri.shape) * comb(len(trees[0]), 4)
    total_quartets -= tri
    tri /= total_quartets

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

