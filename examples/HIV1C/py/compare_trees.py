import os

import numpy as np
import pandas as pd
from scipy.misc import comb

from pastml.tree import read_tree

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str, nargs='+')
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--qt', required=True, type=str)
    params = parser.parse_args()

    results = pd.DataFrame(columns=['tree1', 'tree2', 'norm_rf', 'norm_qt'])
    trees = list(params.trees)
    prefix = os.path.commonprefix(trees)
    path2tree = {_: read_tree(_) for _ in trees}

    qt_total = comb(len(next(iter(path2tree.values()))), 4)

    dm = np.fromfile(params.qt, dtype=float, sep='\t')
    tri = np.zeros((len(trees), len(trees)))
    tri[np.tril_indices(len(trees), 0)] = dm

    for i in range(len(trees) - 1):
        path1 = trees[i]
        tree1 = path2tree[path1]
        path1 = path1[len(prefix):]
        for j in range(i + 1, len(trees)):
            path2 = trees[j]
            tree2 = path2tree[path2]
            path2 = path2[len(prefix):]
            nrf = tree1.compare(tree2, unrooted=True)['norm_rf']
            nqt = tri[j, i] / comb(len(tree1), 4)
            results = results.append({'tree1': path1, 'tree2': path2, 'norm_rf': nrf, 'norm_qt': nqt}, ignore_index=True)
    mrf = results['norm_rf'].mean()
    mqt = results['norm_qt'].mean()
    results = results.append({'tree1': 'mean', 'tree2': 'mean', 'norm_rf': mrf, 'norm_qt': mqt}, ignore_index=True)

    results.to_csv(params.output, sep='\t', index=False)

