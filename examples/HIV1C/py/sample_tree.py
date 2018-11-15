import pandas as pd
import numpy as np

from pastml.tree import read_tree, remove_certain_leaves

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_tree', required=True, type=str)
    parser.add_argument('--metadata', required=True, type=str)
    parser.add_argument('--out_tree', required=True, type=str)
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--column', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_table(params.metadata, header=0, index_col=0)
    df.index = df.index.map(str)
    tree = read_tree(params.in_tree)
    df = df[np.in1d(df.index, [n.name for n in tree.iter_leaves()])]
    sample = []
    for _ in df[params.column].unique():
        if not pd.isnull(_):
            n = min(len(df[df[params.column] == _]), params.n)
            sample.extend(df[df[params.column] == _].sample(n).index)
    tree = read_tree(params.in_tree)
    tree = remove_certain_leaves(tree, to_remove=lambda node: node.name not in sample)
    tree.write(outfile=params.out_tree, format=3)