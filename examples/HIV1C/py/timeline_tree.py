import pandas as pd
import numpy as np

from pastml.tree import read_tree, remove_certain_leaves

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_tree', required=True, type=str)
    parser.add_argument('--metadata', required=True, type=str)
    parser.add_argument('--out_tree_pattern', required=True, type=str)
    parser.add_argument('--drm', required=True, type=str)
    parser.add_argument('--date_column', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_table(params.metadata, header=0, index_col=0)
    df.index = df.index.map(str)
    tree = read_tree(params.in_tree)
    df = df[np.in1d(df.index, [n.name for n in tree.iter_leaves()])]
    res_df = df[df[params.drm] == 'resistant']
    first_year = min([_ for _ in res_df[params.date_column].unique() if not pd.isnull(_)])
    last_year = max([_ for _ in df[params.date_column].unique() if not pd.isnull(_)])

    for year, label in zip((last_year - 10, first_year), ('mid', 'first')):
        tree = remove_certain_leaves(tree, to_remove=lambda node: pd.isnull(df.loc[node.name, params.date_column])
                                                                  or df.loc[node.name, params.date_column] > year)
        tree.write(outfile=params.out_tree_pattern.format(label), format=3)
