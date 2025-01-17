from collections import Counter

import pandas as pd
from ete3 import Tree
from pastml.tree import name_tree


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--input_log', required=True, type=str)
    parser.add_argument('--output_tab', required=True, type=str)
    params = parser.parse_args()

    tree = Tree(params.input_tree, format=3)
    df = pd.read_csv(params.input_log, header=0, index_col=0, sep='\t')
    df.index = [_.name for _ in tree.traverse('preorder')]
    df.columns = ['Real']

    df['ACR'] = df['Real']
    df.loc[[_.name for _ in tree.traverse() if not _.is_leaf()], 'ACR'] = None
    df.to_csv(params.output_tab, sep='\t')
