from collections import Counter

import pandas as pd
from ete3 import Tree


def name_tree(tree):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter((_.name for _ in tree.traverse() if _.name))
    i = 0
    for node in tree.traverse('postorder'):
        if not node.name or existing_names[node.name] > 1:
            name = 'ROOT' if node.is_root() else None
            while not name or name in existing_names:
                name = '{}_{}'.format('tip' if node.is_leaf() else ('ROOT' if node.is_root() else 'node'), i)
                i += 1
            node.name = name


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--input_log', required=True, type=str)
    parser.add_argument('--output_tab', required=True, type=str)
    parser.add_argument('--output_tree', required=True, type=str)
    params = parser.parse_args()

    tree = Tree(params.input_tree, format=5)
    name_tree(tree)
    df = pd.read_table(params.input_log, header=0, index_col=0)
    df.index = [_.name for _ in tree.traverse('preorder')]
    df.columns = ['Real']

    df['ACR'] = df['Real']
    df.loc[[_.name for _ in tree.traverse() if not _.is_leaf()], 'ACR'] = None

    df.to_csv(params.output_tab, sep='\t')
    tree.write(outfile=params.output_tree, format=3, format_root_node=True)
