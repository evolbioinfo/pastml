import numpy as np

from pastml.tree import read_tree, remove_certain_leaves

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_tree', required=True, type=str)
    parser.add_argument('--n', required=True, type=int)
    parser.add_argument('--out_tree', required=True, type=str)
    params = parser.parse_args()

    tree = read_tree(params.in_tree)
    ids = set(np.random.choice([_.name for _ in tree], size=params.n, replace=False))
    tree = remove_certain_leaves(tree, to_remove=lambda node: node.name not in ids)
    tree.write(outfile=params.out_tree, format=3)
