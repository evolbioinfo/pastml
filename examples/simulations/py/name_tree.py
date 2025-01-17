from collections import Counter

import pandas as pd
from ete3 import Tree
from pastml.tree import name_tree


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_tree', required=True, type=str)
    parser.add_argument('--output_tree', required=True, type=str)
    params = parser.parse_args()

    tree = Tree(params.input_tree, format=5)
    name_tree(tree)
    tree.write(outfile=params.output_tree, format=3, format_root_node=True)
