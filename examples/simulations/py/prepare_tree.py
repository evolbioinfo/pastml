from ete3 import Tree
import numpy as np

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    params = parser.parse_args()

    tree = Tree(params.input, format=3)
    avg_nonzero_br_len = np.mean([_.dist for _ in tree.traverse() if _.dist])
    for _ in tree.traverse():
        _.dist /= avg_nonzero_br_len
    tree.resolve_polytomy()
    tree.write(outfile=params.output, format=5)
