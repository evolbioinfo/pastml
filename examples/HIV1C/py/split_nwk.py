import os
from Bio.Phylo import write, NewickIO

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str)
    parser.add_argument('--pattern', required=True, type=str)
    params = parser.parse_args()

    i = 0
    for tree in NewickIO.parse(params.trees):
        os.makedirs(os.path.dirname(params.pattern % i), exist_ok=True)
        write([tree], params.pattern % i, 'newick', plain=True)
        i += 1

