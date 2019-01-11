import os
from Bio.Phylo import write, parse

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str)
    parser.add_argument('--pattern', required=True, type=str)
    params = parser.parse_args()

    i = 0
    for tree in parse(params.trees, 'newick'):
        os.makedirs(os.path.dirname(params.pattern % i), exist_ok=True)
        write([tree], params.pattern % i, 'newick', plain=True)
        i += 1

    print('Split a multi-tree file into {} one-tree ones.'.format(i))
