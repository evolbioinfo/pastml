import logging
import os
from Bio.Phylo import NexusIO, write


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--trees', required=True, type=str)
    parser.add_argument('--pattern', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    i = 0
    for tree in NexusIO.parse(params.trees):
        os.makedirs(os.path.dirname(params.pattern % i), exist_ok=True)
        write([tree], params.pattern % i, 'newick', plain=True)
        i += 1
    logging.info('Converted %d trees to newick' % i)
