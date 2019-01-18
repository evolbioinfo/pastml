import logging
import os

import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_aln', required=True, type=str)
    parser.add_argument('--out_aln', required=True, type=str)
    parser.add_argument('--ids', required=True, type=str, nargs='+')
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)

    ids = set()
    for id_txt in params.ids:
        if os.path.getsize(id_txt) > 0:
            ids |= set(pd.read_table(id_txt, index_col=0, header=None).index.map(str))

    count = SeqIO.write((seq for seq in SeqIO.parse(params.in_aln, 'fasta', alphabet=generic_dna) if seq.id in ids),
                        params.out_aln, 'fasta')
    logging.info("Kept {} sequences".format(count))
