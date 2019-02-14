import logging
from collections import Counter

import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
from numpy.random import choice

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--input_tab', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--output_tab', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)

    id2seq = {_.id: _ for _ in SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna)}
    ids = choice(list(id2seq.keys()), size=len(id2seq), replace=True)

    used_ids = Counter()

    df = pd.read_csv(params.input_tab, header=0, index_col=0, sep='\t')
    df = df.loc[set(ids), :]

    def get_seq(id):
        seq = id2seq[id]
        n = used_ids[id]
        used_ids[id] += 1
        if n:
            new_id = '{}.{}'.format(id, n)
            seq = SeqRecord(seq=seq.seq, id=new_id, description='')
            df.loc[new_id, :] = df.loc[id, :]
        return seq


    count = SeqIO.write((get_seq(_) for _ in ids), params.output_fa, "fasta")
    df.to_csv(params.output_tab, index_label='id', sep='\t')
