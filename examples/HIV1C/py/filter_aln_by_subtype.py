import logging
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_aln', required=True, type=str)
    parser.add_argument('--out_aln', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)
    df = pd.read_csv(params.data, index_col=0, header=0, sep='\t')
    ids = set(df[df['Subtype'] == df['Sierra subtype']].index)

    count = SeqIO.write((seq for seq in SeqIO.parse(params.in_aln, 'fasta', alphabet=generic_dna) if seq.id in ids),
                        params.out_aln, 'fasta')
    logging.info("Kept {} sequences".format(count))
