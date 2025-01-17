import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--input_data', required=True, type=str)
    params = parser.parse_args()

    id2acc = pd.read_csv(params.input_data, header=0, index_col=0, sep='\t')['Access Number'].to_dict()

    SeqIO.write((SeqIO.SeqRecord(id=id2acc[seq.id], seq=seq.seq, description='')
                 for seq in SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna)),
                params.output_fa, "fasta")
