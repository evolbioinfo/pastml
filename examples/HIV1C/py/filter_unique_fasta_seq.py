from Bio import SeqIO
from Bio.Alphabet import generic_dna

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--fa_in', required=True, type=str)
    parser.add_argument('--fa_out', required=False, type=str, default=None)
    params = parser.parse_args()

    unique_sequences = []
    ids = set()
    for seq in SeqIO.parse(params.fa_in, "fasta", alphabet=generic_dna):
        if seq.id in ids:
            continue
        ids.add(seq.id)
        unique_sequences.append(seq)

    SeqIO.write(unique_sequences, params.fa_out if params.fa_out else params.fa_in, "fasta")
