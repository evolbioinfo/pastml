import logging
from Bio import SeqIO
from Bio.Alphabet import generic_dna

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--in_aln', required=True, type=str)
    parser.add_argument('--out_aln_pattern', required=True, type=str)
    parser.add_argument('--chunk_size', required=False, type=int, default=10)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)

    sequences = list(SeqIO.parse(params.in_aln, 'fasta', alphabet=generic_dna))
    i = 0
    start = 0
    while start < len(sequences):
        SeqIO.write(sequences[start: min(len(sequences), start + params.chunk_size)],
                    params.out_aln_pattern.format(i), 'fasta')
        start += params.chunk_size
        i += 1
