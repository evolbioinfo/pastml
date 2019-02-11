import logging
from collections import Counter
from Bio import SeqIO
from Bio.Alphabet import generic_dna


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--format', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)

    sequences = list(SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna))
    common_len = Counter(len(seq.seq) for seq in sequences).most_common(n=1)[0][0]
    bad_sequences = [(seq.id, len(seq.seq)) for seq in sequences if len(seq.seq) != common_len]
    if bad_sequences:
        logging.error('Sequences {} have bad length'.format(bad_sequences))
        sequences = [seq for seq in sequences if len(seq.seq) == common_len]
    count = SeqIO.write(sequences, params.output_fa, params.format)
    logging.info("Converted {} records to {}".format(count, params.format))
