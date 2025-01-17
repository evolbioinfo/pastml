import logging
import re
import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna
from Bio.Seq import Seq


def clean_sequences(sequences, pos):
    for sequence in sequences:
        seq = str(sequence.seq)
        res = []
        end = 0
        for start in pos:
            res += seq[end: (start - 1) * 3]
            end = start * 3
        res += seq[end:]
        sequence.seq = Seq(''.join(res), generic_dna)
        yield sequence


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', required=True, type=str)
    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--PR_start_pos', required=True, type=int)
    parser.add_argument('--RT_start_pos', required=True, type=int)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S",
                        filename=None)

    DRMS = list(pd.read_csv(params.input_data, index_col=0, sep='\t').columns)
    pos = sorted({params.PR_start_pos + int(re.findall('\d+', _)[0]) for _ in DRMS if 'PR:' in _}) \
          + sorted({params.RT_start_pos + int(re.findall('\d+', _)[0]) for _ in DRMS if 'RT:' in _})

    count = SeqIO.write(clean_sequences(SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna), pos),
                        params.output_fa, "fasta")
    logging.info("Converted %d records to fasta" % count)



