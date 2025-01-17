import pandas as pd
from Bio import SeqIO
from Bio.Alphabet import generic_dna
import re


def get_date(name):
    date = re.sub(r'\..*', '', re.sub(r'^[AC]\.\w+\.', '', name))
    if date == 'x':
        return None
    if date.startswith('0') or date.startswith('1'):
        return '20' + date
    return '19' + date


def get_loc(name):
    loc = re.sub(r'\..*', '', re.sub(r'^[AC]\.', '', name))
    if loc == 'x':
        return None
    return loc


def get_st(name):
    st = re.sub(r'\..*', '', name)
    if st == 'x':
        return None
    return st


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--output_data', required=True, type=str)
    params = parser.parse_args()

    sequences = [_ for _ in SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna) if _.id.startswith('C.')]
    SeqIO.write((SeqIO.SeqRecord(id=re.search(r'[^.]+$', seq.id)[0], seq=seq.seq, description='') for seq in sequences),
                params.output_fa, "fasta")

    df = pd.DataFrame(data=[[_.id] for _ in sequences], index=[re.search(r'[^.]+$', _.id)[0] for _ in sequences],
                      columns=['Name'])

    df['Year'] = df['Name'].map(get_date)
    df['Country Code'] = df['Name'].map(get_loc)
    df['Subtype'] = df['Name'].map(get_st)

    df.to_csv(params.output_data, sep='\t', header=True, index=True, index_label='Access Number')
