import pandas
from functools import reduce

ALIGNED_GENE_SEQ_COL = 'alignedGeneSequences'
INPUT_SEQUENCE_COL = 'inputSequence'
SDRMS_COL = 'SDRMs'
GENE_COL = 'gene'

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--json', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    params = parser.parse_args()

    df = pandas.read_json(params.json)
    df[INPUT_SEQUENCE_COL] = df[INPUT_SEQUENCE_COL].apply(lambda d: d['header'])
    df.set_index(keys=INPUT_SEQUENCE_COL, drop=True, inplace=True)

    # split a list of genes inside the alignedGeneSequences column into several rows
    # then split the dictionaries inside s rows into several columns
    gene_df = df[ALIGNED_GENE_SEQ_COL].apply(pandas.Series, 1).stack().apply(pandas.Series)
    gene_df[GENE_COL] = gene_df[GENE_COL].apply(lambda d: d['name']).astype('category', ordered=True)
    gene_df[SDRMS_COL] = gene_df[SDRMS_COL].apply(lambda l: [d['text'] for d in l])
    gene_df.index = gene_df.index.droplevel(-1)

    # Put all the DRMs together and make them columns
    gene_df[SDRMS_COL] = \
        gene_df.apply(lambda row: {'%s:%s' % (row[GENE_COL], m): True for m in row[SDRMS_COL]}, axis=1)


    def join_dicts(ds):
        return reduce(lambda d1, d2: {**d1, **d2}, ds, {})


    gene_df = gene_df.groupby(gene_df.index)[SDRMS_COL].apply(list).apply(join_dicts).apply(pandas.Series)

    lbls_to_drop = [ALIGNED_GENE_SEQ_COL]
    df.drop(labels=lbls_to_drop, axis=1).join(gene_df).to_csv(params.data, sep='\t')

