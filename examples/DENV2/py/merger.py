import pandas as pd

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', type=str, nargs='+')
    parser.add_argument('--output_data', required=True, type=str)
    params = parser.parse_args()

    df = None
    for tab in params.input_data:
        ddf = pd.read_csv(tab, sep='\t', header=0, index_col=0, skipfooter=1)
        if df is None:
            df = ddf
        else:
            df = df.append(ddf)
    df = df[~df.index.duplicated(keep='first')]
    df.index = df.index.map(lambda _: _.replace('_', ''))
    df.to_csv(params.output_data, sep='\t', index_label='accession')
    mrf = df['norm_rf'].mean()
    mqt = df['norm_qt'].mean()
    df.loc['mean', ['norm_rf', 'norm_qt']] = mrf, mqt
    df.to_csv(params.output_data, sep='\t')

