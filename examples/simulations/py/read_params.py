from collections import Counter

import pandas as pd

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_params', type=str, nargs='+')
    parser.add_argument('--output_log', required=True, type=str)
    params = parser.parse_args()

    data = []
    for f in params.input_params:
        df = pd.read_csv(f, header=0, index_col=0, sep='\t')
        if 'A' in df.index and 'C' in df.index and 'G' in df.index and 'T' in df.index:
            data.append([float(df.loc['scaling_factor', 'value']), (float(df.loc['kappa', 'value']) if 'kappa' in df.index else 1), float(df.loc['A', 'value']), float(df.loc['C', 'value']), float(df.loc['G', 'value']), float(df.loc['T', 'value'])])
    df = pd.DataFrame(data=data, columns=['SF', 'kappa', 'A', 'C', 'G', 'T'])
    df.describe().to_csv(params.output_log, sep='\t')
    print(df[df['SF'] > 5])

