import pandas as pd
import numpy as np

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dates', required=False, type=str, default=None)
    parser.add_argument('--c_tip', required=True, type=str)
    parser.add_argument('--date_col', required=True, type=str)
    parser.add_argument('--random_date_files', type=str, nargs='*', default=[])
    params = parser.parse_args()

    date_df = pd.read_csv(params.data, index_col=0, sep='\t')[[params.date_col]]
    date_df.fillna(params.c_tip, inplace=True)

    if params.dates:
        with open(params.dates, 'w+') as f:
            f.write('%d\n' % date_df.shape[0])
        date_df.to_csv(params.dates, sep='\t', header=False, mode='a')

    for random_date_file in params.random_date_files:
        with open(random_date_file, 'w+') as f:
            f.write('%d\n' % date_df.shape[0])

        random_col = 'random_{}'.format(params.date_col)
        date_df[random_col] = np.random.choice(date_df[params.date_col], size=len(date_df), replace=False)
        date_df[[random_col]].to_csv(random_date_file, sep='\t', header=False, mode='a')

