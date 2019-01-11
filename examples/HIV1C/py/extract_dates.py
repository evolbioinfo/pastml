import pandas as pd

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', required=True, type=str)
    parser.add_argument('--dates', required=True, type=str)
    parser.add_argument('--c_tip', required=True, type=str)
    parser.add_argument('--date_col', required=True, type=str)
    params = parser.parse_args()

    date_df = pd.read_table(params.data, index_col=0)[[params.date_col]]
    date_df.fillna(params.c_tip, inplace=True)
    with open(params.dates, 'w+') as f:
        f.write('%d\n' % date_df.shape[0])

    date_df.to_csv(params.dates, sep='\t', header=False, mode='a')
