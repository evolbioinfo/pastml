import logging
import pandas as pd

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_data', required=True, type=str)
    parser.add_argument('--output_data', required=True, type=str)
    parser.add_argument('--col_name', required=True, type=str)
    parser.add_argument('--col_value', required=True, type=str)
    params = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

    df = pd.read_csv(params.input_data, index_col=0, sep='\t')
    df = df[df[params.col_name] == params.col_value]
    logging.info('Extracted %d ids matching the specified criteria' % len(df))
    with open(params.output_data, 'w+') as f:
        f.write('\n'.join(list(df.index.map(str))))
