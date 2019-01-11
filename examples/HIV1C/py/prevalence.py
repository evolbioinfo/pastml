import pandas as pd


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_table(params.input)
    df = df[[col for col in df.columns if 'RT:' in col or 'PR:' in col]]
    ((df == 'resistant').astype(int).sum().sort_values() / len(df)).to_csv(params.output, header=False, index=True, sep='\t')
