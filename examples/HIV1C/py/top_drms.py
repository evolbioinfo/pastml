import pandas as pd


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--n', required=True, type=int)
    params = parser.parse_args()

    df = pd.read_table(params.input)
    top_mutations = sorted([col for col in df.columns if 'RT:' in col or 'PR:' in col],
                           key=lambda drm: -len(df[df[drm] == 'resistant']))[: params.n]
    with open(params.output, 'w+') as f:
        f.write(' '.join("'{}'".format(drm) for drm in top_mutations))
