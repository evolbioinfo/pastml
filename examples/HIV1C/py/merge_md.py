import pandas as pd

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_phy', required=True, type=str)
    parser.add_argument('--data_la', required=True, type=str)
    parser.add_argument('--data', required=True, type=str)
    params = parser.parse_args()

    df_phy = pd.read_table(params.data_phy, header=0)[['Access Number', 'Country Code', 'Year', 'Subtype', 'Name']]
    df_phy.index = df_phy['Access Number']
    df_phy.drop(['Access Number'], axis=1, inplace=True)

    df_la = pd.read_table(params.data_la, header=0, index_col=0)
    df = pd.concat([df_la, df_phy])
    df = df[~df.index.duplicated(keep='first')]
    df.to_csv(params.data, sep='\t', header=True, index=True, index_label='Access Number')

