import pandas as pd

from pastml.tree import read_tree

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', required=True, type=str)
    parser.add_argument('--states', required=True, type=str)
    parser.add_argument('--drm', required=True, type=str)
    parser.add_argument('--loc', required=True, type=str)
    parser.add_argument('--out_tree', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_table(params.states, header=0, index_col=0)
    df.index = df.index.map(str)
    tree = read_tree(params.tree)

    drm = ''.join(_ for _ in params.drm if _.isalnum())
    loc = ''.join(_ for _ in params.loc if _.isalnum())

    max_tdr_size = 0
    tdr_root = None
    for _ in tree.traverse('postorder'):
        resistant = df.loc[_.name, drm]
        tdr_size = 0
        if _.is_leaf():
            tdr_size = 1 if 'resistant' == resistant else 0
        else:
            tdr_size = sum(getattr(c, 'TDR_size', 0) for c in _.children) if 'resistant' == resistant else 0
        _.add_feature('TDR_size', tdr_size)
        if tdr_size > max_tdr_size:
            max_tdr_size = tdr_size
            tdr_root = _

    tdr_loc = df.loc[tdr_root.name, loc]
    while tdr_root.up and tdr_loc == df.loc[tdr_root.up.name, loc]:
        tdr_root = tdr_root.up

    tdr_root.write(outfile=params.out_tree, format=3)
