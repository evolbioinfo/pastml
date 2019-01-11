import pandas as pd

from pastml.annotation import preannotate_tree
from pastml.tree import read_tree
from pastml import col_name2cat

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', required=True, type=str)
    parser.add_argument('--states', required=True, type=str)
    parser.add_argument('--drm', required=True, type=str)
    parser.add_argument('--loc', required=True, type=str)
    parser.add_argument('--out_tree', required=True, type=str)
    params = parser.parse_args()

    drm = col_name2cat(params.drm)
    loc = col_name2cat(params.loc)

    df = pd.read_table(params.states, header=0, index_col=0)[[drm, loc]]
    df.index = df.index.map(str)
    tree = read_tree(params.tree)
    preannotate_tree(df, tree)

    max_tdr_size = 0
    tdr_root = None
    for _ in tree.traverse('postorder'):
        resistant = getattr(_, drm, set())
        tdr_size = 0
        if resistant == {'resistant'}:
            tdr_size = 1 if _.is_leaf() else sum(getattr(c, 'TDR_size', 0) for c in _.children)
        _.add_feature('TDR_size', tdr_size)
        if tdr_size > max_tdr_size:
            max_tdr_size = tdr_size
            tdr_root = _

    tdr_loc = getattr(tdr_root, loc, set())
    while tdr_root.up and tdr_loc == getattr(tdr_root.up, loc, set()):
        tdr_root = tdr_root.up

    tdr_root.write(outfile=params.out_tree, format=3)
