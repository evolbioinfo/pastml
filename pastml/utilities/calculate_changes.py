from collections import Counter

import pandas as pd

from pastml.annotation import preannotate_forest
from pastml.tree import read_tree


def state_combinations(node, columns, i=0):
    if i < len(columns):
        if i == len(columns) - 1:
            for state in getattr(node, columns[i]):
                yield (state,)
        else:
            for cmb in state_combinations(node, columns, i + 1):
                for state in getattr(node, columns[i]):
                    yield (state, *cmb)


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', required=True, type=str)
    parser.add_argument('--acr', required=True, type=str)
    parser.add_argument('--columns', default=None, type=str, nargs='*')
    parser.add_argument('--out_log', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_csv(params.acr, header=0, index_col=0, sep='\t')
    df.index = df.index.map(str)
    tree = read_tree(params.tree)
    preannotate_forest(df=df, forest=[tree])

    if params.columns is None or len(params.columns) == 0:
        params.columns = df.columns

    for n in tree.traverse():
        n.add_feature('state', set(state_combinations(n, params.columns)))
    
    from_to_count = Counter()
    for n in tree.traverse('preorder'):
        if n.is_root():
            continue
        states = getattr(n, 'state')
        up_states = getattr(n.up, 'state')
        for s in states:
            for u in up_states:
                if s != u:
                    from_to_count[(u, s)] += 1 / len(states) / len(up_states)

    df = pd.DataFrame(index=range(len(from_to_count)))
    df['from_to'] = list(from_to_count.keys())
    df['from'] = df['from_to'].apply(lambda _: _[0])
    df['to'] = df['from_to'].apply(lambda _: _[1])
    df.drop(labels=['from_to'], axis=1, inplace=True)
    df['count'] = from_to_count.values()
    df['normalized count'] = df['count'] / df['count'].sum()
    df.sort_values(by=['count', 'from', 'to'], axis=0, inplace=True, ascending=False)
    df.to_csv(params.out_log, index=False, sep='\t')
