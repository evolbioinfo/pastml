import logging

import pandas as pd


def get_tree_stats(tree):
    len_sum = 0
    num_zero_nodes = 0
    max_polynomy = 0
    max_len = 0
    num_tips = 0
    num_nodes = 0
    num_zero_tips = 0
    tip_len_sum = 0

    for node in tree.traverse():
        num_nodes += 1
        len_sum += node.dist
        max_polynomy = max(len(node.children), max_polynomy)
        max_len = max(max_len, node.dist)
        if not node.dist:
            num_zero_nodes += 1

        if node.is_leaf():
            num_tips += 1
            tip_len_sum += node.dist
            if not node.dist:
                num_zero_tips += 1

    avg_br_len = len_sum / (num_nodes - num_zero_nodes)
    logging.getLogger('pastml').debug('\n=============TREE STATISTICS===================\n'
                                      '\tnumber of tips:\t{}\n'
                                      '\tnumber of zero-branch tips:\t{}\n'
                                      '\tnumber of internal nodes:\t{}\n'
                                      '\tmax number of children per node:\t{}\n'
                                      '\tmax branch length:\t{:.5f}\n'
                                      '\tavg non-zero branch length:\t{:.5f}.'
                                      .format(num_tips,
                                              num_zero_tips,
                                              num_nodes - num_tips,
                                              max_polynomy,
                                              max_len,
                                              avg_br_len))
    return avg_br_len, num_nodes, num_tips


def preannotate_tree(df, tree):
    df.fillna('', inplace=True)
    gb = df.groupby(df.index)
    gdf = pd.DataFrame(columns=df.columns)
    for c in df.columns:
        gdf[c] = gb[c].apply(lambda vs: {v for v in vs if not pd.isnull(v) and v != ''})
    for node in tree.traverse('postorder'):
        if node.name in gdf.index:
            node.add_features(**gdf.loc[node.name, :].to_dict())
    return df.columns
