import logging

import pandas as pd
import numpy as np


def get_min_forest_stats(forest):
    len_sum = 0
    num_zero_nodes = 0
    num_tips = 0
    num_nodes = 0

    for tree in forest:
        for node in tree.traverse():
            num_nodes += 1

            if not node.dist:
                num_zero_nodes += 1

            len_sum += node.dist

            if node.is_leaf():
                num_tips += 1

    avg_len = len_sum / (num_nodes - num_zero_nodes)
    return [avg_len, num_nodes, num_tips, len_sum]


class ForestStats(object):

    def __init__(self, forest):
        self.avg_nonzero_brlen, self.num_nodes, self.num_tips, self.forest_length = get_forest_stats(forest)
        self.num_trees = len(forest)


def get_forest_stats(forest):
    len_sum_ext = 0
    len_sum_int = 0
    num_zero_nodes = 0
    max_polynomy = 0
    max_len_ext = 0
    max_len_int = 0
    min_len_ext = np.inf
    min_len_int = np.inf
    num_tips = 0
    num_nodes = 0
    num_zero_tips = 0
    tip_len_sum = 0

    for tree in forest:
        for node in tree.traverse():
            num_nodes += 1
            max_polynomy = max(len(node.children), max_polynomy)

            if not node.dist:
                num_zero_nodes += 1

            if node.is_leaf():
                num_tips += 1
                tip_len_sum += node.dist
                if node.dist:
                    min_len_ext = min(node.dist, min_len_ext)
                    len_sum_ext += node.dist
                    max_len_ext = max(max_len_ext, node.dist)
                else:
                    num_zero_tips += 1
            else:
                if node.dist:
                    min_len_int = min(node.dist, min_len_int)
                    len_sum_int += node.dist
                    max_len_int = max(max_len_int, node.dist)

    avg_len = (len_sum_ext + len_sum_int) / (num_nodes - num_zero_nodes) if num_nodes > num_zero_nodes else 0
    avg_len_ext = len_sum_ext / (num_tips - num_zero_tips) if num_tips > num_zero_tips else 0
    avg_len_int = len_sum_int / (num_nodes - num_tips - num_zero_nodes + num_zero_tips) \
        if (num_nodes - num_tips - num_zero_nodes + num_zero_tips) > 0 else 0

    logging.getLogger('pastml').debug('\n=============TREE STATISTICS===================\n'
                                      '\tnumber of tips:\t{}\n'
                                      '\tnumber of zero-branch tips:\t{}\n'
                                      '\tnumber of internal nodes:\t{}\n'
                                      '\tmax number of children per node:\t{}\n'
                                      '\tmax tip branch length:\t{:.5f}\n'
                                      '\tmax internal branch length:\t{:.5f}\n'
                                      '\tmin non-zero tip branch length:\t{:.5f}\n'
                                      '\tmin non-zero internal branch length:\t{:.5f}\n'
                                      '\tavg non-zero tip branch length:\t{:.5f}\n'
                                      '\tavg non-zero internal branch length:\t{:.5f}\n'
                                      '\tavg non-zero branch length:\t{:.5f}.'
                                      .format(num_tips,
                                              num_zero_tips,
                                              num_nodes - num_tips,
                                              max_polynomy,
                                              max_len_ext,
                                              max_len_int,
                                              min_len_ext,
                                              min_len_int,
                                              avg_len_ext,
                                              avg_len_int,
                                              avg_len))
    return [avg_len, num_nodes, num_tips, len_sum_ext + len_sum_int]


def df2gdf(df):
    df.fillna('', inplace=True)
    gb = df.groupby(df.index)
    gdf = pd.DataFrame(columns=df.columns)
    for c in df.columns:
        gdf[c] = gb[c].apply(lambda vs: {v for v in vs if not pd.isnull(v) and v != ''})
    return gdf


def preannotate_forest(forest, df=None, gdf=None):
    if gdf is None:
        gdf = df2gdf(df)
    for tree in forest:
        for node in tree.traverse('postorder'):
            if node.name in gdf.index:
                node.add_features(**gdf.loc[node.name, :].to_dict())
            else:
                for c in gdf.columns:
                    node.del_feature(c)
    return gdf.columns, gdf
