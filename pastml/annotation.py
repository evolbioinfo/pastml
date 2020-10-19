import logging

import pandas as pd
import numpy as np

from pastml import MODEL_ID
from pastml.tree import DATE


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

    avg_len = (len_sum_ext + len_sum_int) / (num_nodes - num_zero_nodes)
    avg_len_ext = len_sum_ext / (num_tips - num_zero_tips)
    avg_len_int = len_sum_int / (num_nodes - num_tips - num_zero_nodes + num_zero_tips)
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


def annotate_skyline(forest, skyline):
    if not skyline:
        return [], 1
    min_date, max_date = min(getattr(tree, DATE) for tree in forest), \
                         max(max(getattr(_, DATE) for _ in tree) for tree in forest)
    # let's make the skyline contain starting dates, so that the intervals will be [skyline[i], skyline[i+1])
    start_skyline = max(_ for _ in skyline if _ <= min_date) if any(_ for _ in skyline if _ <= min_date) else min_date
    if not any(_ for _ in skyline if _ < max_date):
        skyline = []
    else:
        end_skyline = max(_ for _ in skyline if _ < max_date)
        skyline = sorted([_ for _ in skyline if start_skyline <= _ <= end_skyline])
    if not skyline or len(skyline) == 1 and skyline[0] <= min_date:
        logging.getLogger('pastml').warning('The skyline dates provided are outside of the tree dates: {} - {}, '
                                            'so we will apply the same model everywhere.'.format(min_date, max_date))
        return [], 1
    elif skyline[0] > min_date:
        skyline = [min_date] + skyline
    skyline_nodes = []

    logging.getLogger('pastml').debug('The tree(s) cover the period between {} and {}, '
                                      'the skyline intervals start at the following dates: {}.'
                                      .format(min_date, max_date, ', '.join(str(_) for _ in skyline)))

    def annotate_node_skyline(node, i):
        for j in range(i, len(skyline)):
            if skyline[j] <= getattr(node, DATE) and (j + 1 == len(skyline) or getattr(node, DATE) < skyline[j + 1]):
                break
        node.add_feature(MODEL_ID, j)
        children = list(node.children)
        for child in children:
            if j < len(skyline) and getattr(child, DATE) > skyline[j + 1]:
                new_child_dist = getattr(child, DATE) - skyline[j + 1]
                skyline_node = node.add_child(dist=child.dist - new_child_dist)
                node.remove_child(child)
                skyline_node.add_child(child, dist=new_child_dist)
                skyline_nodes.append(skyline_node)
            annotate_skyline(child, j)

    for tree in forest:
        annotate_node_skyline(tree, 0)

    return skyline_nodes, len(skyline)


def remove_skyline(skyline_nodes):
    for n in skyline_nodes:
        parent = n.up
        for child in n.children:
            n.remove_child(child)
            parent.add_child(child, dist=child.dist + n.dist)
        parent.remove_child(n)

