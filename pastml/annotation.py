import logging
from collections import defaultdict

import pandas as pd
import numpy as np

from pastml import MODEL_ID, SKYLINE
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


def annotate_skyline(forest, skyline, first_column, column2states, skyline_mapping=None):
    min_date, max_date = min(getattr(tree, DATE) for tree in forest), \
                         max(max(getattr(_, DATE) for _ in tree) for tree in forest)
    if not skyline:
        for col in column2states.keys():
            column2states[col] = [column2states[col]]
        return None, None, [min_date]
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
        for col in column2states.keys():
            column2states[col] = [column2states[col]]
        return None, None, [min_date]
    elif skyline[0] > min_date:
        skyline = [min_date] + skyline

    logging.getLogger('pastml').debug('The tree(s) cover the period between {} and {}, '
                                      'the skyline intervals start at the following dates: {}.'
                                      .format(min_date, max_date, ', '.join(str(_) for _ in skyline)))

    if skyline_mapping:
        df = pd.read_csv(skyline_mapping, sep='\t')
        expected_columns = [first_column] + [str(_) for _ in skyline[1:]]
        try:
            converted_columns = {float(_) for _ in df.columns if _ != first_column}
        except:
            converted_columns = set()
        if first_column not in df.columns or set(skyline[1:]) != converted_columns:
            raise ValueError('Skyline mapping is specified in {} but instead of containing columns {} it contains {}'
                             .format(skyline_mapping, ', '.join(expected_columns), df.columns))

        def get_states(state, source_col, target_col):
            target_states = {_ for _ in df.loc[df[source_col] == state, target_col].unique() if not pd.isna(_)}
            if not target_states:
                raise ValueError('Could not find the states corresponding to {} of {} in {}'
                                 .format(state, source_col, target_col))
            return target_states

        mapping = {}
        prev_states = None
        prev_col = None
        all_states = []
        for i, year in enumerate(skyline[1:], start=0):
            col = next(_ for _ in df.columns if _ != first_column and float(_) == year)
            states = {_ for _ in df[col].unique() if not pd.isna(_)}
            if i > 0:
                mapping[(i - 1, i)] = {state: get_states(state, prev_col, col) for state in prev_states}
                mapping[(i, i - 1)] = {state: get_states(state, col, prev_col) for state in states}
            prev_col, prev_states = col, states
            all_states.append(np.array(sorted(prev_states)))

        states = {_ for _ in df[first_column].unique() if not pd.isna(_)}
        mapping[(len(skyline) - 2, len(skyline) - 1)] = {state: get_states(state, prev_col, first_column) for state in prev_states}
        mapping[(len(skyline) - 1, len(skyline) - 2)] = {state: get_states(state, first_column, prev_col) for state in states}
        all_states.append(np.array(sorted(states)))

        skyline_mapping = {}
        skyline_mapping_pars = mapping
        for (i, j), state2states in mapping.items():
            mapping_ij = np.zeros(shape=(len(all_states[i]), len(all_states[j])), dtype=float)
            skyline_mapping[(i, j)] = mapping_ij
            for (from_i,  from_state) in enumerate(all_states[i]):
                to_states = state2states[from_state]
                for (to_j, to_state) in enumerate(all_states[j]):
                    if to_state in to_states:
                        mapping_ij[from_i, to_j] = 1

        column2states[first_column] = all_states
    else:
        skyline_mapping = None
        skyline_mapping_pars = None
        column2states[first_column] = [column2states[first_column]] * len(skyline)
    skyline_mapping = {first_column: skyline_mapping}
    skyline_mapping_pars = {first_column: skyline_mapping_pars}
    for col in column2states.keys():
        if col != first_column:
            column2states[col] = [column2states[col]] * len(skyline)
            skyline_mapping[col] = None
            skyline_mapping_pars[col] = None

    i2state_set = dict(zip(range(len(skyline)), (set(_) for _ in column2states[first_column])))

    def annotate_node_skyline(node, i):
        n_sk = 0
        for j in range(i, len(skyline)):
            if skyline[j] <= getattr(node, DATE) and (j + 1 == len(skyline) or getattr(node, DATE) < skyline[j + 1]):
                break
        node.add_feature(MODEL_ID, j)
        node_states = getattr(node, first_column, set())
        if node_states - i2state_set[j]:
            raise ValueError('Node {} has states that do not correspond to its skyline date ({}): {}.'
                             .format(node.name, getattr(node, DATE), ', '.join(node_states - i2state_set[j])))
        children = list(node.children)
        for child in children:
            if (j + 1) < len(skyline) and getattr(child, DATE) > skyline[j + 1]:
                new_child_dist = getattr(child, DATE) - skyline[j + 1]
                skyline_node_up = node.add_child(name='{}_{}_skyline_up'.format(child.name, skyline[j + 1]),
                                                 dist=child.dist - new_child_dist)
                skyline_node_up.add_feature(DATE, skyline[j + 1])
                skyline_node_up.add_feature(SKYLINE, True)
                skyline_node_up.add_feature(MODEL_ID, j)
                skyline_node_down = skyline_node_up.add_child(name='{}_{}_skyline_down'.format(child.name, skyline[j + 1]),
                                                              dist=0)
                skyline_node_down.add_feature(DATE, skyline[j + 1])
                skyline_node_down.add_feature(SKYLINE, True)
                skyline_node_down.add_feature(MODEL_ID, j + 1)

                node.remove_child(child)
                skyline_node_down.add_child(child, dist=new_child_dist)
                n_sk += 1 + annotate_node_skyline(child, j + 1)
            else:
                n_sk += annotate_node_skyline(child, j)
        return n_sk

    n_skyline = sum(annotate_node_skyline(tree, 0) for tree in forest)

    logging.getLogger('pastml').debug('Created {} skyline nodes.'.format(n_skyline))

    return skyline_mapping, skyline_mapping_pars, skyline


def remove_skyline(forest):
    n_skyline = 0
    for tree in forest:
        for n in tree.traverse('postorder'):
            n.del_feature(MODEL_ID)
            if getattr(n, SKYLINE, False):
                parent = n.up
                children = list(n.children)
                for child in children:
                    n.remove_child(child)
                    parent.add_child(child, dist=child.dist + n.dist)
                parent.remove_child(n)
                n_skyline += 1
    if n_skyline:
        logging.getLogger('pastml').debug('Removed {} skyline nodes.'.format(n_skyline / 2))

