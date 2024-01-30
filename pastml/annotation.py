import logging
from collections import Counter, defaultdict

import pandas as pd
import numpy as np

from pastml import col_name2cat, datetime2numeric
from pastml.tree import DATE, DATE_CI
from pastml.visualisation.cytoscape_manager import DIST_TO_ROOT_LABEL, DATE_LABEL


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


def _quote(str_list):
    return ', '.join('"{}"'.format(_) for _ in str_list) if str_list is not None else ''


def parse_date(d):
    try:
        return float(d)
    except ValueError:
        try:
            return datetime2numeric(pd.to_datetime(d, infer_datetime_format=True))
        except ValueError:
            raise ValueError('Could not infer the date format for root date "{}", please check it.'
                             .format(d))


def annotate_dates(forest, root_dates=None):
    # Process root dates
    if root_dates is not None:
        root_dates = [parse_date(d) for d in (root_dates if isinstance(root_dates, list) else [root_dates])]
        if 1 < len(root_dates) < len(forest):
            raise ValueError('{} trees are given, but only {} root dates.'.format(len(forest), len(root_dates)))
        elif 1 == len(root_dates):
            root_dates *= len(forest)
    age_label = DIST_TO_ROOT_LABEL \
        if (root_dates is None and not next((True for root in forest if getattr(root, DATE, None) is not None), False)) \
        else DATE_LABEL
    if root_dates is None:
        root_dates = [0] * len(forest)
    for tree, root_date in zip(forest, root_dates):
        for node in tree.traverse('preorder'):
            if getattr(node, DATE, None) is None:
                if node.is_root():
                    node.add_feature(DATE, root_date if root_date else 0)
                else:
                    node.add_feature(DATE, getattr(node.up, DATE) + node.dist)
            else:
                node.add_feature(DATE, float(getattr(node, DATE)))
            ci = getattr(node, DATE_CI, None)
            if ci and not isinstance(ci, list) and not isinstance(ci, tuple):
                node.del_feature(DATE_CI)
                if isinstance(ci, str) and '|' in ci:
                    try:
                        node.add_feature(DATE_CI, [float(_) for _ in ci.split('|')])
                    except:
                        pass
    return age_label


def annotate_forest(forest, columns=None, data=None, data_sep='\t', id_index=0,
                    unknown_treshold=0.9, state_threshold=.75):
    logger = logging.getLogger('pastml')

    if not columns and not data:
        raise ValueError("If you don't provide the metadata file(s), "
                         "you need to provide an annotated tree and specify the columns argument, "
                         "which will be used to look for character annotations in your input tree.")

    if columns and isinstance(columns, str):
        columns = [columns]

    if data and isinstance(data, str):
        data = [data]

    column2annotated = Counter()
    column2states = defaultdict(set)

    if columns:
        columns = [col_name2cat(column) for column in columns]
    new_columns = []

    node_names = set.union(*[{n.name for n in root.traverse() if n.name} for root in forest])

    if data:
        unknown_columns = set(columns) if columns else None
        for data_table in data:
            df = pd.read_csv(data_table, sep=data_sep, index_col=id_index, header=0, dtype=str)
            # Strip whitespaces and quotes around the index values
            df.index = df.index.map(lambda _: str(_).strip(" ").strip("'").strip('"'))
            logger.debug('Read the annotation file {}.'.format(data_table))
            df.columns = [col_name2cat(column) for column in df.columns]
            if columns:
                unknown_columns -= set(df.columns)
                found_columns = [c for c in df.columns if c in columns]
                if not found_columns:
                    continue
                df = df[found_columns]
            else:
                new_columns.extend(df.columns)

            df_index_names = set(df.index)
            common_ids = node_names & df_index_names

            filtered_df = df.loc[list(common_ids), :]
            if not filtered_df.shape[0]:
                tip_name_representatives = []
                for _ in forest[0].iter_leaves():
                    if len(tip_name_representatives) < 3:
                        tip_name_representatives.append(_.name)
                    else:
                        break
                raise ValueError(
                    'Your tree tip names (e.g. {}) do not correspond to annotation id column values (e.g. {}). '
                    'Check your annotation file.'
                    .format(', '.join(tip_name_representatives),
                            ', '.join(list(df_index_names)[: min(len(df_index_names), 3)])))
            logger.debug('Checked that (at least some of) tip names correspond to annotation file index.')

            preannotate_forest(forest, df=df)
            for c in df.columns:
                column2states[c] |= {_ for _ in df[c].unique() if pd.notnull(_) and _ != ''}

        if unknown_columns:
            raise ValueError('{} of the specified columns ({}) {} not found among the annotation columns: {}.'
                             .format('One' if len(unknown_columns) == 1 else 'Some',
                                     _quote(unknown_columns),
                                     'is' if len(unknown_columns) == 1 else 'are',
                                     _quote(new_columns)))
        if not columns:
            columns = new_columns

    num_tips = 0

    column2annotated_states = defaultdict(set)
    for root in forest:
        for n in root.traverse():
            for c in columns:
                vs = getattr(n, c, set())
                column2states[c] |= vs
                column2annotated_states[c] |= vs
                if vs:
                    column2annotated[c] += 1
            if n.is_leaf():
                num_tips += 1

    if column2annotated:
        c, num_annotated = min(column2annotated.items(), key=lambda _: _[1])
    else:
        c, num_annotated = columns[0], 0
    percentage_unknown = (num_tips - num_annotated) / num_tips
    if percentage_unknown >= unknown_treshold:
        raise ValueError('{:.1f}% of tip annotations for character "{}" are unknown, '
                         'not enough data to infer ancestral states. '
                         '{}'
                         .format(percentage_unknown * 100, c,
                                 'Check your annotation file and if its ids correspond to the tree tip/node names.'
                                 if data
                                 else 'You tree file should contain character state annotations, '
                                      'otherwise consider specifying a metadata file.'))
    if state_threshold < 1:
        c, states = min(column2annotated_states.items(), key=lambda _: len(_[1]))
        if len(states) > num_tips * state_threshold:
            raise ValueError('Character "{}" has {} unique states annotated in this tree: {}, '
                             'which is too much to infer on a {} with only {} tips. '
                             'Make sure the character you are analysing is discrete, and if yes use a larger tree.'
                             .format(c, len(states), states, 'tree' if len(forest) == 1 else 'forest', num_tips))

    return columns, {c: np.array(sorted(states)) for c, states in column2states.items()}

