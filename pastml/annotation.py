import logging
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from pastml import col_name2cat, quote
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


class ForestStats(object):

    def __init__(self, forest, character=None, min_date=-np.inf, max_date=np.inf):
        self._forest = forest
        self._character = character
        self._min_interval_date = min_date
        self._max_interval_date = max_date
        self._min_tree_date, self._max_tree_date = None, None
        self._observed_frequencies, self._missing_data = None, None
        self._n_trees, self._n_nodes, self._n_tips, self._n_zero_nodes, self._n_zero_tips, \
            self._length, self._avg_dist, self._n_polytomies \
            = None, None, None, None, None, None, None, None

    @property
    def min_interval_date(self):
        """
        Returns the lower bound (inclusive) of the time interval that is considered.

        :return: the lower bound of the time interval.
        """
        return self._min_interval_date

    @property
    def max_interval_date(self):
        """
        Returns the upper bound (exclusive) of the time interval that is considered.

        :return: the upper bound of the time interval.
        """
        return self._max_interval_date

    @property
    def min_tree_date(self):
        """
        Returns the minimal root tree among the forest trees.
        If the interval is given, this date will be corrected by the interval
        (and set to np.inf if all the trees are outside the interval).

        :return: the minimal root tree among the forest trees.
        """

        def analyse_tree(tree):
            return max(getattr(tree, DATE) - tree.dist, self._min_interval_date) \
                if self.tree_within_bounds(tree) else np.inf

        if self._min_tree_date is None:
            self._min_tree_date = min(analyse_tree(_) for _ in self._forest)
        return self._min_tree_date

    @property
    def max_tree_date(self):
        """
        Returns the minimal root tree among the forest trees.
        If the interval is given, this date will be corrected by the interval
        (and set to -np.inf if all the trees are outside the interval).

        :return: the minimal root tree among the forest trees.
        """

        def analyse_tree(tree):
            return min(max(getattr(tip, DATE) for tip in tree), self._max_interval_date) \
                if self.tree_within_bounds(tree) else -np.inf

        if self._max_tree_date is None:
            self._max_tree_date = max(analyse_tree(tree) for tree in self._forest)
        return self._max_tree_date

    def node_within_bounds(self, node):
        return self._min_interval_date <= getattr(node, DATE) < self._max_interval_date

    def tree_within_bounds(self, tree):
        return getattr(tree, DATE) - tree.dist < self._max_interval_date \
            and max(getattr(tip, DATE) for tip in tree) >= self._min_interval_date

    def branch_within_bounds(self, node):
        """
        Checks if the node's branch is (at least partially) within the time interval.

        :param node: tree node
        :return: whether the branch is (at least partially) within the time interval.
        """
        date = getattr(node, DATE)
        up_date = date - node.dist
        return up_date < self._max_interval_date and date >= self._min_interval_date

    @property
    def n_nodes(self):
        """
        Returns the total number of nodes in all the forest trees.
        If a time interval is given, only nodes within it will be counted.

        :return: number of nodes
        """
        if self._n_nodes is None:
            # We check for branch and not for node as if the branch is within bounds and not node,
            # it creates a "new tip"
            self._n_nodes = sum(sum(1 for _ in tree.traverse()
                                    if self.branch_within_bounds(_)) for tree in self._forest)
        return self._n_nodes

    @property
    def n_trees(self):
        """
        Returns the number of trees in the forest.
        If a time interval is given, only trees that intersect with it will be counted
        (an initial tree might become split into several subtrees if its top part is cut).

        :return: number of trees
        """
        if self._n_trees is None:
            self._n_trees = 0
            for tree in self._forest:
                if max(getattr(tip, DATE) for tip in tree) < self._min_interval_date:
                    continue
                todo = [tree]
                while todo:
                    n = todo.pop()
                    if self.branch_within_bounds(n):
                        self._n_trees += 1
                    else:
                        todo.extend(n.children)
        return self._n_trees

    @property
    def n_zero_nodes(self):
        """
        Returns the total number of nodes with zero branch length in the trees of the forest.
        If a time interval is given, only the tree parts that intersect with it will be considered
        and pending branches will be considered as tips.

        :return: number of zero-branch nodes
        """
        if self._n_zero_nodes is None:
            self._n_zero_nodes = 0
            todo = list(self._forest)
            while todo:
                n = todo.pop()
                date = getattr(n, DATE)
                if self.branch_within_bounds(n):
                    up_date = date - n.dist
                    if min(self._max_interval_date, date) - max(self._min_interval_date, up_date) == 0:
                        self._n_zero_nodes += 1
                if date < self._max_interval_date:
                    todo.extend(n.children)
        return self._n_zero_nodes

    @property
    def n_zero_tips(self):
        """
        Returns the total number of tips with zero branch length in the trees of the forest.
        If a time interval is given, only the tree parts that intersect with it will be considered
        and pending branches will be considered as tips.

        :return: number of zero-branch tips
        """
        if self._n_zero_tips is None:
            self._n_zero_tips = 0
            todo = list(self._forest)
            while todo:
                n = todo.pop()
                date = getattr(n, DATE)
                if (n.dist == 0 and n.is_leaf()) or date == self._min_interval_date:
                    self._n_zero_tips += 1
                if date < self._max_interval_date:
                    todo.extend(n.children)
        return self._n_zero_tips

    @property
    def n_tips(self):
        """
        Calculate the number of tips in the forest. If time interval is specified, and this time interval cuts the tree,
        pending branches (cut by the interval) will be counted as tips.

        :return: number of tips
        """
        if self._n_tips is None:
            self._n_tips = 0
            todo = list(self._forest)
            while todo:
                n = todo.pop()
                date = getattr(n, DATE)
                if self.branch_within_bounds(n):
                    if date >= self._max_interval_date or n.is_leaf():
                        self._n_tips += 1
                if date < self._max_interval_date:
                    todo.extend(n.children)
        return self._n_tips

    @property
    def length(self):
        """
        Calculate the length of the tree parts of the forest that are within the time interval.

        :return: length
        """
        if self._length is None:
            self._length = 0
            todo = list(self._forest)
            while todo:
                n = todo.pop()
                date = getattr(n, DATE)
                if self.branch_within_bounds(n):
                    up_date = date - n.dist
                    self._length += min(date, self._max_interval_date) - max(up_date, self._min_interval_date)
                if date < self._max_interval_date:
                    todo.extend(n.children)
        return self._length

    @property
    def avg_dist(self):
        """
        Calculates the average branch distance among non-zero tree branches of the forest trees.
        If time interval is specified, and this time interval cuts the tree,
        only the branch parts that are within the interval will be considered.

        :return: average branch distance among non-zero tree branches
        """
        if self._avg_dist is None:
            n_non_zero_nodes = self.n_nodes - self.n_zero_nodes
            self._avg_dist = self.length / n_non_zero_nodes if n_non_zero_nodes else 0
        return self._avg_dist

    @property
    def n_polytomies(self):
        """
        Returns the total number of unresolved internal nodes (i.e., with more than two children)
        in the trees of the forest.
        If a time interval is given, only the tree parts that intersect with it will be considered.

        :return: number of polytomies
        """
        if self._n_polytomies is None:
            self._n_polytomies = \
                sum(sum(1 for _ in tree.traverse() if len(_.children) > 2 and self.node_within_bounds(_))
                    for tree in self._forest)
        return self._n_polytomies

    def __str__(self):
        return '\n=============FOREST STATISTICS{}===================\n' \
               '\tnumber of trees:\t{}\n' \
               '\ttime period covered by trees:\t{:g}-{:g}\n' \
               '\tnumber of tips:\t{}\n' \
               '\tnumber of zero-branch tips:\t{}\n' \
               '\ttotal number of nodes:\t{}\n' \
               '\tnumber of polytomies:\t{}\n' \
               '\taverage non-zero branch length:\t{:.5f}\n' \
               '\tobserved frequencies for {}:{}{}' \
            .format(' ({:g}-{:g})'.format(self.min_interval_date, self.max_interval_date)
                    if self.min_interval_date > -np.inf or self.max_interval_date < np.inf else '',
                    self.n_trees,
                    self.min_tree_date, self.max_tree_date,
                    self.n_tips, self.n_zero_tips,
                    self.n_nodes,
                    self.n_polytomies,
                    self.avg_dist,
                    self._character,
                    ''.join('\n\t\t{}:\t{:.6f}'.format(state, self.observed_frequencies[state])
                            for state in sorted(self.observed_frequencies.keys())),
                    '\n\t\tfraction of missing data:\t{:.6f}'.format(self.missing_data) if self.missing_data else ''
                    )

    @property
    def observed_frequencies(self):
        """
        Returns a dictionary with the frequencies of character states observed at tips of the forest trees.
        If the time interval is specified, only tips (actual, not created by cutting branches by skyline) within it
        will be considered.

        :return: dict{state: its frequency}
        """
        if self._observed_frequencies is None:
            self._calculate_observed_frequencies()
        return self._observed_frequencies

    @property
    def missing_data(self):
        """
        Returns the proportion of tips with unknown states in the forest trees.
        If the time interval is specified, only tips (actual, not created by cutting branches by skyline) within it
        will be considered.

        :return: proportion of tips with unknown states (between 0 and 1)
        """
        if self._observed_frequencies is None:
            self._calculate_observed_frequencies()
        return self._missing_data

    def _calculate_observed_frequencies(self):
        self._missing_data = 0
        total_data = 0
        self._observed_frequencies = defaultdict(lambda: 0)
        for tree in self._forest:
            for _ in tree:
                if self.node_within_bounds(_):
                    state = getattr(_, self._character, set())
                    total_data += 1
                    if state:
                        num_node_states = len(state)
                        for _ in state:
                            self._observed_frequencies[_] += 1. / num_node_states
                    else:
                        self._missing_data += 1
        for _ in self._observed_frequencies.keys():
            self._observed_frequencies[_] /= (total_data - self._missing_data)
        if total_data:
            self._missing_data /= total_data


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
                                     quote(unknown_columns),
                                     'is' if len(unknown_columns) == 1 else 'are',
                                     quote(new_columns)))
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
