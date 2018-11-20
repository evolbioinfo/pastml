import logging
import random
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from ete3 import Tree

DATE = 'date'

REASONABLE_NUMBER_OF_TIPS = 15

CATEGORIES = 'categories'

NODE_SIZE = 'node_size'
NUM_TIPS_INSIDE = 'max_size'
NODE_NAME = 'node_name'

TIPS_INSIDE = 'in_tips'
TIPS_BELOW = 'all_tips'

EDGE_SIZE = 'edge_size'
EDGE_NAME = 'edge_name'
METACHILD = 'metachild'
FONT_SIZE = 'node_fontsize'


def get_dist_to_root(tip):
    dist_to_root = 0
    n = tip
    while not n.is_root():
        dist_to_root += n.dist
        n = n.up
    return dist_to_root


def date2years(d):
    if pd.notnull(d):
        first_jan_this_year = pd.datetime(year=d.year, month=1, day=1)
        day_of_this_year = d - first_jan_this_year
        first_jan_next_year = pd.datetime(year=d.year + 1, month=1, day=1)
        days_in_this_year = first_jan_next_year - first_jan_this_year
        return d.year + day_of_this_year / days_in_this_year
    else:
        return None


def date_tips(tree, date_df):
    """
    Adds dates to the tips as 'date' attribute.

    :param tree: ete3.Tree
    :param date_df: a pandas.Series with tip ids as indices and dates as values
    :return: void, modifies the initial tree
    """
    date_df.index = date_df.index.map(str)
    dated_tips, undated_tips = [], []
    min_date, max_date = None, None
    for tip in tree:
        if tip.name in date_df.index:
            date = date2years(date_df.loc[tip.name])
            if date is not None:
                date = int(date)
                tip.add_feature(DATE, date)
                dated_tips.append(tip)
                min_date = min(min_date, date) if min_date is not None else date
                max_date = max(max_date, date) if max_date is not None else date
            else:
                undated_tips.append(tip)
        else:
            undated_tips.append(tip)

    if len(dated_tips) < len(undated_tips):
        for tip in dated_tips:
            tip.del_feature(DATE)
        raise ValueError('Too few dates are provided (only {:.2f}% of tips are dated)!'
                         .format(100 * len(dated_tips) / (len(dated_tips) + len(undated_tips))))

    return min_date, max_date


def name_tree(tree):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter((_.name for _ in tree.traverse() if _.name))
    i = 0
    for node in tree.traverse('postorder'):
        if not node.name or existing_names[node.name] > 1:
            name = 'ROOT' if node.is_root() else None
            while not name or name in existing_names:
                name = '{}_{}'.format('tip' if node.is_leaf() else ('ROOT' if node.is_root() else 'node'), i)
                i += 1
            node.name = name


def collapse_zero_branches(tree):
    num_collapsed = 0
    for n in list(tree.traverse('postorder')):
        for child in list(n.children):
            if not child.is_leaf() and child.dist <= 0:
                n.remove_child(child)
                for grandchild in child.children:
                    n.add_child(grandchild)
                num_collapsed += 1
            if child.is_leaf() and child.dist < 0:
                child.dist = 0
    logging.getLogger('pastml').debug('Collapsed {} zero branches.'.format(num_collapsed))


def remove_certain_leaves(tr, to_remove=lambda node: False):
    """
    Removes all the branches leading to leaves identified positively by to_remove function.
    :param tr: the tree of interest (ete3 Tree)
    :param to_remove: a method to check is a leaf should be removed.
    :return: void, modifies the initial tree.
    """

    tips = [tip for tip in tr if to_remove(tip)]
    for node in tips:
        if node.is_root():
            return None
        parent = node.up
        parent.remove_child(node)
        # If the parent node has only one child now, merge them.
        if len(parent.children) == 1:
            brother = parent.children[0]
            brother.dist += parent.dist
            if parent.is_root():
                brother.up = None
                tr = brother
            else:
                grandparent = parent.up
                grandparent.remove_child(parent)
                grandparent.add_child(brother)
    return tr


def read_tree(tree_path):
    for f in (3, 2, 5, 1, 0, 3, 4, 6, 7, 8, 9):
        try:
            return Tree(tree_path, format=f)
        except:
            continue
    raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))


def sum_len_values(kv_dict):
    return sum(len(_) for _ in kv_dict.values())


def compress_tree(tree, column2states, can_merge_diff_sizes=True, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS):
    for n in tree.traverse('postorder'):
        n.add_feature(TIPS_INSIDE, defaultdict(list))
        n.add_feature(TIPS_BELOW, defaultdict(list))
        if n.is_leaf():
            getattr(n, TIPS_INSIDE)[getattr(n, DATE, 0)].append(n.name)
            getattr(n, TIPS_BELOW)[getattr(n, DATE, 0)].append(n.name)
        else:
            for _ in n:
                getattr(n, TIPS_BELOW)[getattr(_, DATE, 0)].append(_.name)

    one_column = next(iter(column2states.keys())) if len(column2states) == 1 else None

    if one_column:
        def get_states(n):
            value = getattr(n, one_column, [])
            if not isinstance(value, list):
                value = [value]
            return tuple(value)
    else:
        def get_states(n):
            return tuple(((column, getattr(n, column))
                          for column in column2states.keys() if not isinstance(getattr(n, column, []), list)))

    collapse_vertically(tree, get_states)
    remove_mediators(tree, get_states)

    for n in tree.traverse():
        n.add_feature(NUM_TIPS_INSIDE, sum_len_values(getattr(n, TIPS_INSIDE)))
        n.add_feature(TIPS_INSIDE, [getattr(n, TIPS_INSIDE)])
        n.add_feature(TIPS_BELOW, [getattr(n, TIPS_BELOW)])

    get_bin = lambda _: _
    collapse_horizontally(tree, get_states, get_bin)

    if can_merge_diff_sizes and len(tree) > tip_size_threshold:
        get_bin = lambda _: int(np.log10(max(1, _)))

        logging.getLogger('pastml').debug('Allowed merging nodes of different sizes.')
        collapse_horizontally(tree, get_states, get_bin)

    if len(tree) > tip_size_threshold:
        for n in tree.traverse():
            n.add_feature(NUM_TIPS_INSIDE, sum(sum_len_values(_) for _ in getattr(n, TIPS_INSIDE))
                          / len(getattr(n, TIPS_INSIDE)))

        def get_tsize(n):
            res = getattr(n, NUM_TIPS_INSIDE) * len(getattr(n, TIPS_INSIDE))
            while n.up:
                n = n.up
                res *= len(getattr(n, TIPS_INSIDE))
            return res

        tip_sizes = [get_tsize(_) for _ in tree]
        if len(tip_sizes) > tip_size_threshold:
            threshold = sorted(tip_sizes)[-tip_size_threshold]
            logging.getLogger('pastml').debug('Set tip size threshold to {}.'.format(threshold))
            remove_small_tips(tree, to_be_removed=lambda _: get_tsize(_) <= threshold)
            remove_mediators(tree, get_states)

            collapse_horizontally(tree, get_states, get_bin)
    return tree


def collapse_horizontally(tree, get_states, tips2bin):
    config_cache = {}

    def get_configuration(n):
        if n not in config_cache:
            config_cache[n] = (len(getattr(n, TIPS_INSIDE)),
                               (tips2bin(getattr(n, NUM_TIPS_INSIDE)),
                                get_states(n),
                                tuple(sorted([get_configuration(_) for _ in n.children]))))
        return config_cache[n]

    collapsed_configurations = 0

    for n in tree.traverse('postorder'):
        config2children = defaultdict(list)
        for _ in n.children:
            config2children[get_configuration(_)[1]].append(_)
        for children in (_ for _ in config2children.values() if len(_) > 1):
            collapsed_configurations += 1
            children = sorted(children, key=lambda _: getattr(_, DATE, 0))
            child = children[0]
            for c in children[1:]:
                n.remove_child(c)
            child.add_feature(METACHILD, True)

            tips_inside, tips_below = [], []
            for _ in children:
                tips_inside.extend(getattr(_, TIPS_INSIDE))
                tips_below.extend(getattr(_, TIPS_BELOW))

            child.add_feature(TIPS_INSIDE, tips_inside)
            child.add_feature(TIPS_BELOW, tips_below)
            child.add_feature(NUM_TIPS_INSIDE, sum(sum_len_values(_) for _ in getattr(child, TIPS_INSIDE))
                              / len(getattr(child, TIPS_INSIDE)))
            child.add_feature(DATE, min(getattr(_, DATE, 0) for _ in children))
            if child in config_cache:
                config_cache[child] = (len(getattr(child, TIPS_INSIDE)), config_cache[child][1])

    logging.getLogger('pastml').debug('Collapsed {} sets of equivalent configurations horizontally.'.format(collapsed_configurations))


def remove_small_tips(tree, to_be_removed):
    num_removed = 0
    changed = True
    while changed:
        changed = False
        for l in tree.get_leaves():
            parent = l.up
            if parent and to_be_removed(l):
                num_removed += 1
                parent.remove_child(l)
                parent.add_feature(DATE,
                                   min(min(_.keys()) if _ else 200000 for _ in getattr(parent, TIPS_INSIDE))
                                   if getattr(parent, TIPS_INSIDE) else 200000)
                if parent.children:
                    parent.add_feature(DATE, min(getattr(parent, DATE), min(getattr(_, DATE) for _ in parent.children)))
                changed = True
    logging.getLogger('pastml').debug('Recursively removed {} tips of size smaller or equal to the threshold.'.format(num_removed))


def merge_features(main_node, nodes, features, op, default_value=0):
    for feature in features:
        main_node.add_feature(feature, op([getattr(node, feature, default_value) for node in nodes]))


def collapse_vertically(tree, get_states):
    """
    Collapses a child node into its parent if they are in the same state.
    :param get_states: a function that returns a set of node states
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """
    num_collapsed = 0
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            continue

        states = get_states(n)
        children = list(n.children)
        for child in children:
            # merge the child into this node if their states are the same
            if get_states(child) == states:
                for date, tip_names in getattr(child, TIPS_INSIDE).items():
                    getattr(n, TIPS_INSIDE)[date].extend(tip_names)

                n.remove_child(child)
                grandchildren = list(child.children)
                for grandchild in grandchildren:
                    n.add_child(grandchild)
                num_collapsed += 1
    logging.getLogger('pastml').debug('Collapsed vertically {} internal nodes where there was no state change.'.format(num_collapsed))


def remove_mediators(tree, get_states):
    """
    Removes intermediate nodes that are just mediators between their parent and child states.
    :param get_states: a function that returns a set of node states
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """
    num_removed = 0
    for n in tree.traverse('postorder'):
        if getattr(n, METACHILD, False) or n.is_leaf() or len(n.children) > 1 or not n.up:
            continue
        states = get_states(n)
        parent = n.up
        parent_states = get_states(parent)
        child = n.children[0]
        child_states = get_states(child)
        compatible = states == parent_states == child_states
        if not compatible:
            # if mediator has unresolved states, it should hesitate between the parent and the child:
            compatible = set(states) == set(parent_states) | set(child_states)
        if compatible:
            n_tips_inside = getattr(n, TIPS_INSIDE)[0] if isinstance(getattr(n, TIPS_INSIDE), list) \
                else getattr(n, TIPS_INSIDE)
            tips_inside = getattr(parent, TIPS_INSIDE)[0] if isinstance(getattr(parent, TIPS_INSIDE), list) \
                else getattr(parent, TIPS_INSIDE)
            for date, tip_names in n_tips_inside.items():
                tips_inside[date].extend(tip_names)

            parent.remove_child(n)
            parent.add_child(child)
            num_removed += 1
    if num_removed:
        logging.getLogger('pastml').debug("Removed {} internal nodes"
                      " with the state unresolved between their parent's and their only child's."
                      .format(num_removed))
