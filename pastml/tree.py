import logging
from collections import Counter

import pandas as pd
from ete3 import Tree

LEVEL = 'level'

DATE = 'date'


def get_dist_to_root(tip):
    dist_to_root = 0
    n = tip
    while not n.is_root():
        dist_to_root += n.dist
        n = n.up
    return dist_to_root


def annotate_dates(forest, date_feature=DATE, level_feature=LEVEL, root_dates=None):
    if root_dates is None:
        root_dates = [0] * len(forest)
    for tree, root_date in zip(forest, root_dates):
        for node in tree.traverse('preorder'):
            if node.is_root():
                node.add_feature(date_feature, root_date if root_date else 0)
                node.add_feature(level_feature, 0)
            else:
                node.add_feature(date_feature, getattr(node.up, date_feature) + node.dist)
                node.add_feature(level_feature, getattr(node.up, level_feature) + 1)


def name_tree(tree, suffix=""):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter((_.name for _ in tree.traverse() if _.name))
    if sum(1 for _ in tree.traverse()) == len(existing_names):
        return
    i = 0
    existing_names = Counter()
    for node in tree.traverse('preorder'):
        name = node.name if node.is_leaf() else ('root{}'.format(suffix) if node.is_root() else None)
        while name is None or name in existing_names:
            name = '{}{}{}'.format('t' if node.is_leaf() else 'n', i, suffix)
            i += 1
        node.name = name
        existing_names[name] += 1


def collapse_zero_branches(forest, features_to_be_merged=None):
    """
    Collapses zero branches in tre tree/forest.

    :param forest: tree or list of trees
    :type forest: ete3.Tree or list(ete3.Tree)
    :param features_to_be_merged: list of features whose values are to be merged
        in case the nodes are merged during collapsing
    :type features_to_be_merged: list(str)
    :return: void
    """
    num_collapsed = 0

    if features_to_be_merged is None:
        features_to_be_merged = []

    for tree in forest:
        for n in list(tree.traverse('postorder')):
            zero_children = [child for child in n.children if not child.is_leaf() and child.dist <= 0]
            if not zero_children:
                continue
            for feature in features_to_be_merged:
                feature_intersection = set.intersection(*(getattr(child, feature, set()) for child in zero_children)) \
                                       & getattr(n, feature, set())
                if feature_intersection:
                    value = feature_intersection
                else:
                    value = set.union(*(getattr(child, feature, set()) for child in zero_children)) \
                            | getattr(n, feature, set())
                if value:
                    n.add_feature(feature, value)
            for child in zero_children:
                n.remove_child(child)
                for grandchild in child.children:
                    n.add_child(grandchild)
            num_collapsed += len(zero_children)
    if num_collapsed:
        logging.getLogger('pastml').debug('Collapsed {} internal zero branches.'.format(num_collapsed))


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


def read_forest(tree_path):
    with open(tree_path, 'r') as f:
        nwks = f.read().replace('\n', '').split(';')
    if not nwks:
        raise ValueError('Could not find any trees (in newick format) in the file {}.'.format(tree_path))
    return [read_tree(nwk + ';') for nwk in nwks[:-1]]


def read_tree(tree_path):
    for f in (3, 2, 5, 1, 0, 3, 4, 6, 7, 8, 9):
        try:
            return Tree(tree_path, format=f)
        except:
            continue
    raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))

