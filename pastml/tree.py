import logging
import os
import re
from collections import Counter
from datetime import datetime

from Bio import Phylo
from ete3 import Tree, TreeNode

DATE = 'date'
DATE_CI = 'date_CI'

DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'
DATE_COMMENT_REGEX = '[&]date[=]"({})"'.format(DATE_REGEX)
CI_DATE_REGEX = '[&]CI_date[=]"({}) ({})"'.format(DATE_REGEX, DATE_REGEX)


def get_dist_to_root(tip):
    dist_to_root = 0
    n = tip
    while not n.is_root():
        dist_to_root += n.dist
        n = n.up
    return dist_to_root


def annotate_dates(forest, root_dates=None):
    if root_dates is None:
        root_dates = [0] * len(forest)
    for tree, root_date in zip(forest, root_dates):
        for node in tree.traverse('preorder'):
            if getattr(node, DATE, None) is None:
                if node.is_root():
                    node.add_feature(DATE, root_date if root_date else 0)
                else:
                    node.add_feature(DATE, getattr(node.up, DATE) + node.dist)


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
    try:
        return parse_nexus(tree_path)
    except:
        with open(tree_path, 'r') as f:
            nwks = f.read().replace('\n', '').split(';')
        if not nwks:
            raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
        return [read_tree(nwk + ';') for nwk in nwks[:-1]]


def read_tree(tree_path):
    for f in (3, 2, 5, 1, 0, 3, 4, 6, 7, 8, 9):
        try:
            return Tree(tree_path, format=f)
        except:
            continue
    raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))


def parse_nexus(tree_path):
    trees = []
    for nex_tree in read_nexus(tree_path):

        todo = [(nex_tree.root, None)]
        tree = None
        while todo:
            clade, parent = todo.pop()
            dist = 0
            try:
                dist = float(clade.branch_length)
            except:
                pass
            node = TreeNode(dist=dist, name=getattr(clade, 'name', None))
            if parent is None:
                tree = node
            else:
                parent.add_child(node)

            # Parse LSD2 dates and CIs
            date, ci = None, None
            comment = getattr(clade, 'comment', None)
            if comment is not None:
                date = next(iter(re.findall(DATE_COMMENT_REGEX, comment)), None)
            ci_attr = getattr(clade, 'branch_length' if not parent else 'confidence', None)
            if ci_attr is not None:
                ci = next(iter(re.findall(CI_DATE_REGEX, ci_attr)), None)

            if date is not None:
                try:
                    date = float(date)
                    node.add_feature(DATE, date)
                except:
                    pass
            if ci is not None:
                try:
                    ci = [float(_) for _ in ci]
                    node.add_feature(DATE_CI, ci)
                except:
                    pass
            todo.extend((c, node) for c in clade.clades)
        for n in tree.traverse('preorder'):
            date, ci = getattr(n, DATE, None), getattr(n, DATE_CI, None)
            if date is not None or ci is not None:
                for c in n.children:
                    if c.dist == 0:
                        if getattr(c, DATE, None) is None:
                            c.add_feature(DATE, date)
                        if getattr(c, DATE_CI, None) is None:
                            c.add_feature(DATE_CI, ci)
        for n in tree.traverse('postorder'):
            date, ci = getattr(n, DATE, None), getattr(n, DATE_CI, None)
            if not n.is_root() and n.dist == 0 and (date is not None or ci is not None):
                if getattr(n.up, DATE, None) is None:
                    n.up.add_feature(DATE, date)
                if getattr(n.up, DATE_CI, None) is None:
                    n.up.add_feature(DATE_CI, ci)

        # propagate dates up to the root if needed
        if getattr(tree, DATE, None) is None:
            dated_node = next((n for n in tree.traverse() if getattr(n, DATE, None) is not None), None)
            if dated_node:
                while dated_node != tree:
                    if getattr(dated_node.up, DATE, None) is None:
                        dated_node.up.add_feature(DATE, getattr(dated_node, DATE) - dated_node.dist)
                    dated_node = dated_node.up

        trees.append(tree)
    return trees


def read_nexus(tree_path):
    with open(tree_path, 'r') as f:
        nexus = f.read()
    # replace CI_date="2019(2018,2020)" with CI_date="2018 2020"
    nexus = re.sub(r'CI_date="({})\(({}),({})\)"'.format(DATE_REGEX, DATE_REGEX, DATE_REGEX), r'CI_date="\2 \3"',
                   nexus)
    temp = tree_path + '.{}.temp'.format(datetime.timestamp(datetime.now()))
    with open(temp, 'w') as f:
        f.write(nexus)
    trees = list(Phylo.parse(temp, 'nexus'))
    os.remove(temp)
    return trees