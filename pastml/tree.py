import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime

from Bio import Phylo
from ete3 import Tree, TreeNode
from ete3.parser.newick import NewickError
import pandas as pd
from pastml import datetime2numeric, PASTML_VERSION, _set_up_pastml_logger

POSTORDER = 'postorder'

INORDER = 'inorder'

PREORDER = 'preorder'

DATE = 'date'
DATE_CI = 'date_CI'

DATE_REGEX = r'[+-]*[\d]+[.\d]*(?:[e][+-][\d]+){0,1}'
DATE_COMMENT_REGEX = '[&,:]date[=]["]{{0,1}}({})["]{{0,1}}'.format(DATE_REGEX)
CI_DATE_REGEX_LSD = '[&,:]CI_date[=]["]{{0,1}}[{{]{{0,1}}({})\s*[,;]{{0,1}}\s*({})[}}]{{0,1}}["]{{0,1}}'.format(
    DATE_REGEX, DATE_REGEX)
CI_DATE_REGEX_PASTML = '[&,:]date_CI[=]["]{{0,1}}({})[|]({})["]{{0,1}}'.format(DATE_REGEX, DATE_REGEX)
COLUMN_REGEX_PASTML = '[&,]{column}[=]([^]^,]+)'

IS_POLYTOMY = 'polytomy'


def parse_date(d):
    try:
        return float(d)
    except ValueError:
        try:
            return datetime2numeric(pd.to_datetime(d, infer_datetime_format=True))
        except ValueError:
            raise ValueError('Could not infer the date format for root date "{}", please check it.'.format(d))


def get_dist_to_root(tip):
    dist_to_root = 0
    n = tip
    while not n.is_root():
        dist_to_root += n.dist
        n = n.up
    return dist_to_root


def annotate_dates(forest, root_dates=None, annotate_zeros=True):
    if root_dates is None:
        root_dates = [None] * len(forest)
    for tree, root_date in zip(forest, root_dates):
        for node in tree.traverse('preorder'):
            if getattr(node, DATE, None) is None:
                if node.is_root():
                    if root_date is not None or annotate_zeros:
                        node.add_feature(DATE, root_date if root_date else 0)
                else:
                    parent_date = getattr(node.up, DATE, None)
                    if parent_date is not None:
                        node.add_feature(DATE, parent_date + node.dist)
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


def name_tree(tree, suffix=""):
    """
    Names all the tree nodes that are not named or have non-unique names, with unique names.

    :param tree: tree to be named
    :type tree: ete3.Tree

    :return: void, modifies the original tree
    """
    existing_names = Counter()
    n_nodes = 0
    for _ in tree.traverse():
        n_nodes += 1
        if _.name:
            existing_names[_.name] += 1
            if '.polytomy_' in _.name:
                _.add_feature(IS_POLYTOMY, 1)
    if n_nodes == len(existing_names):
        return
    i = 0
    new_existing_names = Counter()
    for node in tree.traverse('preorder'):
        name_prefix = node.name if node.name and existing_names[node.name] < 10 \
            else '{}{}{}'.format('t' if node.is_leaf() else 'n', i, suffix)
        name = 'root{}'.format(suffix) if node.is_root() else name_prefix
        while name is None or name in new_existing_names:
            name = '{}{}{}'.format(name_prefix, i, suffix)
            i += 1
        node.name = name
        new_existing_names[name] += 1


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


def read_forest(tree_path, columns=None):
    roots = None
    try:
        roots = parse_nexus(tree_path, columns=columns)
    except:
        pass
    if not roots:
        with open(tree_path, 'r') as f:
            nwks = f.read().replace('\n', '').split(';')
        if not nwks:
            raise ValueError('Could not find any trees (in newick or nexus format) in the file {}.'.format(tree_path))
        roots = [read_tree(nwk + ';', columns) for nwk in nwks[:-1]]

    num_neg = 0
    for root in roots:
        for _ in root.traverse():
            if _.dist < 0:
                num_neg += 1
                _.dist = 0
    if num_neg:
        logging.getLogger('pastml').warning('Input tree{} contained {} negative branches: we put them to zero.'
                                            .format('s' if len(roots) > 0 else '', num_neg))
    logging.getLogger('pastml').debug('Read the tree{} {}.'.format('s' if len(roots) > 0 else '', tree_path))
    return roots


def read_tree(tree_path, columns=None):
    tree = None
    for f in (3, 2, 5, 0, 1, 4, 6, 7, 8, 9):
        try:
            tree = Tree(tree_path, format=f)
            break
        except NewickError:
            try:
                tree = Tree(tree_path, format=f, quoted_node_names=True)
                break
            except NewickError:
                continue
    if not tree:
        raise ValueError('Could not read the tree {}. Is it a valid newick?'.format(tree_path))
    if columns:
        for n in tree.traverse():
            for c in columns:
                vs = set(getattr(n, c).split('|')) if hasattr(n, c) else set()
                if vs:
                    n.add_feature(c, vs)
    return tree


def parse_nexus(tree_path, columns=None):
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
            name = getattr(clade, 'name', None)
            if not name:
                name = getattr(clade, 'confidence', None)
                if not isinstance(name, str):
                    name = None
            node = TreeNode(dist=dist, name=name)
            if parent is None:
                tree = node
            else:
                parent.add_child(node)

            # Parse LSD2 dates and CIs, and PastML columns
            date, ci = None, None
            columns2values = defaultdict(set)
            comment = getattr(clade, 'comment', None)
            if isinstance(comment, str):
                date = next(iter(re.findall(DATE_COMMENT_REGEX, comment)), None)
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
                if columns:
                    for column in columns:
                        values = \
                            set.union(*(set(_.split('|')) for _ in re.findall(COLUMN_REGEX_PASTML.format(column=column),
                                                                              comment)), set())
                        if values:
                            columns2values[column] |= values
            comment = getattr(clade, 'branch_length', None)
            if not ci and not parent and isinstance(comment, str):
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
            comment = getattr(clade, 'confidence', None)
            if ci is None and comment is not None and isinstance(comment, str):
                ci = next(iter(re.findall(CI_DATE_REGEX_LSD, comment)), None)
                if ci is None:
                    ci = next(iter(re.findall(CI_DATE_REGEX_PASTML, comment)), None)
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
            if columns2values:
                for c, vs in columns2values.items():
                    node.add_feature(c, vs)
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


def depth_first_traversal(node):
    yield node, PREORDER
    for i, child in enumerate(node.children):
        if i != 0:
            yield node, INORDER
        for _ in depth_first_traversal(child):
            yield _
    yield node, POSTORDER


def resolve_trees(column2states, forest):
    """
    Resolved polytomies based on state predictions:
    if a parent P in a state A has n children 2 <= m < n of which are in state B,
    we add a parent to these children (who becomes a child of P) in a state B
    at the distance of the oldest of the children.

    :param column2states: character to possible state mapping
    :type column2states: dict
    :param forest: a forest of trees of interest
    :type list(ete.Tree)
    :return: number of newly created nodes.
    :rtype: int
    """
    columns = sorted(column2states.keys())

    col2state2i = {c: dict(zip(states, range(len(states)))) for (c, states) in column2states.items()}

    def get_prediction(n):
        return '.'.join('-'.join(str(i) for i in sorted([col2state2i[c][_] for _ in getattr(n, c, set())]))
                        for c in columns)

    num_new_nodes = 0

    for tree in forest:
        todo = [tree]
        while todo:
            n = todo.pop()
            todo.extend(n.children)
            if len(n.children) > 2:
                state2children = defaultdict(list)
                for c in n.children:
                    state2children[get_prediction(c)].append(c)
                if len(state2children) > 1:
                    for state, children in state2children.items():
                        if group_children_if_needed(n, children, columns, state):
                            num_new_nodes += 1
    if num_new_nodes:
        logging.getLogger('pastml').debug(
            'Created {} new internal nodes while resolving polytomies'.format(num_new_nodes))
    else:
        logging.getLogger('pastml').debug('Could not resolve any polytomy')
    return num_new_nodes


def states_are_different(n1, n2, columns):
    for c in columns:
        if not getattr(n1, c, set()) & getattr(n2, c, set()):
            return True
    return False


def group_children_if_needed(n, children, columns, state):
    if len(children) <= 1:
        return False
    child = min(children, key=lambda _: _.dist)
    if not states_are_different(n, child, columns):
        return False
    dist = child.dist
    pol = n.add_child(dist=dist, name='{}.polytomy_{}'.format(n.name, state))
    pol.add_feature(IS_POLYTOMY, 1)
    c_date = getattr(child, DATE)
    pol.add_feature(DATE, c_date)
    n_ci = getattr(n, DATE_CI, None)
    c_ci = getattr(child, DATE_CI, None)
    pol.add_feature(DATE_CI, (None if not n_ci or not isinstance(n_ci, list)
                              else [n_ci[0],
                                    (c_ci[1] if c_ci and isinstance(c_ci, list) and len(c_ci) > 1
                                     else c_date)]))
    for c in columns:
        pol.add_feature(c, getattr(child, c))
    for c in children:
        n.remove_child(c)
        pol.add_child(c, dist=c.dist - dist)
    return True


def unresolve_trees(column2states, forest):
    """
    Unresolves polytomies whose states do not correspond to child states after likelihood recalculation.

    :param column2states: character to possible state mapping
    :type column2states: dict
    :param forest: a forest of trees of interest
    :type list(ete.Tree)
    :return: number of newly deleted nodes.
    :rtype: int
    """
    columns = sorted(column2states.keys())

    col2state2i = {c: dict(zip(states, range(len(states)))) for (c, states) in column2states.items()}

    def get_prediction(n):
        return '.'.join('-'.join(str(i) for i in sorted([col2state2i[c][_] for _ in getattr(n, c, set())]))
                        for c in columns)

    num_removed_nodes = 0
    num_new_nodes = 0

    def remove_node(n):
        parent = n.up
        for c in n.children:
            parent.add_child(c, dist=c.dist + n.dist)
        parent.remove_child(n)

    num_polytomies = 0

    for tree in forest:
        for n in tree.traverse('postorder'):
            if getattr(n, IS_POLYTOMY, False):
                num_polytomies += 1

                state2children = defaultdict(list)
                n_children = list(n.children)
                for c in n_children:
                    state2children[get_prediction(c)].append(c)
                parent = n.up

                # if the state is the same as all the child states, it's still a good polytomy resolution
                if len(state2children) == 1 and not states_are_different(n, n_children[0], columns):
                    # Just need to check that it is not the same state as the parent (then we don't need this polytomy)
                    if not states_are_different(n, parent, columns):
                        num_removed_nodes += 1
                        remove_node(n)
                    continue

                num_removed_nodes += 1
                remove_node(n)

                # now let's try to create new polytomies above
                above_state2children = defaultdict(list)
                for c in parent.children:
                    state2children[get_prediction(c)].append(c)
                for state, children in above_state2children.items():
                    if len(children) <= 1:
                        continue
                    if set(children) != set(n_children):
                        if group_children_if_needed(parent, children, columns, state):
                            num_new_nodes += 1
    if num_removed_nodes:
        logging.getLogger('pastml').debug(
            'Removed {} polytomy resolution{} as inconsistent with model parameters.'
            .format(num_removed_nodes, 's' if num_removed_nodes > 1 else ''))
        if num_new_nodes:
            logging.getLogger('pastml').debug(
                'Created {} new polytomy resolution{}.'.format(num_new_nodes, 's' if num_new_nodes > 1 else ''))
    elif num_polytomies - num_removed_nodes + num_new_nodes:
        logging.getLogger('pastml').debug('All the polytomy resolutions are consistent with model parameters.')
    return num_removed_nodes


def clear_extra_features(forest, features):
    features = set(features) | {'name', 'dist', 'support'}
    for tree in forest:
        for n in tree.traverse():
            for f in set(n.features) - features:
                if f not in features:
                    n.del_feature(f)


def copy_forest(forest, features=None):
    features = set(features if features else forest[0].features)
    copied_forest = []
    for tree in forest:
        copied_tree = TreeNode()
        todo = [(tree, copied_tree)]
        copied_forest.append(copied_tree)
        while todo:
            n, copied_n = todo.pop()
            copied_n.dist = n.dist
            copied_n.support = n.support
            copied_n.name = n.name
            for f in features:
                if hasattr(n, f):
                    copied_n.add_feature(f, getattr(n, f))
            for c in n.children:
                todo.append((c, copied_n.add_child()))
    return copied_forest


def main():
    """
    Entry point, calling :py:func:`pastml.tree.name_tree` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Name internal tree nodes as PastMl would.", prog='name_tree')

    parser.add_argument('-i', '--in_tree', help="input tree(s) in newick or nexus format (must be rooted).",
                        type=str, required=True)

    parser.add_argument('-o', '--out_tree', required=False, type=str,
                        help="path where to save the named output tree(s) in newick format.")

    parser.add_argument('--root_date', required=False, default=None,
                        help="date(s) of the root(s) (for dated tree(s) only), "
                             "if specified, the corresponding dates will be added to the tree node annotations.",
                        type=str, nargs='*')
    parser.add_argument('-c', '--columns', nargs='*',
                        help="names of the annotation columns of the input tree (if any) "
                             "to be kept in the output tree. "
                             "If there are LSD2-like date-related columns ({} and {}) present in the tree, "
                             "they will be kept in any case, so no need to specify them among columns here."
                        .format(DATE, DATE_CI),
                        type=str)
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=PASTML_VERSION))

    params = parser.parse_args()

    roots = read_forest(params.in_tree, columns=params.columns)
    root_dates = None
    if params.root_date is not None:
        root_dates = [parse_date(d) for d in params.root_date]
        if 1 < len(root_dates) < len(roots):
            raise ValueError('{} trees are given, but only {} root dates.'.format(len(roots), len(root_dates)))
        elif 1 == len(root_dates):
            root_dates *= len(roots)
    annotate_dates(roots, root_dates=root_dates, annotate_zeros=False)
    for i, tree in enumerate(roots):
        name_tree(tree, suffix='' if len(roots) == 1 else '_{}'.format(i))
    with open(params.out_tree, 'w+') as f:
        f.write(
            '\n'.join(
                [root.write(format_root_node=True, format=3,
                            features=[DATE, DATE_CI] + (params.columns if params.columns else []))
                 for root in roots]))


if '__main__' == __name__:
    main()
