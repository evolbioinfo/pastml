import logging
from collections import defaultdict

import numpy as np

IS_TIP = 'is_tip'

REASONABLE_NUMBER_OF_TIPS = 15

CATEGORIES = 'categories'

NUM_TIPS_INSIDE = 'max_size'

TIPS_INSIDE = 'in_tips'
INTERNAL_NODES_INSIDE = 'in_ns'
TIPS_BELOW = 'all_tips'
ROOTS = 'roots'

COMPRESSED_NODE = 'compressed_node'

METACHILD = 'metachild'


def compress_tree(tree, columns, can_merge_diff_sizes=True, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS):
    compressed_tree = tree.copy()

    for n_compressed, n in zip(compressed_tree.traverse('postorder'), tree.traverse('postorder')):
        n_compressed.add_feature(TIPS_INSIDE, [])
        n_compressed.add_feature(INTERNAL_NODES_INSIDE, [])
        n_compressed.add_feature(ROOTS, [n])
        if n_compressed.is_leaf():
            getattr(n_compressed, TIPS_INSIDE).append(n)
        else:
            getattr(n_compressed, INTERNAL_NODES_INSIDE).append(n)
        n.add_feature(COMPRESSED_NODE, n_compressed)

    collapse_vertically(compressed_tree, columns)

    for n in compressed_tree.traverse():
        n.add_feature(NUM_TIPS_INSIDE, len(getattr(n, TIPS_INSIDE)))
        n.add_feature(TIPS_INSIDE, [getattr(n, TIPS_INSIDE)])
        n.add_feature(INTERNAL_NODES_INSIDE, [getattr(n, INTERNAL_NODES_INSIDE)])

    get_bin = lambda _: _
    collapse_horizontally(compressed_tree, columns, get_bin)

    if can_merge_diff_sizes and len(compressed_tree) > tip_size_threshold:
        get_bin = lambda _: int(np.log10(max(1, _)))
        logging.getLogger('pastml').debug('Allowed merging nodes of different sizes.')
        collapse_horizontally(compressed_tree, columns, get_bin)

    if len(compressed_tree) > tip_size_threshold:
        for n in compressed_tree.traverse('preorder'):
            multiplier = (getattr(n.up, 'multiplier') if n.up else 1) * len(getattr(n, ROOTS))
            n.add_feature('multiplier', multiplier)

        def get_tsize(n):
            return getattr(n, NUM_TIPS_INSIDE) * getattr(n, 'multiplier')

        node_thresholds = []
        for n in compressed_tree.traverse('postorder'):
            children_bs = 0 if not n.children else max(get_tsize(_) for _ in n.children)
            bs = get_tsize(n)
            # if bs > children_bs it means that the trimming threshold for the node is higher
            # than the ones for its children
            if not n.is_root() and bs > children_bs:
                node_thresholds.append(bs)
        threshold = sorted(node_thresholds)[-tip_size_threshold]

        if min(node_thresholds) >= threshold:
            logging.getLogger('pastml')\
                .debug('No tip is smaller than the threshold ({}, the size of the {}-th largest tip).'
                       .format(threshold, tip_size_threshold))
        else:
            logging.getLogger('pastml').debug('Set tip size threshold to {} (the size of the {}-th largest tip).'
                                              .format(threshold, tip_size_threshold))
            remove_small_tips(compressed_tree=compressed_tree, full_tree=tree,
                              to_be_removed=lambda _: get_tsize(_) < threshold)
            remove_mediators(compressed_tree, columns)
            collapse_horizontally(compressed_tree, columns, get_bin)
    return compressed_tree


def collapse_horizontally(tree, columns, tips2bin):
    config_cache = {}

    def get_configuration(n):
        if n.name not in config_cache:
            # Configuration is (branch_width, (size, states, child_configurations)),
            # where branch_width is only used for recursive calls and is ignored when considering a merge
            config_cache[n.name] = (len(getattr(n, TIPS_INSIDE)),
                               (tips2bin(getattr(n, NUM_TIPS_INSIDE)),
                                tuple(tuple(sorted(getattr(n, column, set()))) for column in columns),
                                tuple(sorted([get_configuration(_) for _ in n.children]))))
        return config_cache[n.name]

    collapsed_configurations = 0

    for n in tree.traverse('postorder'):
        config2children = defaultdict(list)
        for _ in n.children:
            # use (size, states, child_configurations) as configuration (ignore branch width)
            config2children[get_configuration(_)[1]].append(_)
        for children in (_ for _ in config2children.values() if len(_) > 1):
            collapsed_configurations += 1
            child = children[0]
            for sibling in children[1:]:
                getattr(child, TIPS_INSIDE).extend(getattr(sibling, TIPS_INSIDE))
                for ti in getattr(sibling, TIPS_INSIDE):
                    for _ in ti:
                        _.add_feature(COMPRESSED_NODE, child)
                getattr(child, INTERNAL_NODES_INSIDE).extend(getattr(sibling, INTERNAL_NODES_INSIDE))
                for ii in getattr(sibling, INTERNAL_NODES_INSIDE):
                    for _ in ii:
                        _.add_feature(COMPRESSED_NODE, child)
                getattr(child, ROOTS).extend(getattr(sibling, ROOTS))
                n.remove_child(sibling)
            child.add_feature(METACHILD, True)
            child.add_feature(NUM_TIPS_INSIDE,
                              sum(len(_) for _ in getattr(child, TIPS_INSIDE)) / len(getattr(child, TIPS_INSIDE)))
            if child.name in config_cache:
                config_cache[child.name] = (len(getattr(child, TIPS_INSIDE)), config_cache[child.name][1])
    if collapsed_configurations:
        logging.getLogger('pastml').debug(
            'Collapsed {} sets of equivalent configurations horizontally.'.format(collapsed_configurations))


def remove_small_tips(compressed_tree, full_tree, to_be_removed):
    num_removed = 0
    changed = True
    while changed:
        changed = False
        for l in compressed_tree.get_leaves():
            parent = l.up
            if parent and to_be_removed(l):
                num_removed += 1
                parent.remove_child(l)
                # remove the corresponding nodes from the non-collapsed tree
                for ti in getattr(l, TIPS_INSIDE):
                    for _ in ti:
                        _.up.remove_child(_)
                for ii in getattr(l, INTERNAL_NODES_INSIDE):
                    for _ in ii:
                        _.up.remove_child(_)
                changed = True

    # if the full tree now contains non-sampled tips,
    # remove them from the tree and from the corresponding collapsed nodes
    todo = list(full_tree)
    while todo:
        t = todo.pop()
        if not getattr(t, IS_TIP, False):
            parent = t.up
            t.up.remove_child(t)
            if parent.is_leaf():
                todo.append(parent)
            for ini_list in getattr(getattr(t, COMPRESSED_NODE), INTERNAL_NODES_INSIDE):
                if t in ini_list:
                    ini_list.remove(t)

    logging.getLogger('pastml').debug(
        'Recursively removed {} tips of size smaller than the threshold.'.format(num_removed))


def collapse_vertically(tree, columns):
    """
    Collapses a child node into its parent if they are in the same state.
    :param columns: a list of characters
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """

    def _same_states(node1, node2, columns):
        for column in columns:
            if getattr(node1, column, set()) != getattr(node2, column, set()):
                return False
        return True

    num_collapsed = 0
    for n in tree.traverse('postorder'):
        if n.is_leaf():
            continue

        children = list(n.children)
        for child in children:
            # merge the child into this node if their states are the same
            if _same_states(n, child, columns):
                getattr(n, TIPS_INSIDE).extend(getattr(child, TIPS_INSIDE))
                for _ in getattr(child, TIPS_INSIDE):
                    _.add_feature(COMPRESSED_NODE, n)
                getattr(n, INTERNAL_NODES_INSIDE).extend(getattr(child, INTERNAL_NODES_INSIDE))
                for _ in getattr(child, INTERNAL_NODES_INSIDE):
                    _.add_feature(COMPRESSED_NODE, n)

                n.remove_child(child)
                grandchildren = list(child.children)
                for grandchild in grandchildren:
                    n.add_child(grandchild)
                num_collapsed += 1
    if num_collapsed:
        logging.getLogger('pastml').debug('Collapsed vertically {} internal nodes without state change.'
                                          .format(num_collapsed))


def remove_mediators(tree, columns):
    """
    Removes intermediate nodes that are just mediators between their parent and child states.
    :param columns: list of characters
    :param tree: ete3.Tree
    :return: void, modifies the input tree
    """
    num_removed = 0
    for n in tree.traverse('postorder'):
        if getattr(n, METACHILD, False) or n.is_leaf() or len(n.children) > 1 or n.is_root() \
                or getattr(n, NUM_TIPS_INSIDE) > 0:
            continue

        parent = n.up
        child = n.children[0]

        compatible = True
        for column in columns:
            states = getattr(n, column, set())
            parent_states = getattr(parent, column, set())
            child_states = getattr(child, column, set())
            # if mediator has unresolved states, it should hesitate between the parent and the child:
            if states != child_states | parent_states:
                compatible = False
                break

        if compatible:
            parent.remove_child(n)
            parent.add_child(child)
            # update the uncompressed tree
            for ii in getattr(n, INTERNAL_NODES_INSIDE):
                for _ in ii:
                    for c in list(_.children):
                        _.up.add_child(c)
                    _.up.remove_child(_)
            num_removed += 1
    if num_removed:
        logging.getLogger('pastml').debug("Removed {} internal node{}"
                                          " with the state unresolved between the parent's and the only child's."
                                          .format(num_removed, '' if num_removed == 1 else 's'))
