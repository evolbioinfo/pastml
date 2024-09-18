import logging
import os
from collections import defaultdict
from glob import glob
from queue import Queue
from shutil import copyfile

import numpy as np
from jinja2 import Environment, PackageLoader

from pastml import numeric2datetime
from pastml.file import get_pastml_colour_file
from pastml.tree import DATE, IS_POLYTOMY, copy_forest
from pastml.visualisation import get_formatted_date
from pastml.visualisation.colour_generator import get_enough_colours, WHITE, parse_colours
from pastml.visualisation.tree_compressor import NUM_TIPS_INSIDE, TIPS_INSIDE, TIPS_BELOW, \
    REASONABLE_NUMBER_OF_TIPS, compress_tree, INTERNAL_NODES_INSIDE, ROOTS, IS_TIP, ROOT_DATES, IN_FOCUS, AROUND_FOCUS, \
    UP_FOCUS, save_to_pajek, VERTICAL

JS_LIST = ["https://pastml.pasteur.fr/static/js/jquery.min.js",
           "https://pastml.pasteur.fr/static/js/jquery.qtip.min.js",
           "https://pastml.pasteur.fr/static/js/cytoscape.min.js",
           "https://pastml.pasteur.fr/static/js/cytoscape-qtip.js",
           "https://pastml.pasteur.fr/static/js/cytoscape-svg.js",
           "https://pastml.pasteur.fr/static/js/layout-base.min.js",
           "https://pastml.pasteur.fr/static/js/cose-base.min.js",
           "https://pastml.pasteur.fr/static/js/cytoscape-cose-bilkent.min.js"]
CSS_LIST = ["https://pastml.pasteur.fr/static/css/jquery.qtip.min.css",
            "https://pastml.pasteur.fr/static/css/bootstrap.min.css"]

MAX_TIPS_FOR_FULL_TREE_VISUALISATION = 5000

TIMELINE_SAMPLED = 'SAMPLED'
TIMELINE_NODES = 'NODES'
TIMELINE_LTT = 'LTT'

TIP_LIMIT = 1000

MIN_EDGE_SIZE = 50
MIN_FONT_SIZE = 80
MIN_NODE_SIZE = 200

UNRESOLVED = 'unresolved'
TIP = 'tip'

TOOLTIP = 'tooltip'
COLOUR = 'colour'

DATA = 'data'
ID = 'id'
EDGES = 'edges'
NODES = 'nodes'
ELEMENTS = 'elements'

NODE_SIZE = 'node_size'
NODE_NAME = 'node_name'
BRANCH_NAME = 'branch_name'
EDGE_SIZE = 'edge_size'
EDGE_NAME = 'edge_name'
FONT_SIZE = 'node_fontsize'

MILESTONE = 'mile'

DATE_LABEL = 'date'

DIST_TO_ROOT_LABEL = 'dist. to root'


def get_fake_node(n_id, x, y):
    attributes = {ID: n_id, 'fake': 1}
    return _get_node(attributes, position=(x, y))


def get_node(n, n_id, tooltip='', clazz=None, x=0, y=0):
    features = {feature: getattr(n, feature) for feature in n.features if feature in [MILESTONE, UNRESOLVED, 'x', 'y']
                or feature.startswith('node_')}
    features[ID] = n_id
    if n.is_leaf():
        features[TIP] = 1
    features[TOOLTIP] = tooltip
    return _get_node(features, clazz=_clazz_list2css_class(clazz), position=(x, y) if x is not None else None)


def get_edge(source_name, target_name, **kwargs):
    return _get_edge(source=source_name, target=target_name, **kwargs)


def get_scaling_function(y_m, y_M, x_m, x_M):
    """
    Returns a linear function y = k x + b, where y \in [m, M]
    :param y_m:
    :param y_M:
    :param x_m:
    :param x_M:
    :return:
    """
    if x_M <= x_m:
        return lambda _: y_m
    k = (y_M - y_m) / (x_M - x_m)
    b = y_m - k * x_m
    return lambda _: int(k * _ + b)


def set_cyto_features_compressed(n, size_scaling, e_size_scaling, font_scaling, transform_size, transform_e_size, state,
                                 root_names, root_dates, suffix='', is_mixed=False):
    tips_inside, tips_below, internal_nodes_inside, roots = \
        getattr(n, TIPS_INSIDE, []), getattr(n, TIPS_BELOW, []), \
        getattr(n, INTERNAL_NODES_INSIDE, []), getattr(n, ROOTS, [])

    def get_min_max_str(values, default_value=0):
        min_v, max_v = (min(len(_) for _ in values), max(len(_) for _ in values)) \
            if values else (default_value, default_value)
        return ' {}'.format('{}-{}'.format(min_v, max_v) if min_v != max_v else min_v), min_v, max_v

    tips_below_str, _, max_n_tips_below = get_min_max_str(tips_below)
    tips_inside_str, _, max_n_tips = get_min_max_str(tips_inside)
    internal_ns_inside_str, _, _ = get_min_max_str(internal_nodes_inside)
    n.add_feature('{}{}'.format(NODE_NAME, suffix),
                  '{}{}'.format(state, tips_inside_str)
                  if not is_mixed or (not getattr(n, IN_FOCUS, False) and not getattr(n, UP_FOCUS, False)) else
                  '{}{}{}'.format(state, ':' if state else '', root_names[0]))
    size_factor = 2 if getattr(n, UNRESOLVED, False) else 1
    n.add_feature('{}{}'.format(NODE_SIZE, suffix),
                  (size_scaling(transform_size(max_n_tips)) if max_n_tips else int(MIN_NODE_SIZE / 1.5))
                  * size_factor)
    n.add_feature('{}{}'.format(FONT_SIZE, suffix),
                  font_scaling(transform_size(max_n_tips)) if max_n_tips else MIN_FONT_SIZE)

    n.add_feature('node_{}{}'.format(TIPS_INSIDE, suffix), tips_inside_str)
    n.add_feature('node_{}{}'.format(INTERNAL_NODES_INSIDE, suffix), internal_ns_inside_str)
    n.add_feature('node_{}{}'.format(TIPS_BELOW, suffix), tips_below_str)
    root_name2date = dict(zip(root_names, root_dates))
    root_names = sorted(root_names)
    n.add_feature('node_{}{}'.format(ROOTS, suffix), ', '.join(root_names))
    n.add_feature('node_{}{}'.format(ROOT_DATES, suffix), ', '.join(str(root_name2date[_]) for _ in root_names))

    edge_size = len(roots)
    if edge_size > 1:
        n.add_feature('node_meta{}'.format(suffix), 1)
    n.add_feature('{}{}'.format(EDGE_NAME, suffix), str(edge_size) if edge_size != 1 else '')
    e_size = e_size_scaling(transform_e_size(edge_size))
    n.add_feature('{}{}'.format(EDGE_SIZE, suffix), e_size)


def set_cyto_features_tree(n, state):
    n.add_feature(NODE_NAME, state)
    n.add_feature(EDGE_NAME, n.dist)


def _forest2json_compressed(forest, compressed_forest, columns, name_feature, get_date, milestones=None,
                            dates_are_dates=True, is_mixed=False):
    e_size_scaling, font_scaling, size_scaling, transform_e_size, transform_size = \
        get_size_transformations(compressed_forest)

    sort_key = lambda n: (getattr(n, UNRESOLVED, 0),
                          get_column_value_str(n, name_feature, format_list=True) if name_feature else '',
                          *(get_column_value_str(n, column, format_list=True) for column in columns),
                          -getattr(n, NUM_TIPS_INSIDE),
                          -len(getattr(n, ROOTS)),
                          n.name)
    i = 0
    node2id = {}
    todo = Queue()
    for compressed_tree in compressed_forest:
        todo.put_nowait(compressed_tree)
        node2id[compressed_tree] = i
        i += 1
        while not todo.empty():
            n = todo.get_nowait()
            for c in sorted(n.children, key=sort_key):
                node2id[c] = i
                i += 1
                todo.put_nowait(c)

    n2state = {}

    # Set the cytoscape features
    for compressed_tree in compressed_forest:
        for n in compressed_tree.traverse():
            state = get_column_value_str(n, name_feature, format_list=False, list_value='') if name_feature else ''
            n2state[n] = state
            root_names = [_.name for _ in getattr(n, ROOTS)]
            root_dates = [get_formatted_date(_, dates_are_dates) for _ in getattr(n, ROOTS)]
            set_cyto_features_compressed(n, size_scaling, e_size_scaling, font_scaling,
                                         transform_size, transform_e_size, state, root_names, root_dates,
                                         is_mixed=is_mixed)

    # Calculate node coordinates
    min_size = 2 * min(min(getattr(_, NODE_SIZE) for _ in compressed_tree.traverse())
                       for compressed_tree in compressed_forest)
    n2width = {}
    for compressed_tree in compressed_forest:
        for n in compressed_tree.traverse('postorder'):
            n2width[n] = max(getattr(n, NODE_SIZE),
                             sum(n2width[c] for c in n.children) + min_size * (len(n.children) - 1))

    n2x, n2y = {}, {compressed_tree: 0 for compressed_tree in compressed_forest}
    n2offset = {}
    tree_offset = 0
    for compressed_tree in compressed_forest:
        n2offset[compressed_tree] = tree_offset
        tree_offset += n2width[compressed_tree] + 2 * min_size
        for n in compressed_tree.traverse('preorder'):
            n2x[n] = n2offset[n] + n2width[n] / 2
            offset = n2offset[n]
            if not n.is_leaf():
                for c in sorted(n.children, key=lambda c: node2id[c]):
                    n2offset[c] = offset
                    offset += n2width[c] + min_size
                    n2y[c] = n2y[n] + getattr(n, NODE_SIZE) / 2 + getattr(c, NODE_SIZE) / 2 + min_size
        for n in compressed_tree.traverse('postorder'):
            if not n.is_leaf():
                n2x[n] = np.mean([n2x[c] for c in n.children])

    def filter_by_date(items, date):
        return [_ for _ in items if get_date(_) <= date]

    # Set the cytoscape feature for different timeline points
    for tree, compressed_tree in zip(forest, compressed_forest):
        if len(milestones) > 1:
            nodes = list(compressed_tree.traverse())
            for i in range(len(milestones) - 1, -1, -1):
                milestone = milestones[i]
                nodes_i = []

                # remove too recent nodes from the original tree
                for n in tree.traverse('postorder'):
                    if n.is_root():
                        continue
                    if get_date(n) > milestone:
                        n.up.remove_child(n)

                suffix = '_{}'.format(i)
                for n in nodes:
                    state = n2state[n]
                    tips_inside, tips_below, internal_nodes_inside, roots = getattr(n, TIPS_INSIDE, []), \
                                                                            getattr(n, TIPS_BELOW, []), \
                                                                            getattr(n, INTERNAL_NODES_INSIDE, []), \
                                                                            getattr(n, ROOTS, [])
                    tips_inside_i, tips_below_i, internal_nodes_inside_i, roots_i = [], [], [], []
                    for ti, tb, ini, root in zip(tips_inside, tips_below, internal_nodes_inside, roots):
                        if get_date(root) <= milestone:
                            roots_i.append(root)

                            ti = filter_by_date(ti, milestone)
                            tb = [_ for _ in tb if getattr(_, DATE) <= milestone]
                            ini = filter_by_date(ini, milestone)

                            tips_inside_i.append(ti + [_ for _ in ini if _.is_leaf()])
                            tips_below_i.append(tb)
                            internal_nodes_inside_i.append([_ for _ in ini if not _.is_leaf()])
                    n.add_features(**{TIPS_INSIDE: tips_inside_i, TIPS_BELOW: tips_below_i,
                                      INTERNAL_NODES_INSIDE: internal_nodes_inside_i, ROOTS: roots_i})
                    if roots_i:
                        n.add_feature(MILESTONE, i)
                        root_names = [getattr(_, BRANCH_NAME) if getattr(_, DATE) > milestone else _.name for _ in
                                      roots_i]
                        root_dates = [getattr(_, DATE) for _ in roots_i]
                        if dates_are_dates:
                            try:
                                root_dates = [numeric2datetime(_).strftime("%d %b %Y") for _ in root_dates]
                            except:
                                pass
                        set_cyto_features_compressed(n, size_scaling, e_size_scaling, font_scaling, transform_size,
                                                     transform_e_size, state, root_names=root_names, suffix=suffix,
                                                     root_dates=root_dates, is_mixed=is_mixed)
                        nodes_i.append(n)
                nodes = nodes_i

    # Save the structure
    clazzes = set()
    nodes, edges = [], []

    one_column = columns[0] if len(columns) == 1 else None

    for n, n_id in node2id.items():
        if one_column:
            values = getattr(n, one_column, set())
            clazz = tuple(sorted(values))
        else:
            clazz = tuple('{}_{}'.format(column, get_column_value_str(n, column, format_list=False, list_value=''))
                          for column in columns)
        if clazz:
            clazzes.add(clazz)
        nodes.append(get_node(n, n_id, tooltip=get_tooltip(n, columns),
                              clazz=clazz, x=n2x[n], y=n2y[n]))

        for child in sorted(n.children, key=lambda _: node2id[_]):
            edge_attributes = {feature: getattr(child, feature) for feature in child.features
                               if feature.startswith('edge_') or feature == MILESTONE or feature == IS_POLYTOMY}
            source_name = n_id
            edges.append(get_edge(source_name, node2id[child], **edge_attributes))

    json_dict = {NODES: nodes, EDGES: edges}
    return json_dict, sorted(clazzes)


def binary_search(start, end, value, array):
    if start >= end - 1:
        return start
    i = int((start + end) / 2)
    if array[i] == value or array[i] > value and (i == start or value > array[i - 1]):
        return i
    if array[i] > value:
        return binary_search(start, i, value, array)
    return binary_search(i + 1, end, value, array)


def _forest2json(forest, columns, name_feature, get_date, milestones=None, timeline_type=TIMELINE_SAMPLED,
                 dates_are_dates=True):
    min_root_date = min(getattr(tree, DATE) for tree in forest)
    width = sum(len(tree) for tree in forest)
    height_factor = 300 * width / max(
        (max(getattr(_, DATE) for _ in tree) - min_root_date + tree.dist) for tree in forest)
    zero_dist = min(min(min(_.dist for _ in tree.traverse() if _.dist > 0) for tree in forest), 300) * height_factor / 2

    # Calculate node coordinates
    n2x, n2y = {}, {}
    x = -600
    for tree in forest:
        x += 600
        for t in tree:
            n2x[t] = x
            x += 600

    for tree in forest:
        for n in tree.traverse('postorder'):
            state = get_column_value_str(n, name_feature, format_list=False, list_value='') if name_feature else ''
            n.add_feature('node_root_id', n.name)
            n.add_feature('node_root_date', get_formatted_date(n, dates_are_dates))
            if not n.is_leaf():
                n2x[n] = np.mean([n2x[_] for _ in n.children])
            n2y[n] = (getattr(n, DATE) - min_root_date) * height_factor
            for c in n.children:
                if n2y[c] == n2y[n]:
                    n2y[c] += zero_dist
            set_cyto_features_tree(n, state)

    # Save cytoscape features at different timeline points
    if len(milestones) > 1:
        for tree in forest:
            for n in tree.traverse('preorder'):
                ms_i = binary_search(0 if n.is_root() else getattr(n.up, MILESTONE),
                                     len(milestones), get_date(n), milestones)
                n.add_feature(MILESTONE, ms_i)
                for i in range(len(milestones) - 1, ms_i - 1, -1):
                    milestone = milestones[i]
                    suffix = '_{}'.format(i)
                    if TIMELINE_LTT == timeline_type:
                        # if it is LTT also cut the branches if needed
                        if getattr(n, DATE) > milestone:
                            n.add_feature('{}{}'.format(EDGE_NAME, suffix),
                                          np.round(milestone - getattr(n.up, DATE), 3))
                        else:
                            n.add_feature('{}{}'.format(EDGE_NAME, suffix), np.round(n.dist, 3))

    # Save the structure
    clazzes = set()
    nodes, edges = [], []

    one_column = columns[0] if len(columns) == 1 else None

    i = 0
    node2id = {}
    for tree in forest:
        for n in tree.traverse():
            node2id[n] = i
            i += 1

    for n, n_id in node2id.items():
        if n.is_root():
            fake_id = 'fake_node_{}'.format(n_id)
            nodes.append(get_fake_node(fake_id, n2x[n], n2y[n] - n.dist * height_factor))
            edges.append(get_edge(fake_id, n_id, **{feature: getattr(n, feature) for feature in n.features
                                                    if feature.startswith('edge_') or feature == MILESTONE}))
        if one_column:
            values = getattr(n, one_column, set())
            clazz = tuple(sorted(values))
        else:
            clazz = tuple('{}_{}'.format(column, get_column_value_str(n, column, format_list=False, list_value=''))
                          for column in columns)
        if clazz:
            clazzes.add(clazz)
        nodes.append(get_node(n, n_id, tooltip=get_tooltip(n, columns), clazz=clazz, x=n2x[n], y=n2y[n]))

        for child in n.children:
            edge_attributes = {feature: getattr(child, feature) for feature in child.features
                               if feature.startswith('edge_') or feature == MILESTONE or feature == IS_POLYTOMY}
            source_name = n_id
            target_name = 'fake_node_{}'.format(node2id[child])
            nodes.append(get_fake_node(target_name, x=n2x[child], y=n2y[n]))
            edges.append(get_edge(source_name, target_name, fake=1,
                                  **{k: v for (k, v) in edge_attributes.items()
                                     if EDGE_NAME not in k and IS_POLYTOMY not in k}))
            source_name = target_name
            edges.append(get_edge(source_name, node2id[child], **edge_attributes))

    json_dict = {NODES: nodes, EDGES: edges}
    return json_dict, sorted(clazzes)


def _forest2json_transitions(states, counts, transitions, state2colour, threshold=0):
    nodes, edges = [], []
    n = len(states)

    n_scaler = get_scaling_function(y_m=200, y_M=800, x_m=min(counts), x_M=max(counts))
    font_scaler = get_scaling_function(y_m=MIN_FONT_SIZE, y_M=MIN_FONT_SIZE * 3, x_m=min(counts), x_M=max(counts))
    positive_transitions = transitions[transitions > 0]
    e_scaler = get_scaling_function(y_m=30, y_M=200, x_m=min(positive_transitions), x_M=max(positive_transitions))

    transtions_to_from = np.transpose(transitions.copy())
    np.fill_diagonal(transtions_to_from, 0)
    nums = np.triu(transitions + transtions_to_from).flatten()
    positive_nums = nums[nums > 0]

    max_transition_num = max(positive_nums)
    if max_transition_num <= 2:
        miles = sorted((set({0, threshold, 1 if max_transition_num > 1 else 0}
                                     | set(np.round(positive_nums, 3))) - {np.round(max_transition_num, 3)}))
    else:
        miles = sorted(set({0, threshold, 1} | set(np.trunc(positive_nums))))
    miles = np.array([_ for _ in miles if threshold <= _ < max_transition_num])
    if not len(miles):
        miles = [0]

    # we hide connections when they are < mile
    def get_mile(start, end, value):
        if start == end:
            return None
        i = int((start + end) / 2)
        if miles[i] <= value:
            if i == end - 1 or value < miles[i + 1]:
                return i
            return get_mile(i + 1, end, value)
        return get_mile(start, i, value)

    i2mile = defaultdict(lambda: -1)
    for i in range(n):
        from_state = states[i]
        n_tips = counts[i]
        i_node_size = n_scaler(n_tips)
        if n_tips > 0:
            for j in range(i, n):
                if counts[j] > 0:
                    to_state = states[j]
                    n_ij = transitions[i, j]
                    n_ji = transitions[j, i]
                    node_size = (i_node_size + n_scaler(counts[j])) / 2
                    if n_ij > 0:
                        mile = get_mile(0, len(miles), n_ij)
                        if mile is not None:
                            i2mile[i] = max(mile, i2mile[i])
                            i2mile[j] = max(mile, i2mile[j])
                            edges.append(get_edge(i, j,
                                                  **{ID: '{}_{}'.format(i, j),
                                                     EDGE_SIZE: e_scaler(n_ij),
                                                     NODE_SIZE: node_size / (2 if i != j else 1),
                                                     EDGE_NAME: n_ij,
                                                     TOOLTIP: '{} transitions from {} to {}'
                                                  .format(n_ij, from_state, to_state),
                                                     MILESTONE: mile}))
                    if n_ji > 0 and i != j:
                        mile = get_mile(0, len(miles), n_ji)
                        if mile is not None:
                            i2mile[i] = max(mile, i2mile[i])
                            i2mile[j] = max(mile, i2mile[j])
                            edges.append(get_edge(j, i,
                                                  **{ID: '{}_{}'.format(j, i),
                                                     EDGE_SIZE: e_scaler(n_ji),
                                                     NODE_SIZE: node_size / 2,
                                                     EDGE_NAME: n_ji,
                                                     TOOLTIP: '{} transitions from {} to {}'
                                                  .format(n_ji, to_state, from_state),
                                                     MILESTONE: mile}))
            if i2mile[i] >= 0:
                nodes.append(_get_node(data={ID: i, NODE_NAME: '{} ({:.0f})'.format(from_state, n_tips),
                                             NODE_SIZE: i_node_size,
                                             FONT_SIZE: font_scaler(n_tips),
                                             TOOLTIP: '{} is represented by {:.0f} samples.'.format(from_state, n_tips),
                                             COLOUR: state2colour[from_state],
                                             MILESTONE: i2mile[i]}))
    json_dict = {NODES: nodes, EDGES: edges}
    return json_dict, ['{:g}'.format(_) for _ in miles]


def get_size_transformations(forest):
    max_size, min_size, max_e_size, min_e_size = 1, np.inf, 1, np.inf
    for tree in forest:
        for n in tree.traverse():
            sz = max(getattr(n, NUM_TIPS_INSIDE), 1)
            max_size = max(max_size, sz)
            min_size = min(min_size, sz)
            e_sz = len(getattr(n, ROOTS))
            max_e_size = max(max_e_size, e_sz)
            min_e_size = min(min_e_size, e_sz)

    need_log = max_size / min_size > 100
    transform_size = lambda _: np.power(np.log10(_ + 9) if need_log else _, 1 / 2)

    need_e_log = max_e_size / min_e_size > 100
    transform_e_size = lambda _: np.log10(_) if need_e_log else _

    size_scaling = get_scaling_function(y_m=MIN_NODE_SIZE, y_M=MIN_NODE_SIZE * min(8, int(max_size / min_size)),
                                        x_m=transform_size(min_size), x_M=transform_size(max_size))
    font_scaling = get_scaling_function(y_m=MIN_FONT_SIZE, y_M=MIN_FONT_SIZE * min(3, int(max_size / min_size)),
                                        x_m=transform_size(min_size), x_M=transform_size(max_size))
    e_size_scaling = get_scaling_function(y_m=MIN_EDGE_SIZE, y_M=MIN_EDGE_SIZE * min(3, int(max_e_size / min_e_size)),
                                          x_m=transform_e_size(min_e_size), x_M=transform_e_size(max_e_size))

    return e_size_scaling, font_scaling, size_scaling, transform_e_size, transform_size


def get_tooltip(n, columns):
    return '<br>'.join('{}: {}'.format(column, get_column_value_str(n, column, format_list=True))
                       for column in columns)


def save_as_cytoscape_html(forest, out_html, column2states, name_feature, name2colour, compressed_forest,
                           milestone_label, timeline_type, milestones, get_date, work_dir, local_css_js=False,
                           milestone_labels=None, is_mixed=False):
    """
    Converts a forest to an html representation using Cytoscape.js.

    If categories are specified they are visualised as pie-charts inside the nodes,
    given that each node contains features corresponding to these categories with values being the percentage.
    For instance, given categories ['A', 'B', 'C'], a node with features {'A': 50, 'B': 50}
    will have a half-half pie-chart (half-colored in a colour of A, and half B).

    :param name_feature: str, a node feature whose value will be used as a label
    :param name2colour: dict, str to str, category name to HEX colour mapping 
    :param forest: list(ete3.Tree)
    :param out_html: path where to save the resulting html file.
    """
    graph_name = os.path.splitext(os.path.basename(out_html))[0]
    columns = sorted(column2states.keys())
    if milestone_labels is None:
        milestone_labels = ['{:g}'.format(_) for _ in milestones]

    if compressed_forest is not None:
        json_dict, clazzes \
            = _forest2json_compressed(forest, compressed_forest, columns, name_feature=name_feature, get_date=get_date,
                                      milestones=milestones, dates_are_dates=milestone_label == DATE_LABEL,
                                      is_mixed=is_mixed)
    else:
        json_dict, clazzes \
            = _forest2json(forest, columns, name_feature=name_feature, get_date=get_date, milestones=milestones,
                           timeline_type=timeline_type, dates_are_dates=milestone_label == DATE_LABEL)
    loader = PackageLoader('pastml')
    env = Environment(loader=loader)
    template = env.get_template('pie_tree.js') if compressed_forest is not None \
        else env.get_template('pie_tree_simple.js')

    clazz2css = {}
    for clazz_list in clazzes:
        n = len(clazz_list)
        css = ''
        for i, cat in enumerate(clazz_list, start=1):
            css += """
                'pie-{i}-background-color': "{colour}",
                'pie-{i}-background-size': '{percent}\%',
            """.format(i=i, percent=round(100 / n, 2), colour=name2colour[cat])
        clazz2css[_clazz_list2css_class(clazz_list)] = css
    graph = template.render(clazz2css=clazz2css.items(), elements=json_dict, title=graph_name,
                            years=milestone_labels,
                            tips='samples' if TIMELINE_SAMPLED == timeline_type
                            else ('lineages ending' if TIMELINE_LTT == timeline_type else 'external nodes'),
                            internal_nodes='internal nodes' if TIMELINE_NODES == timeline_type
                            else 'diversification events',
                            age_label=milestone_label)
    slider = env.get_template('time_slider.html').render(min_date=0, max_date=len(milestones) - 1,
                                                         cur_date=len(milestones) - 1,
                                                         name=milestone_label) if len(milestones) > 1 else ''

    template = env.get_template('index.html')
    os.makedirs(os.path.abspath(os.path.dirname(out_html)), exist_ok=True)

    if local_css_js:
        js_list = []
        os.makedirs(os.path.join(work_dir, 'js'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'fonts'), exist_ok=True)

        template_dir = os.path.join(os.path.abspath(os.path.split(__file__)[0]), '..', 'templates')
        for _ in sorted(glob(os.path.join(template_dir, 'js', '*.js*'))):
            cp = os.path.join(work_dir, 'js', os.path.split(_)[1])
            copyfile(_, cp)
            if cp.endswith('.js'):
                js_list.append(cp)
        css_list = []
        for _ in glob(os.path.join(template_dir, 'css', '*.css*')):
            cp = os.path.join(work_dir, 'css', os.path.split(_)[1])
            copyfile(_, cp)
            if cp.endswith('.css'):
                css_list.append(cp)
        for _ in glob(os.path.join(template_dir, 'fonts', '*.*')):
            cp = os.path.join(work_dir, 'fonts', os.path.split(_)[1])
            copyfile(_, cp)
    else:
        js_list = JS_LIST
        css_list = CSS_LIST
    page = template.render(graph=graph, title=graph_name, slider=slider, js_list=js_list, css_list=css_list)

    with open(out_html, 'w+') as fp:
        fp.write(page)


def save_as_transition_html(character, states, counts, transitions, out_html, state2colour, work_dir,
                            local_css_js=False, threshold=0):
    """
    Converts transition count data to an html representation using Cytoscape.js.

    :param out_html: path where to save the resulting html file.
    """
    graph_name = os.path.splitext(os.path.basename(out_html))[0]
    transitions[transitions < threshold] = 0
    json_dict, thresholds = _forest2json_transitions(states, counts, transitions, state2colour, threshold=threshold)

    loader = PackageLoader('pastml')
    env = Environment(loader=loader)
    template = env.get_template('transitions.js')

    graph = template.render(elements=json_dict, character=character,
                            years=thresholds)
    slider = env.get_template('time_slider.html').render(min_date=0, max_date=len(thresholds) - 1,
                                                         name='transition number threshold', cur_date=0) if len(
        thresholds) > 1 else ''

    template = env.get_template('index.html')
    os.makedirs(os.path.abspath(os.path.dirname(out_html)), exist_ok=True)

    if local_css_js:
        js_list = []
        os.makedirs(os.path.join(work_dir, 'js'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'css'), exist_ok=True)
        os.makedirs(os.path.join(work_dir, 'fonts'), exist_ok=True)

        template_dir = os.path.join(os.path.abspath(os.path.split(__file__)[0]), '..', 'templates')
        for _ in sorted(glob(os.path.join(template_dir, 'js', '*.js*'))):
            cp = os.path.join(work_dir, 'js', os.path.split(_)[1])
            copyfile(_, cp)
            if cp.endswith('.js'):
                js_list.append(cp)
        css_list = []
        for _ in glob(os.path.join(template_dir, 'css', '*.css*')):
            cp = os.path.join(work_dir, 'css', os.path.split(_)[1])
            copyfile(_, cp)
            if cp.endswith('.css'):
                css_list.append(cp)
        for _ in glob(os.path.join(template_dir, 'fonts', '*.*')):
            cp = os.path.join(work_dir, 'fonts', os.path.split(_)[1])
            copyfile(_, cp)
    else:
        js_list = JS_LIST
        css_list = CSS_LIST
    page = template.render(graph=graph, title=graph_name, slider=slider, js_list=js_list, css_list=css_list)

    with open(out_html, 'w+') as fp:
        fp.write(page)


def _clazz_list2css_class(clazz_list):
    if not clazz_list:
        return None
    return ''.join(c for c in '-'.join(clazz_list) if c.isalnum() or '-' == c)


def _get_node(data, clazz=None, position=None):
    if position:
        data['node_x'] = float(position[0])
        data['node_y'] = float(position[1])
    res = {DATA: data}
    if clazz:
        res['classes'] = clazz
    return res


def _get_edge(**data):
    return {DATA: data}


def get_column_value_str(n, column, format_list=True, list_value=''):
    values = getattr(n, column, set())
    if isinstance(values, str):
        return values
    return ' or '.join(sorted(values)) if format_list or len(values) == 1 else list_value


def visualize(forest, column2states, work_dir, name_column=None, html=None, html_compressed=None, html_mixed=None,
              tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, date_label='Dist. to root', timeline_type=TIMELINE_SAMPLED,
              local_css_js=False, column2colours=None, focus=None, pajek=None, pajek_timing=VERTICAL):
    for tree in forest:
        nodes_in_focus = set()
        for node in tree.traverse():
            for column in column2states.keys():
                col_state = getattr(node, column, set())
                if focus and col_state & focus[column]:
                    nodes_in_focus.add(node)
        for node in nodes_in_focus:
            node.add_feature(IN_FOCUS, True)
            if not node.is_root() and node.up not in nodes_in_focus:
                node.up.add_feature(UP_FOCUS, True)
            for c in node.children:
                if c not in nodes_in_focus:
                    c.add_feature(AROUND_FOCUS, True)

    one_column = next(iter(column2states.keys())) if len(column2states) == 1 else None

    name2colour = {}
    for column, states in column2states.items():
        num_unique_values = len(states)
        colours = None
        if column2colours and column in column2colours:
            try:
                colours = parse_colours(column2colours[column], states)
            except ValueError as e:
                logging.getLogger('pastml').error('Failed to parse the input colours: {}'.format(e))
        if colours is None:
            colours = get_enough_colours(num_unique_values)
        for value, col in zip(states, colours):
            name2colour[value if one_column else '{}_{}'.format(column, value)] = col
        state2color = dict(zip(states, colours))
        # let ambiguous values be white
        if one_column is None:
            name2colour['{}_'.format(column)] = WHITE
        if column == name_column:
            for tree in forest:
                for n in tree.traverse():
                    sts = getattr(n, column, set())
                    if len(sts) == 1 and not n.is_root() and getattr(n.up, column, set()) == sts:
                        n.add_feature('edge_color', state2color[next(iter(sts))])
        out_colour_file = os.path.join(work_dir, get_pastml_colour_file(column))
        # Not using DataFrames to speed up document writing
        with open(out_colour_file, 'w+') as f:
            f.write('state\tcolour\n')
            for s in sorted(states):
                f.write('{}\t{}\n'.format(s, state2color[s]))
        logging.getLogger('pastml').debug('Mapped states to colours for {} as following: {} -> {}, '
                                          'and serialized this mapping to {}.'
                                          .format(column, states, colours, out_colour_file))
    for tree in forest:
        for node in tree.traverse():
            if node.is_leaf():
                node.add_feature(IS_TIP, True)
            node.add_feature(BRANCH_NAME, '{}-{}'.format(node.up.name if not node.is_root() else '', node.name))
            for column in column2states.keys():
                col_state = getattr(node, column, set())
                if len(col_state) != 1:
                    node.add_feature(UNRESOLVED, 1)
                    break

    if TIMELINE_NODES == timeline_type:
        def get_date(node):
            return getattr(node, DATE)
    elif TIMELINE_SAMPLED == timeline_type:
        max_date = max(max(getattr(_, DATE) for _ in tree) for tree in forest)

        def get_date(node):
            tips = [_ for _ in node if getattr(_, IS_TIP, False)]
            return min(getattr(_, DATE) for _ in tips) if tips else max_date
    elif TIMELINE_LTT == timeline_type:
        def get_date(node):
            return getattr(node, DATE) if node.is_root() else (getattr(node.up, DATE) + 1e-6)
    else:
        raise ValueError('Unknown timeline type: {}. Allowed ones are {}, {} and {}.'
                         .format(timeline_type, TIMELINE_NODES, TIMELINE_SAMPLED, TIMELINE_LTT))
    dates = []
    for tree in forest:
        dates.extend([getattr(_, DATE) for _ in (tree.traverse()
                                                 if timeline_type in [TIMELINE_LTT, TIMELINE_NODES] else tree)])
    dates = sorted(dates)
    milestones = sorted({dates[0], dates[len(dates) // 8], dates[len(dates) // 4], dates[3 * len(dates) // 8],
                         dates[len(dates) // 2], dates[5 * len(dates) // 8], dates[3 * len(dates) // 4],
                         dates[7 * len(dates) // 8], dates[-1]})
    milestone_labels = None
    if DATE_LABEL == date_label:
        try:
            milestone_labels = [numeric2datetime(_).strftime("%d %b %Y") for _ in milestones]
        except:
            pass
    if milestone_labels is None:
        milestone_labels = ['{:g}'.format(_) for _ in milestones]

    if html:
        total_num_tips = sum(len(tree) for tree in forest)
        if total_num_tips > MAX_TIPS_FOR_FULL_TREE_VISUALISATION:
            logging.getLogger('pastml').error('The full tree{} will not be visualised as {} too large ({} tips): '
                                              'the limit is {} tips. Check out upload to iTOL option instead.'
                                              .format('s' if len(forest) > 1 else '',
                                                      'they are' if len(forest) > 1 else 'it is',
                                                      total_num_tips, MAX_TIPS_FOR_FULL_TREE_VISUALISATION))
        else:
            save_as_cytoscape_html(forest, html, column2states=column2states, name2colour=name2colour,
                                   name_feature='name', compressed_forest=None, milestone_label=date_label,
                                   timeline_type=timeline_type, milestones=milestones, get_date=get_date,
                                   work_dir=work_dir, local_css_js=local_css_js, milestone_labels=milestone_labels)
    if html_compressed and html_mixed:
        forest_mixed = copy_forest(forest)
    else:
        forest_mixed = forest

    if html_compressed:
        pajek_vert_arcs = [[], []] if pajek else None
        compressed_forest = [compress_tree(tree, columns=column2states.keys(), tip_size_threshold=tip_size_threshold,
                                           mixed=False, pajek=pajek_vert_arcs, pajek_timing=pajek_timing)
                             for tree in forest]
        if pajek:
            save_to_pajek(*pajek_vert_arcs, pajek)

        milestone_labels, milestones = update_milestones(forest, date_label, milestone_labels, milestones,
                                                         timeline_type)

        save_as_cytoscape_html(forest, html_compressed,
                               column2states=column2states, name2colour=name2colour,
                               name_feature=name_column, compressed_forest=compressed_forest,
                               milestone_label=date_label, timeline_type=timeline_type,
                               milestones=milestones, get_date=get_date, work_dir=work_dir, local_css_js=local_css_js,
                               milestone_labels=milestone_labels, is_mixed=False)

    if html_mixed:
        mixed_forest = [compress_tree(tree, columns=column2states.keys(), tip_size_threshold=tip_size_threshold,
                                      mixed=True) for tree in forest_mixed]
        milestone_labels, milestones = update_milestones(forest_mixed, date_label, milestone_labels, milestones,
                                                         timeline_type)
        save_as_cytoscape_html(forest_mixed, html_mixed,
                               column2states=column2states, name2colour=name2colour,
                               name_feature=name_column, compressed_forest=mixed_forest,
                               milestone_label=date_label, timeline_type=timeline_type,
                               milestones=milestones, get_date=get_date, work_dir=work_dir, local_css_js=local_css_js,
                               milestone_labels=milestone_labels, is_mixed=True)


def update_milestones(forest, date_label, milestone_labels, milestones, timeline_type):
    # If we trimmed a few tips while compressing and they happened to be the oldest/newest ones,
    # we should update the milestones accordingly.
    first_date, last_date = np.inf, -np.inf
    for tree in forest:
        for _ in (tree.traverse() if timeline_type in [TIMELINE_LTT, TIMELINE_NODES] else tree):
            first_date = min(first_date, getattr(_, DATE))
            last_date = max(last_date, getattr(_, DATE))
    milestones = [ms for ms in milestones if first_date <= ms <= last_date]
    if milestones[0] > first_date:
        milestones.insert(0, first_date)
    if milestones[-1] < last_date:
        milestones.append(last_date)
    if DATE_LABEL == date_label:
        try:
            milestone_labels = [numeric2datetime(_).strftime("%d %b %Y") for _ in milestones]
        except:
            pass
    if milestone_labels is None:
        milestone_labels = ['{:g}'.format(_) for _ in milestones]
    return milestone_labels, milestones
