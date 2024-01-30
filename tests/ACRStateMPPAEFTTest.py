import os
import unittest
from collections import Counter

import numpy as np
import pandas as pd

from pastml.acr import acr
from pastml.annotation import annotate_forest
from pastml.ml import MPPA
from pastml.models.EFTModel import EFT
from pastml.tree import collapse_zero_branches, read_forest

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')

feature = 'Country'
tree = read_forest(TREE_NWK)[0]
collapse_zero_branches([tree])
_, column2states = annotate_forest([tree], columns=feature, data=STATES_INPUT, data_sep=',')
acr_result = acr([tree], character=feature, states=column2states[feature], prediction_method=MPPA, model=EFT)


class ACRStateMPPAEFTTest(unittest.TestCase):

    def test_collapsed_vs_full(self):
        tree_uncollapsed = read_forest(TREE_NWK)[0]
        annotate_forest([tree_uncollapsed], columns=feature, data=STATES_INPUT, data_sep=',')
        acr([tree_uncollapsed], character=feature, states=column2states[feature], prediction_method=MPPA, model=EFT)

        def get_state(node):
            return ', '.join(sorted(getattr(node, feature)))

        df_full = pd.DataFrame.from_dict({node.name: get_state(node) for node in tree_uncollapsed.traverse()},
                                         orient='index', columns=['full'])
        df_collapsed = pd.DataFrame.from_dict({node.name: get_state(node) for node in tree.traverse()},
                                              orient='index', columns=['collapsed'])
        df_joint = df_collapsed.join(df_full, how='left')
        self.assertTrue(np.all((df_joint['collapsed'] == df_joint['full'])),
                        msg='All the node states of the collapsed tree should be the same as of the full one.')

    def test_num_nodes(self):
        state2num = Counter()
        root = tree.copy()
        collapse_zero_branches([root])
        for node in root.traverse():
            state = getattr(node, feature)
            if len(state) > 1:
                state2num['unresolved'] += 1
            else:
                state2num[next(iter(state))] += 1
        expected_state2num = {'unresolved': 8, 'Africa': 109, 'Albania': 50, 'Greece': 65, 'WestEurope': 29, 'EastEurope': 16}
        self.assertDictEqual(expected_state2num, state2num, msg='Was supposed to have {} as states counts, got {}.'
                             .format(expected_state2num, state2num))

    def test_state_root(self):
        expected_state = {'Africa'}
        state = getattr(tree, feature)
        self.assertSetEqual(expected_state, state,
                         msg='Root state was supposed to be {}, got {}.'.format(expected_state, state))

    def test_state_unresolved_internal_node(self):
        expected_state = {'Africa', 'Greece'}
        for node in tree.traverse():
            if 'node_79' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                    .format(node.name, expected_state, state))
                break

    def test_state_node_32(self):
        expected_state = {'WestEurope'}
        for node in tree.traverse():
            if 'node_32' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_resolved_internal_node(self):
        expected_state = {'Greece'}
        for node in tree.traverse():
            if 'node_80' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_zero_tip(self):
        expected_state = {'Albania'}
        for node in tree.traverse():
            if '01ALAY1715' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_tip(self):
        expected_state = {'WestEurope'}
        for node in tree:
            if '94SEAF9671' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break
