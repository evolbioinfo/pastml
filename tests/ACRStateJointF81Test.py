import os
import unittest
from collections import Counter

import pandas as pd
import numpy as np

from pastml.tree import read_tree, collapse_zero_branches
from pastml.acr import acr
from pastml.ml import JOINT, F81

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')


class ACRStateJointF81Test(unittest.TestCase):

    def setUp(self):
        self.feature = 'Country'
        self.df = pd.read_csv(STATES_INPUT, index_col=0, header=0)[[self.feature]]
        self.tree = read_tree(TREE_NWK)
        collapse_zero_branches(self.tree)
        acr(self.tree, self.df, prediction_method=JOINT, model=F81)

    def test_collapsed_vs_full(self):
        tree = read_tree(TREE_NWK)
        acr(tree, self.df, prediction_method=JOINT, model=F81)

        def get_state(node):
            state = getattr(node, self.feature)
            return state if not isinstance(state, list) else ', '.join(sorted(state))

        df_full = pd.DataFrame.from_dict({node.name: get_state(node) for node in tree.traverse()},
                                         orient='index', columns=['full'])
        df_collapsed = pd.DataFrame.from_dict({node.name: get_state(node) for node in self.tree.traverse()},
                                              orient='index', columns=['collapsed'])
        df = df_collapsed.join(df_full, how='left')
        self.assertTrue(np.all((df['collapsed'] == df['full'])),
                        msg='All the node states of the collapsed tree should be the same as of the full one.')

    def test_num_nodes(self):
        state2num = Counter()
        for node in self.tree.traverse():
            state = getattr(node, self.feature)
            if isinstance(state, list):
                state2num['unresolved'] += 1
            else:
                state2num[state] += 1
        expected_state2num = {'Africa': 114, 'Albania': 50, 'Greece': 69, 'WestEurope': 28, 'EastEurope': 16}
        self.assertDictEqual(expected_state2num, state2num, msg='Was supposed to have {} as states counts, got {}.'
                             .format(expected_state2num, state2num))

    def test_state_root(self):
        expected_state = 'Africa'
        state = getattr(self.tree, self.feature)
        self.assertEqual(expected_state, state,
                         msg='Root state was supposed to be {}, got {}.'.format(expected_state, state))

    def test_state_resolved_internal_node_1(self):
        expected_state = 'Africa'
        for node in self.tree.traverse():
            if 'node_79' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_resolved_internal_node_2(self):
        expected_state = 'Greece'
        for node in self.tree.traverse():
            if 'node_80' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_resolved_internal_node_3(self):
        expected_state = 'Greece'
        for node in self.tree.traverse():
            if 'node_25' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_resolved_internal_node_4(self):
        expected_state = 'Greece'
        for node in self.tree.traverse():
            if 'node_48' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_zero_tip(self):
        expected_state = 'Albania'
        for node in self.tree.traverse():
            if '01ALAY1715' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_tip(self):
        expected_state = 'WestEurope'
        for node in self.tree:
            if '94SEAF9671' == node.name:
                state = getattr(node, self.feature)
                self.assertEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break
