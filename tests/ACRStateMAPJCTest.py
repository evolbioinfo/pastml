import os
import unittest
from collections import Counter

import pandas as pd
import numpy as np

from pastml.tree import read_tree, collapse_zero_branches
from pastml.acr import acr
from pastml.ml import MAP
from pastml.models.f81_like import JC

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')

feature = 'Country'
df = pd.read_csv(STATES_INPUT, index_col=0, header=0)[[feature]]
tree = read_tree(TREE_NWK)
acr(tree, df, prediction_method=MAP, model=JC)


class ACRStateMAPJCTest(unittest.TestCase):

    def test_collapsed_vs_full(self):
        tree_uncollapsed = read_tree(TREE_NWK)
        acr(tree_uncollapsed, df, prediction_method=MAP, model=JC)

        def get_state(node):
            state = getattr(node, feature)
            return state if not isinstance(state, list) else ', '.join(sorted(state))

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
        expected_state2num = {'Africa': 114, 'Albania': 50, 'Greece': 67, 'WestEurope': 30, 'EastEurope': 16}
        self.assertDictEqual(expected_state2num, state2num, msg='Was supposed to have {} as states counts, got {}.'
                             .format(expected_state2num, state2num))

    def test_state_root(self):
        expected_state = {'Africa'}
        state = getattr(tree, feature)
        self.assertSetEqual(expected_state, state,
                         msg='Root state was supposed to be {}, got {}.'.format(expected_state, state))

    def test_state_node_79(self):
        expected_state = {'Africa'}
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

    def test_state_node_80(self):
        expected_state = {'Greece'}
        for node in tree.traverse():
            if 'node_80' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_node_25(self):
        expected_state = {'WestEurope'}
        for node in tree.traverse():
            if 'node_25' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_node_48(self):
        expected_state = {'Greece'}
        for node in tree.traverse():
            if 'node_48' == node.name:
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
