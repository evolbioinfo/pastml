import os
import unittest
from collections import Counter

from pastml.acr import acr
from pastml.annotation import annotate_forest
from pastml.parsimony import ACCTRAN, STEPS
from pastml.tree import collapse_zero_branches, read_forest

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')

feature = 'Country'
tree = read_forest(TREE_NWK)[0]
collapse_zero_branches([tree])
_, column2states = annotate_forest([tree], columns=feature, data=STATES_INPUT, data_sep=',')
acr_result = acr([tree], character=feature, states=column2states[feature], prediction_method=ACCTRAN)


class ACRStateAcctranTest(unittest.TestCase):

    def test_num_steps(self):
        self.assertEqual(32, acr_result[STEPS],
                         msg='Was supposed to have {} parsimonious steps, got {}.'.format(32, acr_result[STEPS]))

    def test_num_nodes(self):
        state2num = Counter()
        for node in tree.traverse():
            state = getattr(node, feature)
            if len(state) > 1:
                state2num['unresolved'] += 1
            else:
                state2num[next(iter(state))] += 1
        expected_state2num = {'unresolved': 2, 'Africa': 107, 'Albania': 50, 'Greece': 70, 'WestEurope': 32, 'EastEurope': 16}
        self.assertDictEqual(expected_state2num, state2num, msg='Was supposed to have {} as states counts, got {}.'
                             .format(expected_state2num, state2num))

    def test_state_root(self):
        expected_state = {'Africa'}
        state = getattr(tree, feature)
        self.assertSetEqual(expected_state, state,
                         msg='Root state was supposed to be {}, got {}.'.format(expected_state, state))

    def test_state_resolved_node_129(self):
        expected_state = {'Greece'}
        for node in tree.traverse():
            if 'node_129' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_resolved_node_25(self):
        expected_state = {'WestEurope'}
        for node in tree.traverse():
            if 'node_25' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                 .format(node.name, expected_state, state))
                break

    def test_state_unresolved_node_21(self):
        expected_state = {'WestEurope', 'Greece'}
        for node in tree.traverse():
            if 'node_21' == node.name:
                state = getattr(node, feature)
                self.assertSetEqual(expected_state, state, msg='{} state was supposed to be {}, got {}.'
                                    .format(node.name, expected_state, state))
                break

    def test_state_resolved_node_32(self):
        expected_state = {'WestEurope'}
        for node in tree.traverse():
            if 'node_32' == node.name:
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
