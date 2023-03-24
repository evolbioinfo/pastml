import os
import unittest

import numpy as np
import pandas as pd

from pastml.models.f81_like import JC
from pastml.tree import read_tree
from pastml import get_personalized_feature_name, STATES
from pastml.acr import acr
from pastml.ml import LH, LH_SF, MPPA, LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR, CHANGES_PER_AVG_BRANCH, \
    SCALING_FACTOR, FREQUENCIES, MARGINAL_PROBABILITIES

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')


def reroot_tree_randomly():
    rerooted_tree = read_tree(TREE_NWK)
    new_root = np.random.choice([_ for _ in rerooted_tree.traverse()
                                 if not _.is_root() and not _.up.is_root() and _.dist])
    old_root_child = rerooted_tree.children[0]
    old_root_child_dist = old_root_child.dist
    other_children = list(rerooted_tree.children[1:])
    old_root_child.up = None
    for child in other_children:
        child.up = None
        old_root_child.add_child(child, dist=old_root_child_dist + child.dist)
    old_root_child.set_outgroup(new_root)
    new_root = new_root.up
    for _ in new_root.traverse():
        if not _.name:
            _.name = 'unknown'
    return new_root


feature = 'Country'
df = pd.read_csv(STATES_INPUT, index_col=0, header=0)[[feature]]
tree = read_tree(TREE_NWK)
acr_result = acr(tree, df, prediction_method=MPPA, model=JC)[0]


class ACRParameterOptimisationMPPAJCTest(unittest.TestCase):

    def test_rerooted_values_are_the_same(self):
        for _ in range(5):
            rerooted_tree = reroot_tree_randomly()
            rerooted_acr_result = acr(rerooted_tree, df, prediction_method=MPPA, model=JC)[0]
            for (state, freq, refreq) in zip(acr_result[STATES], acr_result[FREQUENCIES],
                                             rerooted_acr_result[FREQUENCIES]):
                self.assertAlmostEqual(freq, refreq, places=2,
                                       msg='Frequency of {} for the original tree and rerooted tree '
                                           'were supposed to be the same, '
                                           'got {:.3f} vs {:3f}'
                                       .format(state, freq, refreq))
            for label in (LOG_LIKELIHOOD, CHANGES_PER_AVG_BRANCH, SCALING_FACTOR):
                value = acr_result[label]
                rerooted_value = rerooted_acr_result[label]
                self.assertAlmostEqual(value, rerooted_value, places=2,
                                       msg='{} for the original tree and rerooted tree were supposed to be the same, '
                                           'got {:.3f} vs {:3f}'
                                       .format(label, value, rerooted_value))
            mps = acr_result[MARGINAL_PROBABILITIES]
            remps = rerooted_acr_result[MARGINAL_PROBABILITIES]
            for node_name in ('node_4', '02ALAY1660'):
                for loc in acr_result[STATES]:
                    value = mps.loc[node_name, loc]
                    revalue = remps.loc[node_name, loc]
                    self.assertAlmostEqual(value, revalue, places=2,
                                           msg='{}: Marginal probability of {} for the original tree and rerooted tree '
                                               'were supposed to be the same, got {:.3f} vs {:3f}'
                                           .format(node_name, loc, value, revalue))

    def test_likelihood(self):
        self.assertAlmostEqual(-121.873, acr_result[LOG_LIKELIHOOD], places=3,
                               msg='Likelihood was supposed to be the {:.3f}, got {:3f}'
                               .format(-121.873, acr_result[LOG_LIKELIHOOD]))

    def test_restricted_likelihood(self):
        self.assertAlmostEqual(-123.421, acr_result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA)], places=3,
                               msg='Restricted likelihood was supposed to be the {:.3f}, got {:3f}'
                               .format(-123.421, acr_result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA)]))

    def test_changes_per_avg_branch(self):
        self.assertAlmostEqual(0.138, acr_result[CHANGES_PER_AVG_BRANCH], places=3,
                               msg='Scaling factor was supposed to be the {:.3f} changes per avg branch, got {:3f}'
                               .format(0.138, acr_result[CHANGES_PER_AVG_BRANCH]))

    def test_sf(self):
        self.assertAlmostEqual(4.951, acr_result[SCALING_FACTOR], places=3,
                               msg='Scaling factor was supposed to be the {:.3f}, got {:3f}'
                               .format(4.951, acr_result[SCALING_FACTOR]))

    def test_frequencies(self):
        value = acr_result[FREQUENCIES]
        expected_value = np.ones(len(value), np.float64) / len(value)
        self.assertListEqual(value.tolist(), expected_value.tolist(),
                             msg='Frequencies were supposed to be the {}, got {}'.format(expected_value, value))

    def test_frequencies_sum_to_1(self):
        value = acr_result[FREQUENCIES].sum()
        self.assertAlmostEqual(value, 1, places=3,
                               msg='Frequencies were supposed to sum to 1, not to {:3f}'.format(value))

    def test_likelihood_same_for_all_nodes(self):
        """
        Tests if marginal likelihoods were correctly calculated
        by comparing the likelihoods of all the nodes (should be all the same).
        """
        lh_feature = get_personalized_feature_name(feature, LH)
        lh_sf_feature = get_personalized_feature_name(feature, LH_SF)

        for node in tree.traverse():
            if not node.is_root() and not (node.is_leaf() and node.dist == 0):
                node_loglh = np.log10(getattr(node, lh_feature).sum()) - getattr(node, lh_sf_feature)
                parent_loglh = np.log10(getattr(node.up, lh_feature).sum()) - getattr(node.up, lh_sf_feature)
                self.assertAlmostEqual(node_loglh, parent_loglh, places=2,
                                       msg='Likelihoods of {} and {} were supposed to be the same.'
                                       .format(node.name, node.up.name))

    def test_marginal_probs_root(self):
        expected_values = {'Africa': 0.819, 'Albania': 0.020, 'EastEurope': 0.045,
                           'Greece': 0.020, 'WestEurope': 0.095}
        node_name = 'ROOT'
        mps = acr_result[MARGINAL_PROBABILITIES]
        for loc, expected_value in expected_values.items():
            value = mps.loc[node_name, loc]
            self.assertAlmostEqual(value, expected_value, places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the {:.3f}, got {:3f}'
                                   .format(node_name, loc, expected_value, value))

    def test_marginal_probs_internal_node(self):
        expected_values = {'Africa': 0.686, 'Albania': 0.002, 'EastEurope': 0.003,
                           'Greece': 0.002, 'WestEurope': 0.307}
        node_name = 'node_4'
        mps = acr_result[MARGINAL_PROBABILITIES]
        for loc, expected_value in expected_values.items():
            value = mps.loc[node_name, loc]
            self.assertAlmostEqual(value, expected_value, places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the {:.3f}, got {:3f}'
                                   .format(node_name, loc, expected_value, value))

    def test_marginal_probs_tip(self):
        expected_values = {'Africa': 0, 'Albania': 1, 'EastEurope': 0, 'Greece': 0, 'WestEurope': 0}
        node_name = '02ALAY1660'
        mps = acr_result[MARGINAL_PROBABILITIES]
        for loc, expected_value in expected_values.items():
            value = mps.loc[node_name, loc]
            self.assertAlmostEqual(value, expected_value, places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the {:.3f}, got {:3f}'
                                   .format(node_name, loc, expected_value, value))
