import os
import unittest

import numpy as np

from pastml import get_personalized_feature_name
from pastml.acr.acr import acr
from pastml.annotation import annotate_forest
from pastml.acr.maxlikelihood.ml import LH, LH_SF, MPPA, LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR, MARGINAL_PROBABILITIES
from pastml.acr.maxlikelihood.models import MODEL
from pastml.acr.maxlikelihood.models.SimpleModel import SCALING_FACTOR
from pastml.acr.maxlikelihood.models.F81Model import F81
from pastml.tree import read_forest, DATE, annotate_dates

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
STATES_INPUT = os.path.join(DATA_DIR, 'data.txt')


def reroot_tree_randomly():
    rerooted_tree = read_forest(TREE_NWK)[0]
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
        _.del_feature(DATE)
        if not _.name:
            _.name = 'unknown'
    annotate_forest([new_root], columns=feature, data=STATES_INPUT, data_sep=',')
    annotate_dates([new_root])
    return new_root


feature = 'Country'
tree = read_forest(TREE_NWK)[0]
_, column2states = annotate_forest([tree], columns=feature, data=STATES_INPUT, data_sep=',')
acr_result = acr([tree], character=feature, states=column2states[feature], prediction_method=MPPA, model=F81)


class ACRParameterOptimisationMPPAF81Test(unittest.TestCase):

    def test_likelihood(self):
        self.assertAlmostEqual(-110.178, acr_result[LOG_LIKELIHOOD], places=3,
                               msg='Likelihood was supposed to be the {:.3f}, got {:3f}'
                               .format(-110.178, acr_result[LOG_LIKELIHOOD]))

    def test_rerooted_values_are_the_same(self):
        for _ in range(5):
            rerooted_tree = reroot_tree_randomly()
            rerooted_acr_result = acr([rerooted_tree], character=feature, states=column2states[feature],
                                      prediction_method=MPPA, model=F81)
            for (state, freq, refreq) in zip(acr_result[MODEL].get_states(), acr_result[MODEL].get_frequencies(),
                                             rerooted_acr_result[MODEL].get_frequencies()):
                self.assertAlmostEqual(freq, refreq, places=2,
                                       msg='Frequency of {} for the original tree and rerooted tree '
                                           'were supposed to be the same, '
                                           'got {:.3f} vs {:3f}'
                                       .format(state, freq, refreq))
            value = acr_result[LOG_LIKELIHOOD]
            rerooted_value = rerooted_acr_result[LOG_LIKELIHOOD]
            self.assertAlmostEqual(value, rerooted_value, places=2,
                                   msg='{} for the original tree and rerooted tree were supposed to be the same, '
                                       'got {:.3f} vs {:3f}'
                                   .format(LOG_LIKELIHOOD, value, rerooted_value))
            value = acr_result[MODEL].get_sf()
            rerooted_value = rerooted_acr_result[MODEL].get_sf()
            self.assertAlmostEqual(value, rerooted_value, places=2,
                                   msg='{} for the original tree and rerooted tree were supposed to be the same, '
                                       'got {:.3f} vs {:3f}'
                                   .format(SCALING_FACTOR, value, rerooted_value))
            mps = acr_result[MARGINAL_PROBABILITIES]
            remps = rerooted_acr_result[MARGINAL_PROBABILITIES]
            for node_name in ('node_4', '02ALAY1660'):
                for loc in acr_result[MODEL].get_states():
                    value = mps.loc[node_name, loc]
                    revalue = remps.loc[node_name, loc]
                    self.assertAlmostEqual(value, revalue, places=2,
                                           msg='{}: Marginal probability of {} for the original tree and rerooted tree '
                                               'were supposed to be the same, got {:.3f} vs {:3f}'
                                           .format(node_name, loc, value, revalue))

    def test_restricted_likelihood(self):
        self.assertAlmostEqual(-111.662, acr_result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA)], places=3,
                               msg='Restricted likelihood was supposed to be the {:.3f}, got {:3f}'
                               .format(-111.662, acr_result[RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA)]))

    def test_sf(self):
        self.assertAlmostEqual(3.841, acr_result[MODEL].get_sf(), places=3,
                               msg='Scaling factor was supposed to be the {:.3f}, got {:3f}'
                               .format(3.841, acr_result[MODEL].get_sf()))

    def test_frequencies(self):
        for loc, expected_value in {'Africa': 0.082, 'Albania': 0.028, 'EastEurope': 0.081, 'Greece': 0.365,
                                    'WestEurope': 0.444}.items():
            value = acr_result[MODEL].get_frequencies()[np.where(acr_result[MODEL].get_states() == loc)][0]
            self.assertAlmostEqual(value, expected_value, places=3,
                                   msg='Frequency of {} was supposed to be the {:.3f}, got {:3f}'
                                   .format(loc, expected_value, value))

    def test_frequencies_sum_to_1(self):
        value = acr_result[MODEL].get_frequencies().sum()
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
        expected_values = {'Africa': 0.952, 'Albania': 0.001, 'EastEurope': 0.011,
                           'Greece': 0.011, 'WestEurope': 0.025}
        node_name = 'ROOT'
        mps = acr_result[MARGINAL_PROBABILITIES]
        for loc, expected_value in expected_values.items():
            value = mps.loc[node_name, loc]
            self.assertAlmostEqual(value, expected_value, places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the {:.3f}, got {:3f}'
                                   .format(node_name, loc, expected_value, value))

    def test_marginal_probs_internal_node(self):
        expected_values = {'Africa': 0.944, 'Albania': 0.000, 'EastEurope': 0.000,
                           'Greece': 0.001, 'WestEurope': 0.054}
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

