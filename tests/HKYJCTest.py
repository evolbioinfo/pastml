import os
import unittest

import numpy as np

from pastml import get_personalized_feature_name
from pastml.acr.acr import acr
from pastml.acr.maxlikelihood.models.HKYModel import KAPPA, HKY
from pastml.annotation import annotate_forest
from pastml.acr.maxlikelihood.models import MODEL, SCALING_FACTOR
from pastml.acr.maxlikelihood.ml import LH, LH_SF, MPPA, LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR, MARGINAL_PROBABILITIES
from pastml.acr.maxlikelihood.models.JCModel import JC
from pastml.tree import read_forest

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.25.C_0.25.G_0.25.T_0.25.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.25.C_0.25.G_0.25.T_0.25.pastml.tab')

feature = 'ACR'


def read_tree():
    tree = read_forest(TREE_NWK)[0]
    _, column2states = annotate_forest([tree], columns=feature, data=STATES_INPUT)
    return tree, column2states[feature]


tree, states = read_tree()
acr_result_jc = acr([tree], character=feature, states=states, prediction_method=MPPA, model=JC)

tree, states = read_tree()
acr_result_hky = acr([tree], character=feature, prediction_method=MPPA, model=HKY, states=states,
                     parameters={KAPPA: 1, 'A': .25, 'T': .25, 'C': .25, 'G': .25})

tree, states = read_tree()
acr_result_hky_free_kappa = acr([tree], character=feature, prediction_method=MPPA, model=HKY, states=states,
                                parameters={'A': .25, 'T': .25, 'C': .25, 'G': .25})
tree, states = read_tree()
acr_result_hky_free_freqs = acr([tree], character=feature, prediction_method=MPPA, model=HKY, states=states,
                                parameters={KAPPA: 1})
tree, states = read_tree()
acr_result_hky_free = acr([tree], character=feature, prediction_method=MPPA, model=HKY, states=states)

print("Log lh for HKY-all-fixed {}, HKY-freqs-fixed {}, HKY-kappa-fixed {}, HKY {}"
      .format(acr_result_hky[LOG_LIKELIHOOD], acr_result_hky_free_kappa[LOG_LIKELIHOOD],
              acr_result_hky_free_freqs[LOG_LIKELIHOOD], acr_result_hky_free[LOG_LIKELIHOOD]))


class HKYJCTest(unittest.TestCase):

    def test_params(self):
        for param in (LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA)):
            self.assertAlmostEqual(acr_result_hky[param], acr_result_jc[param], places=3,
                                   msg='{} was supposed to be the same for two models'.format(param))
        value_hky = acr_result_hky[MODEL].get_sf()
        value_f81 = acr_result_jc[MODEL].get_sf()
        self.assertAlmostEqual(value_hky, value_f81, places=2,
                               msg='{} was supposed to be the same for two models'.format(SCALING_FACTOR))

    def test_hky_likelihood_is_better_kappa(self):
        self.assertGreater(acr_result_hky_free_kappa[LOG_LIKELIHOOD], acr_result_hky[LOG_LIKELIHOOD],
                           msg='Likelihood with free kappa was supposed to be better')

    def test_hky_likelihood_is_better_all_than_kappa(self):
        self.assertGreater(acr_result_hky_free[LOG_LIKELIHOOD], acr_result_hky_free_kappa[LOG_LIKELIHOOD],
                           msg='Likelihood with all free parameters was supposed to be better than just with free kappa')

    def test_hky_likelihood_is_better_all_than_freq(self):
        self.assertGreater(acr_result_hky_free[LOG_LIKELIHOOD], acr_result_hky_free_freqs[LOG_LIKELIHOOD],
                           msg='Likelihood with all free parameters was supposed to be better than just with free freqs')

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
        node_name = 'ROOT'
        mps_hky = acr_result_hky[MARGINAL_PROBABILITIES]
        mps_f81 = acr_result_jc[MARGINAL_PROBABILITIES]
        for state in acr_result_jc[MODEL].get_states():
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))

    def test_marginal_probs_internal_node(self):
        node_name = 'node_4'
        mps_hky = acr_result_hky[MARGINAL_PROBABILITIES]
        mps_f81 = acr_result_jc[MARGINAL_PROBABILITIES]
        for state in acr_result_jc[MODEL].get_states():
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))

    def test_marginal_probs_tip(self):
        node_name = '02ALAY1660'
        mps_hky = acr_result_hky[MARGINAL_PROBABILITIES]
        mps_f81 = acr_result_jc[MARGINAL_PROBABILITIES]
        for state in acr_result_jc[MODEL].get_states():
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))
