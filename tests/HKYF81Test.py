import os
import unittest

import numpy as np
import pandas as pd

from pastml import get_personalized_feature_name, STATES
from pastml.acr import acr
from pastml.models.hky import KAPPA, HKY
from pastml.ml import LH, LH_SF, MPPA, LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR, \
    CHANGES_PER_AVG_BRANCH, SCALING_FACTOR, FREQUENCIES, MARGINAL_PROBABILITIES
from pastml.models.f81_like import F81
from pastml.tree import read_tree

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NWK = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.6.C_0.15.G_0.2.T_0.05.nwk')
STATES_INPUT = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.6.C_0.15.G_0.2.T_0.05.pastml.tab')
TREE_NWK_JC = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.25.C_0.25.G_0.25.T_0.25.nwk')
STATES_INPUT_JC = os.path.join(DATA_DIR, 'tree.152taxa.sf_0.5.A_0.25.C_0.25.G_0.25.T_0.25.pastml.tab')

feature = 'ACR'
df = pd.read_table(STATES_INPUT, index_col=0, header=0)[[feature]]
acr_result_f81 = acr(read_tree(TREE_NWK), df, prediction_method=MPPA, model=F81)[0]

tree = read_tree(TREE_NWK)
acr_result_hky = acr(tree, df, prediction_method=MPPA, model=HKY, column2parameters={feature: {KAPPA: 1}})[0]
acr_result_hky_free = acr(read_tree(TREE_NWK), df, prediction_method=MPPA, model=HKY)[0]

print("Log lh for HKY-kappa-fixed {}, HKY {}"
      .format(acr_result_hky[LOG_LIKELIHOOD], acr_result_hky_free[LOG_LIKELIHOOD]))


class HKYF81Test(unittest.TestCase):

    def test_params(self):
        for param in (LOG_LIKELIHOOD, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(MPPA), CHANGES_PER_AVG_BRANCH,
                      SCALING_FACTOR):
            self.assertAlmostEqual(acr_result_hky[param], acr_result_f81[param], places=3,
                                   msg='{} was supposed to be the same for two models'.format(param))

    def test_hky_likelihood_is_better(self):
        self.assertGreater(acr_result_hky_free[LOG_LIKELIHOOD], acr_result_hky[LOG_LIKELIHOOD],
                           msg='Likelihood with free kappa was supposed to be better')

    def test_frequencies(self):
        for state in acr_result_f81[STATES]:
            value_f81 = acr_result_f81[FREQUENCIES][np.where(acr_result_f81[STATES] == state)][0]
            value_hky = acr_result_hky[FREQUENCIES][np.where(acr_result_hky[STATES] == state)][0]
            self.assertAlmostEqual(value_f81, value_hky, places=3,
                                   msg='Frequency of {} was supposed to be the same for two models'
                                   .format(state))

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
        mps_f81 = acr_result_f81[MARGINAL_PROBABILITIES]
        for state in acr_result_f81[STATES]:
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))

    def test_marginal_probs_internal_node(self):
        node_name = 'node_4'
        mps_hky = acr_result_hky[MARGINAL_PROBABILITIES]
        mps_f81 = acr_result_f81[MARGINAL_PROBABILITIES]
        for state in acr_result_f81[STATES]:
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))

    def test_marginal_probs_tip(self):
        node_name = '02ALAY1660'
        mps_hky = acr_result_hky[MARGINAL_PROBABILITIES]
        mps_f81 = acr_result_f81[MARGINAL_PROBABILITIES]
        for state in acr_result_f81[STATES]:
            self.assertAlmostEqual(mps_f81.loc[node_name, state], mps_hky.loc[node_name, state], places=3,
                                   msg='{}: Marginal probability of {} was supposed to be the same for two models'
                                   .format(node_name, state))
