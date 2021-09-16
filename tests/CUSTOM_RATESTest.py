import os
import unittest
import shutil

import numpy as np
from ete3 import Tree

from pastml.acr import acr, _serialize_acr, _set_up_pastml_logger
from pastml.file import get_pastml_parameter_file
from pastml.ml import MPPA, LOG_LIKELIHOOD, MARGINAL_PROBABILITIES, SCALING_FACTOR
from pastml.utilities.state_simulator import simulate_states
from pastml.models.rate_matrix import save_custom_rates, load_custom_rates, CUSTOM_RATES
from pastml.models.jtt import JTT_RATE_MATRIX, JTT_FREQUENCIES, JTT_STATES, JTT
from tree import annotate_dates

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
RM = os.path.join(DATA_DIR, 'rate_matrix.txt')
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')
WD = os.path.join(DATA_DIR, 'cr_vs_jtt_test')


class CUSTOM_RATESTest(unittest.TestCase):

    def test_rate_serialisation(self):
        save_custom_rates(JTT_STATES, JTT_RATE_MATRIX, RM)
        states, rate_matrix = load_custom_rates(RM)
        self.assertTrue(np.all(JTT_RATE_MATRIX == rate_matrix),
                        msg='JTT rate matrix was not saved properly')
        os.remove(RM)

    def test_state_serialisation(self):
        save_custom_rates(JTT_STATES, JTT_RATE_MATRIX, RM)
        states, rate_matrix = load_custom_rates(RM)
        self.assertTrue(np.all(JTT_STATES == states),
                        msg='JTT states were not saved properly')
        os.remove(RM)

    def test_tree_likelihood(self):
        _set_up_pastml_logger(True)
        tree = Tree(TREE_NWK, format=3)
        annotate_dates([tree])
        simulate_states(tree, JTT, [JTT_FREQUENCIES], kappa=np.array([1]), tau=0, sf=np.array([1]),
                        character='jtt', rate_matrix=None, n_repetitions=1)
        for tip in tree:
            tip.add_feature('state1', {JTT_STATES[getattr(tip, 'jtt')][0]})
            tip.add_feature('state2', {JTT_STATES[getattr(tip, 'jtt')][0]})

        acr_result_jtt = \
            acr(tree, columns=['state1'], column2states={'state1': JTT_STATES}, prediction_method=MPPA, model=JTT)[0][0]
        os.makedirs(WD, exist_ok=True)
        _serialize_acr((acr_result_jtt, WD, []))
        params = os.path.join(WD, get_pastml_parameter_file(MPPA, JTT, 'state1'))

        save_custom_rates(JTT_STATES, JTT_RATE_MATRIX, RM)
        acr_result_cr = acr(tree, columns=['state2'], prediction_method=MPPA, model=CUSTOM_RATES,
                            column2parameters={'state2': params}, column2rates={'state2': RM},
                            column2states={'state2': JTT_STATES})[0][0]
        self.assertEqual(acr_result_jtt[LOG_LIKELIHOOD], acr_result_cr[LOG_LIKELIHOOD],
                         msg='Likelihood should be the same for JTT and CR with JTT matrix')

        shutil.rmtree(WD)
        os.remove(RM)

    def test_marginal_probs_internal_nodes(self):
        _set_up_pastml_logger(True)
        tree = Tree(TREE_NWK, format=3)
        annotate_dates([tree])
        simulate_states(tree, JTT, [JTT_FREQUENCIES], kappa=np.array([1]), tau=0, sf=np.array([1]),
                        character='jtt', rate_matrix=None, n_repetitions=1)
        for tip in tree:
            tip.add_feature('state1', {JTT_STATES[getattr(tip, 'jtt')][0]})
            tip.add_feature('state2', {JTT_STATES[getattr(tip, 'jtt')][0]})

        acr_result_jtt = \
            acr(tree, columns=['state1'], column2states={'state1': JTT_STATES}, prediction_method=MPPA, model=JTT)[0][0]
        os.makedirs(WD, exist_ok=True)
        _serialize_acr((acr_result_jtt, WD, []))
        params = os.path.join(WD, get_pastml_parameter_file(MPPA, JTT, 'state1'))
        with open(params, 'r') as f:
            lines = [l.strip() for l in f.readlines() if not l.startswith(SCALING_FACTOR)]
        with open(params, 'w+') as f:
            f.write('\n'.join(lines))

        save_custom_rates(JTT_STATES, JTT_RATE_MATRIX, RM)
        acr_result_cr = acr(tree, columns=['state2'], prediction_method=MPPA, model=CUSTOM_RATES,
                            column2parameters={'state2': params}, column2rates={'state2': RM},
                            column2states={'state2': JTT_STATES})[0][0]
        self.assertEqual(acr_result_jtt[LOG_LIKELIHOOD], acr_result_cr[LOG_LIKELIHOOD],
                         msg='Likelihood should be the same for JTT and CR with JTT matrix')

        mps_jtt = acr_result_jtt[MARGINAL_PROBABILITIES]
        mps_cr = acr_result_cr[MARGINAL_PROBABILITIES]
        self.assertTrue(np.all(mps_jtt == mps_cr),
                        msg='Marginal probabilities be the same for JTT and CR with JTT matrix')

        shutil.rmtree(WD)
        os.remove(RM)
