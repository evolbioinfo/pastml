import os
import shutil
import unittest

import numpy as np

from pastml import SKYLINE
from pastml.acr import acr, _serialize_acr, _set_up_pastml_logger, _parse_pastml_parameters
from pastml.annotation import annotate_skyline, remove_skyline
from pastml.file import get_pastml_parameter_file
from pastml.ml import MPPA, LOG_LIKELIHOOD, MAP, JOINT, RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR, AIC
from pastml.models.f81_like import F81
from pastml.tree import read_forest, name_tree, DATE
from pastml.utilities.state_simulator import simulate_states

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NEXUS = os.path.join(DATA_DIR, 'hiv1C.nexus')
PS_SKY = os.path.join(DATA_DIR, 'params.skyline.tab')
PS_NOSKY = os.path.join(DATA_DIR, 'params.noskyline.tab')
WD = os.path.join(DATA_DIR, 'skyline_test')


class SkylineTest(unittest.TestCase):

    def test_parameter_saving(self):
        _set_up_pastml_logger(True)
        states = np.array(['resistant', 'sensitive'])
        tree = read_forest(TREE_NEXUS)[0]
        name_tree(tree)
        skyline = [1991]
        annotate_skyline([tree], skyline=skyline)
        simulate_states(tree, model=F81, frequencies=np.array([np.array([0.0001, 1 - 0.0001]), np.array([0.2, 0.8])]),
                        kappa=None, tau=0, sf=np.array([1 / 100., 1 / 10.]),
                        character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)
        remove_skyline([tree])

        for tip in tree:
            tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
            tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})
        tree.add_feature('state', {'sensitive'})
        tree.add_feature('state2', {'sensitive'})

        acr_result = acr(tree, columns=['state'], column2states={'state': states}, prediction_method=MPPA, model=F81,
                         skyline=skyline)[0]
        os.makedirs(WD, exist_ok=True)
        _serialize_acr((acr_result, WD))
        params = os.path.join(WD, get_pastml_parameter_file(MPPA, F81, 'state'))

        acr_result_cr = acr(tree, columns=['state2'], prediction_method=MPPA, model=F81,
                            column2parameters={'state2': params}, column2states={'state2': states},
                            skyline=skyline)[0]
        self.assertAlmostEqual(acr_result[LOG_LIKELIHOOD], acr_result_cr[LOG_LIKELIHOOD], places=5,
                               msg='Likelihood should be the same for initial and extracted from parameters calculation')

        shutil.rmtree(WD)

    def test_parameter_skyline_same_params(self):
        _set_up_pastml_logger(True)
        states = np.array(['resistant', 'sensitive'])
        tree = read_forest(TREE_NEXUS)[0]
        frequencies, sf, kappa, tau = _parse_pastml_parameters(PS_NOSKY, states, len(tree), reoptimise=False,
                                                               skyline_len=2)
        simulate_states(tree, model=F81, frequencies=frequencies,
                        kappa=kappa, tau=tau, sf=sf, character='M184V', rate_matrix=None, n_repetitions=1)

        for tip in tree:
            tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
            tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})

        acr_result_sky = acr(tree, columns=['state'],
                             column2parameters={'state': PS_SKY},
                             column2states={'state': states}, prediction_method=MPPA,
                             skyline=[1996])[0]

        acr_result_nosky = acr(tree, columns=['state2'], prediction_method=MPPA,
                               column2parameters={'state2': PS_NOSKY}, column2states={'state2': states})[0]
        for method in (MAP, MPPA, JOINT):
            restricted_lh = RESTRICTED_LOG_LIKELIHOOD_FORMAT_STR.format(method)
            self.assertAlmostEqual(acr_result_sky[restricted_lh], acr_result_nosky[restricted_lh], places=5,
                                   msg='{} should be the same  for exactly the same parameters '
                                       'with 2 and 1 skyline intervals'.format(restricted_lh))

        self.assertAlmostEqual(acr_result_sky[LOG_LIKELIHOOD], acr_result_nosky[LOG_LIKELIHOOD], places=5,
                               msg='Loglikelihood should be the same for exactly the same parameters '
                                   'with 2 and 1 skyline intervals.')

    def test_aic_noskyline(self):
        _set_up_pastml_logger(True)
        states = np.array(['resistant', 'sensitive'])
        tree = read_forest(TREE_NEXUS)[0]
        name_tree(tree)
        simulate_states(tree, model=F81, frequencies=np.array([np.array([0.2, 0.8])]),
                        kappa=None, tau=0, sf=np.array([1 / 10.]),
                        character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)

        for tip in tree:
            tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
            tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})

        acr_result_nosky = acr(tree, columns=['state'], column2states={'state': states},
                               prediction_method=MPPA, model=F81, skyline=None)[0]
        acr_result_sky = \
            acr(tree, columns=['state2'], column2states={'state2': states},
                prediction_method=MPPA, model=F81, skyline=[1996])[0]

        self.assertGreater(acr_result_sky[AIC], acr_result_nosky[AIC], msg='NO skyline model should be selected')

    def test_aic_skyline(self):
        _set_up_pastml_logger(True)
        states = np.array(['resistant', 'sensitive'])
        tree = read_forest(TREE_NEXUS)[0]
        skyline = [2000]
        annotate_skyline([tree], skyline=skyline)
        simulate_states(tree, model=F81, frequencies=np.array([np.array([0.01, 1 - 0.01]), np.array([0.4, 0.6])]),
                        kappa=None, tau=0, sf=np.array([1 / 100., 1 / 10.]),
                        character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)
        remove_skyline([tree])

        for tip in tree:
            tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
            tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})

        acr_result_nosky = acr(tree, columns=['state'], column2states={'state': states},
                               prediction_method=MPPA, model=F81, skyline=None)[0]
        acr_result_sky = \
            acr(tree, columns=['state2'], column2states={'state2': states},
                prediction_method=MPPA, model=F81, skyline=skyline)[0]

        self.assertGreater(acr_result_nosky[AIC], acr_result_sky[AIC], msg='Skyline model should be selected')

    def test_num_skyline_nodes(self):
        tree = read_forest(TREE_NEXUS)[0]

        skyline = [1991]
        annotate_skyline([tree], skyline=skyline)

        n_sky = sum(1 for n in tree.traverse() if getattr(n, SKYLINE, False))
        self.assertEqual(3359, n_sky, msg='3359 skyline nodes were expected, found {}'.format(n_sky))

    def test_skyline_node_dates(self):
        tree = read_forest(TREE_NEXUS)[0]

        skyline = [1991]
        annotate_skyline([tree], skyline=skyline)

        for node in (n for n in tree.traverse() if getattr(n, SKYLINE, False)):
            self.assertEqual(getattr(node, DATE), 1991,
                             msg='Skyline node\'s date was supposed to be 1991, got {}'.format(getattr(node, DATE)))
            self.assertGreater(1991, getattr(node.up, DATE),
                               msg='Skyline node\'s parent\'s date was supposed to be before 1991, got {}'
                               .format(getattr(node.up, DATE)))
            self.assertGreater(getattr(node.children[0], DATE), 1991,
                               msg='Skyline node\'s child\'s date was supposed to be after 1991, got {}'
                               .format(getattr(node.children[0], DATE)))

    def test_skyline_node_is_singular(self):
        tree = read_forest(TREE_NEXUS)[0]

        skyline = [1991]
        annotate_skyline([tree], skyline=skyline)

        for node in (n for n in tree.traverse() if getattr(n, SKYLINE, False)):
            self.assertEqual(len(node.children), 1,
                             msg='Skyline node was supposed to have 1 child, got {}'.format(len(node.children)))

    def test_skyline_removal(self):
        tree = read_forest(TREE_NEXUS)[0]

        skyline = [1991]
        annotate_skyline([tree], skyline=skyline)
        remove_skyline([tree])
        n_sky = sum(1 for n in tree.traverse() if getattr(n, SKYLINE, False))
        self.assertEqual(0, n_sky, msg='Found {} skyline nodes that were supposed to be removed'.format(n_sky))
