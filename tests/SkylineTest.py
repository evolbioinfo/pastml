import os
import shutil
import unittest

import numpy as np

from pastml.annotation import annotate_skyline, remove_skyline
from pastml.models.f81_like import F81
from pastml.acr import acr, _serialize_acr, _set_up_pastml_logger
from pastml.file import get_pastml_parameter_file
from pastml.ml import MPPA, LOG_LIKELIHOOD
from pastml.utilities.state_simulator import simulate_states
from pastml.tree import read_forest, name_tree

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NEXUS = os.path.join(DATA_DIR, 'hiv1C.nexus')
WD = os.path.join(DATA_DIR, 'skyline_test')


class SkylineTest(unittest.TestCase):

    def test_parameter_saving(self):
        _set_up_pastml_logger(True)
        states = np.array(['resistant', 'sensitive'])
        tree = read_forest(TREE_NEXUS)[0]
        name_tree(tree)
        skyline = [1991]
        sk_nodes, sk_len = annotate_skyline([tree], skyline=skyline)
        simulate_states(tree, model=F81, frequencies=np.array([np.array([0.0001, 1 - 0.0001]), np.array([0.2, 0.8])]),
                        kappa=None, tau=0, sf=np.array([1 / 100., 1 / 10.]),
                        character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)
        remove_skyline(sk_nodes)

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

        # shutil.rmtree(WD)