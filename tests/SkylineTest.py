import os
import unittest

import numpy as np
from ete3 import Tree

from pastml import MODEL_ID, get_personalized_feature_name
from pastml.acr import acr, model2class
from pastml.annotation import ForestStats, annotate_skyline
from pastml.ml import MPPA, LOG_LIKELIHOOD, LH, BU_LH, TD_LH, ALLOWED_STATES, MARGINAL_PROBABILITIES
from pastml.models import SCALING_FACTOR
from pastml.models.F81Model import F81, F81Model
from pastml.models.HKYModel import HKY_STATES
from pastml.models.JCModel import JCModel
from pastml.models.SkylineModel import SkylineModel
from pastml.tree import annotate_dates

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
TREE_NEXUS = os.path.join(DATA_DIR, 'hiv1C.nexus')
TREE_NWK = os.path.join(DATA_DIR, 'hiv1C.M184V.nwk')
PS_SKY = os.path.join(DATA_DIR, 'params.skyline.tab')
PS_NOSKY = os.path.join(DATA_DIR, 'params.noskyline.tab')
WD = os.path.join(DATA_DIR, 'skyline_test')

def get_forest():
    tree1 = Tree('(A:1,(B:0.5,C:1.5)BC:2)r1:0;', format=3)
    tree2 = Tree('(H:1,I:1.5,J:1)r2:1;', format=3)

    forest = [tree1, tree2]
    annotate_dates(forest, root_dates=[1902, 1904.5])

    annotations1 = {'A': {'Africa'},
                    'B': {'Africa', 'Europe'},
                    'C': set(),
                    'H': {'Europe'},
                    'J': {'Europe'},
                    'I': {'Europe'}}

    annotations2 = {'A': {'Africa'},
                    'B': {'Africa', 'Europe'},
                    'C': set(),
                    'H': {'UK'},
                    'J': {'France'},
                    'I': {'UK', 'France'}}

    for tree in forest:
        for n in tree:
            n.add_feature('loc1', annotations1[n.name])
            n.add_feature('loc2', annotations2[n.name])
    return forest


class SkylineTest(unittest.TestCase):

    def test_get_model_at_time(self):
        jc_model = JCModel(states=HKY_STATES, forest_stats=None, sf=1)
        sm = SkylineModel([jc_model, jc_model, jc_model], [10, 20], None, None)
        self.assertEqual(sm.get_model_id_at_time(5), 0)
        self.assertEqual(sm.get_model_id_at_time(10), 1)
        self.assertEqual(sm.get_model_id_at_time(12), 1)
        self.assertEqual(sm.get_model_id_at_time(20), 2)
        self.assertEqual(sm.get_model_id_at_time(30), 2)

    def test_skyline_mapping_same_states(self):
        forest = get_forest()
        start_date = -np.inf
        skyline_models = []
        skyline = [1905]
        n_sky = len(skyline) + 1
        param_dict = {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6}
        forest_stats = ForestStats(forest, 'loc1')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc1', start_date, end_date)
            m = F81Model(parameter_file=param_dict, rate_matrix_file=None,
                         states=['Africa', 'Europe'], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                      skyline_mapping=None, forest_stats=forest_stats)
        self.assertTrue(np.all(np.eye(2) == sky_model._skyline_mapping[(0, 1)]))
        self.assertTrue(np.all(np.eye(2) == sky_model._skyline_mapping[(1, 0)]))

    def test_skyline_annotation(self):
        forest = get_forest()
        skyline = [1905]
        annotate_skyline(forest, skyline, 'loc1')
        tree1, tree2 = forest
        self.assertEqual(0, getattr(tree1, MODEL_ID))
        self.assertEqual(0, getattr(tree1.children[0], MODEL_ID))
        self.assertEqual(0, getattr(tree1.children[1], MODEL_ID))
        self.assertEqual(0, getattr(tree1.children[1].children[0], MODEL_ID))
        self.assertEqual(1, getattr(tree1.children[1].children[1], MODEL_ID))
        self.assertEqual(0, getattr(tree2, MODEL_ID))
        self.assertEqual(1, getattr(tree2.children[0], MODEL_ID))
        self.assertEqual(1, getattr(tree2.children[1], MODEL_ID))
        self.assertEqual(1, getattr(tree2.children[2], MODEL_ID))

    def test_skyline_Pij(self):
        forest = get_forest()
        start_date = -np.inf
        skyline_models = []
        skyline = [1905]
        n_sky = len(skyline) + 1
        param_dict = {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6}
        forest_stats = ForestStats(forest, 'loc1')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc1', start_date, end_date)
            m = F81Model(parameter_file=param_dict, rate_matrix_file=None,
                         states=['Africa', 'Europe'], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                 skyline_mapping=None, forest_stats=forest_stats)

        C = next(t for t in forest[0] if 'C' == t.name)
        p = m.get_p_ij_child(C)

        annotate_skyline(forest, skyline, 'loc1')
        sky_p = sky_model.get_p_ij_child(C)
        print(sky_p, p)
        self.assertTrue(np.all(np.round(sky_p, 3) == np.round(p, 3)))

    def test_allowed_states(self):
        forest = get_forest()
        start_date = -np.inf
        skyline_models = []
        skyline = [1905]
        n_sky = len(skyline) + 1
        param_dict = {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6}
        forest_stats = ForestStats(forest, 'loc1')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc1', start_date, end_date)
            m = F81Model(parameter_file=param_dict, rate_matrix_file=None,
                         states=['Africa', 'Europe'], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                 skyline_mapping=None, forest_stats=forest_stats)
        annotate_skyline(forest, skyline, 'loc1')
        for tree in forest:
            for node in tree.traverse():
                self.assertTrue(np.all(sky_model.get_allowed_states(node, 'loc1') == m.get_allowed_states(node, 'loc1')))

    def test_skyline_Pji(self):
        forest = get_forest()
        start_date = -np.inf
        skyline_models = []
        skyline = [1905]
        n_sky = len(skyline) + 1
        param_dict = {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6}
        forest_stats = ForestStats(forest, 'loc1')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc1', start_date, end_date)
            m = F81Model(parameter_file=param_dict, rate_matrix_file=None,
                         states=['Africa', 'Europe'], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                 skyline_mapping=None, forest_stats=forest_stats)

        C = next(t for t in forest[0] if 'C' == t.name)
        p = m.get_p_ji_child(C)

        annotate_skyline(forest, skyline, 'loc1')
        sky_p = sky_model.get_p_ji_child(C)
        print(sky_p, p)
        self.assertTrue(np.all(np.round(sky_p, 3) == np.round(p, 3)))

    def test_parameter_skyline_same_params(self):
        forest_nosky = get_forest()
        param_dict = {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6}
        states = ['Africa', 'Europe']
        acr_result_nosky = acr(forest_nosky, character='loc1', states=states,
                               prediction_method=MPPA, model=F81,
                               parameters=[param_dict])

        skyline = [1905]
        forest_sky = get_forest()
        annotate_skyline(forest_sky, skyline, 'loc1')
        acr_result_sky = acr(forest_sky, character='loc1', states=states,
                             prediction_method=MPPA, model=[F81, F81],
                             parameters=[param_dict, param_dict], skyline=skyline)

        self.assertAlmostEqual(acr_result_sky[LOG_LIKELIHOOD], acr_result_nosky[LOG_LIKELIHOOD], places=5,
                               msg='Loglikelihood should be the same for exactly the same parameters '
                                   'with 2 and 1 skyline intervals.')
        mp_sky = acr_result_sky[MARGINAL_PROBABILITIES]
        mp_nosky = acr_result_nosky[MARGINAL_PROBABILITIES]

        for tree in forest_sky:
            for n in tree.traverse():
                for state in states:
                    self.assertAlmostEqual(mp_sky.loc[n.name, state], mp_nosky.loc[n.name, state])

        for tree_sky, tree_nosky in zip(forest_sky, forest_nosky):
            for n_sky, n_nosky in zip(tree_sky.traverse(), tree_nosky.traverse()):
                self.assertEqual(getattr(n_sky, 'loc1'), getattr(n_nosky, 'loc1'))

    def test_skyline_is_better_same_states(self):
        forest_nosky = get_forest()
        states = ['Africa', 'Europe']
        acr_result_nosky = acr(forest_nosky, character='loc1', states=states,
                               prediction_method=MPPA, model=F81)

        skyline = [1905]
        forest_sky = get_forest()
        annotate_skyline(forest_sky, skyline, 'loc1')
        acr_result_sky = acr(forest_sky, character='loc1', states=states,
                             prediction_method=MPPA, model=[F81, F81], skyline=skyline)
        print(acr_result_sky[LOG_LIKELIHOOD], acr_result_nosky[LOG_LIKELIHOOD])
        self.assertGreater(acr_result_sky[LOG_LIKELIHOOD], acr_result_nosky[LOG_LIKELIHOOD],
                               msg='Loglikelihood should be higher with the skyline.')



    #
    # def test_aic_noskyline(self):
    #     set_up_pastml_logger(True)
    #     states = np.array(['resistant', 'sensitive'])
    #     tree = read_forest(TREE_NEXUS)[0]
    #     annotate_dates([tree])
    #     name_tree(tree)
    #     simulate_states(tree, model=F81, frequencies=np.array([np.array([0.2, 0.8])]),
    #                     kappa=None, tau=0, sf=np.array([1 / 10.]),
    #                     character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)
    #
    #     for tip in tree:
    #         tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
    #         tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})
    #
    #     acr_result_nosky = acr(tree, columns=['state'], column2states={'state': states},
    #                            prediction_method=MPPA, model=F81, skyline=None)[0][0]
    #     acr_result_sky = acr(tree, columns=['state2'], column2states={'state2': states},
    #                          prediction_method=MPPA, model=F81, skyline=[1996])[0][0]
    #
    #     self.assertGreater(acr_result_sky[AIC], acr_result_nosky[AIC], msg='NO skyline model should be selected')
    #
    # def test_aic_skyline(self):
    #     set_up_pastml_logger(True)
    #     states = np.array(['resistant', 'sensitive'])
    #     tree = read_forest(TREE_NWK, columns=['state', 'state2'])[0]
    #     # tree = read_forest(TREE_NEXUS)[0]
    #     annotate_dates([tree])
    #     name_tree(tree)
    #     skyline = [2000]
    #     # annotate_skyline([tree], skyline=skyline, column2states={'M184V': states}, first_column='M184V')
    #     # simulate_states(tree, model=F81, frequencies=np.array([np.array([0.01, 1 - 0.01]), np.array([0.4, 0.6])]),
    #     #                 kappa=None, tau=0, sf=np.array([1 / 100., 1 / 10.]),
    #     #                 character='M184V', rate_matrix=None, n_repetitions=1, root_state_id=1)
    #     # remove_skyline([tree])
    #     # for tip in tree:
    #     #     tip.add_feature('state', {states[getattr(tip, 'M184V')][0]})
    #     #     tip.add_feature('state2', {states[getattr(tip, 'M184V')][0]})
    #     #
    #     # tree.write(outfile=TREE_NWK, features=[DATE, 'state', 'state2'], format=3)
    #
    #     acr_result_nosky = acr(tree, columns=['state'], column2states={'state': states},
    #                            prediction_method=MPPA, model=F81, skyline=None)[0][0]
    #     acr_result_sky = \
    #         acr(tree, columns=['state2'], column2states={'state2': states},
    #             prediction_method=MPPA, model=F81, skyline=skyline)[0][0]
    #
    #     self.assertGreater(acr_result_nosky[AIC], acr_result_sky[AIC], msg='Skyline model should be selected')
    #
    # def test_num_skyline_nodes(self):
    #     tree = read_forest(TREE_NEXUS)[0]
    #
    #     skyline = [1991]
    #     states = np.array(['resistant', 'sensitive'])
    #     annotate_skyline([tree], skyline=skyline, column2states={'M184V': states}, first_column='M184V')
    #
    #     n_sky = sum(1 for n in tree.traverse() if getattr(n, SKYLINE, False))
    #     self.assertEqual(3359, n_sky / 2, msg='3359 skyline nodes were expected, found {}'.format(n_sky))
    #
    # def test_skyline_node_dates(self):
    #     tree = read_forest(TREE_NEXUS)[0]
    #
    #     skyline = [1991]
    #     states = np.array(['resistant', 'sensitive'])
    #     annotate_skyline([tree], skyline=skyline, column2states={'M184V': states}, first_column='M184V')
    #
    #     for node in (n for n in tree.traverse() if getattr(n, SKYLINE, False)):
    #         self.assertEqual(getattr(node, DATE), 1991,
    #                          msg='Skyline node\'s date was supposed to be 1991, got {}'.format(getattr(node, DATE)))
    #         self.assertGreater(1991, getattr(node.up if node.dist else node.up.up, DATE),
    #                            msg='Skyline node\'s parent\'s date was supposed to be before 1991, got {}'
    #                            .format(getattr(node.up if node.dist else node.up.up, DATE)))
    #         self.assertGreater(
    #             getattr(node.children[0] if node.children[0].dist else node.children[0].children[0], DATE), 1991,
    #             msg='Skyline node\'s child\'s date was supposed to be after 1991, got {}'
    #                 .format(getattr(node.children[0] if node.children[0].dist else node.children[0].children[0], DATE)))
    #
    # def test_skyline_node_is_singular(self):
    #     tree = read_forest(TREE_NEXUS)[0]
    #
    #     skyline = [1991]
    #     annotate_skyline([tree], skyline=skyline,
    #                      first_column='M184V', column2states={'M184V': ['resistant', 'sensitive']}, skyline_mapping=None)
    #
    #     for node in (n for n in tree.traverse() if getattr(n, SKYLINE, False)):
    #         self.assertEqual(len(node.children), 1,
    #                          msg='Skyline node was supposed to have 1 child, got {}'.format(len(node.children)))
    #
    # def test_skyline_removal(self):
    #     tree = read_forest(TREE_NEXUS)[0]
    #
    #     skyline = [1991]
    #     states = np.array(['resistant', 'sensitive'])
    #     annotate_skyline([tree], skyline=skyline, column2states={'M184V': states}, first_column='M184V')
    #     remove_skyline([tree])
    #     n_sky = sum(1 for n in tree.traverse() if getattr(n, SKYLINE, False))
    #     self.assertEqual(0, n_sky, msg='Found {} skyline nodes that were supposed to be removed'.format(n_sky))
