import os
import unittest

import numpy as np
from ete3 import Tree

from pastml.acr import acr
from pastml.annotation import ForestStats
from pastml.ml import MPPA, LOG_LIKELIHOOD, MARGINAL_PROBABILITIES
from pastml.models import SCALING_FACTOR
from pastml.models.F81Model import F81, F81Model
from pastml.models.HKYModel import HKY_STATES
from pastml.models.JCModel import JCModel, JC
from pastml.models.SkylineModel import SkylineModel, annotate_skyline, parse_skyline_mapping, MODEL_ID
from pastml.tree import annotate_dates

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')
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
                self.assertTrue(
                    np.all(sky_model.get_allowed_states(node, 'loc1') == m.get_allowed_states(node, 'loc1')))

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

        annotate_skyline(forest, skyline, 'loc1')
        sky_p_ij = sky_model.get_p_ij_child(C)
        sky_p_ji = sky_model.get_p_ji_child(C)

        freq0 = skyline_models[0].get_frequencies()
        freq1 = skyline_models[1].get_frequencies()
        # Check reversibility
        for i in range(2):
            for j in range(2):
                print(freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])
                self.assertAlmostEqual(freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])

        # Check it's the same as without the skyline
        p_ij = skyline_models[1].get_p_ij_child(C)
        p_ji = skyline_models[1].get_p_ji_child(C)
        self.assertTrue(np.all(np.round(sky_p_ij, 3) == np.round(p_ij, 3)))
        self.assertTrue(np.all(np.round(sky_p_ji, 3) == np.round(p_ji, 3)))


    def test_skyline_Pij_different_parameters(self):
        forest = get_forest()
        start_date = -np.inf
        skyline_models = []
        skyline = [1905]
        n_sky = len(skyline) + 1
        parameters = [{SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6},
                      {SCALING_FACTOR: 0.1, 'Africa': 0.25, 'Europe': 0.75}]
        forest_stats = ForestStats(forest, 'loc1')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc1', start_date, end_date)
            m = F81Model(parameter_file=parameters[i], rate_matrix_file=None,
                         states=['Africa', 'Europe'], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                 skyline_mapping=None, forest_stats=forest_stats)

        C = next(t for t in forest[0] if 'C' == t.name)

        annotate_skyline(forest, skyline, 'loc1')
        sky_p_ij = sky_model.get_p_ij_child(C)
        sky_p_ji = sky_model.get_p_ji_child(C)

        freq0 = skyline_models[0].get_frequencies()
        freq1 = skyline_models[1].get_frequencies()
        # Check reversibility
        for i in range(2):
            for j in range(2):
                print(freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])
                self.assertAlmostEqual(freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])

        # Check it's not the same as without the skyline
        p_ij = skyline_models[1].get_p_ij_child(C)
        p_ji = skyline_models[1].get_p_ji_child(C)
        self.assertFalse(np.all(np.round(sky_p_ij, 3) == np.round(p_ij, 3)))
        self.assertFalse(np.all(np.round(sky_p_ji, 3) == np.round(p_ji, 3)))

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

    def test_skyline_mapping_parsing(self):
        skyline_mapping = os.path.join(DATA_DIR, 'skyline_mapping.tab')
        with open(skyline_mapping, 'w+') as f:
            f.write("loc2\t1905\nFrance\tEurope\nUK\tEurope\nAfrica\tAfrica")

        skyline_mapping, all_states = parse_skyline_mapping('loc2', [1905], skyline_mapping)

        self.assertTrue(np.all(np.array(['Africa', 'Europe']) == all_states[0]))
        self.assertTrue(np.all(np.array(['Africa', 'France', 'UK']) == all_states[1]))

        self.assertTrue(np.all(skyline_mapping[(0, 1)] == np.array([[1, 0, 0], [0, 1, 1]])))
        self.assertTrue(np.all(skyline_mapping[(1, 0)] == np.array([[1, 0], [0, 1], [0, 1]])))

    def test_skyline_Pij_different_states(self):
        skyline = [1905]
        skyline_mapping = os.path.join(DATA_DIR, 'skyline_mapping.tab')
        with open(skyline_mapping, 'w+') as f:
            f.write("loc2\t1905\nFrance\tEurope\nUK\tEurope\nAfrica\tAfrica")
        forest = get_forest()
        skyline, skyline_mapping = annotate_skyline(forest, skyline, 'loc2', skyline_mapping)
        start_date = -np.inf
        skyline_models = []
        n_sky = len(skyline) + 1
        params = [{SCALING_FACTOR: 0.1, 'Africa': 0.4, 'Europe': 0.6},
                  {SCALING_FACTOR: 0.1, 'Africa': 0.4, 'France': 0.3, 'UK': 0.3}]
        forest_stats = ForestStats(forest, 'loc2')
        for i in range(n_sky):
            end_date = skyline[i] if i < len(skyline) else np.inf
            sub_forest_stats = ForestStats(forest, 'loc2', start_date, end_date)
            m = F81Model(parameter_file=params[i], rate_matrix_file=None,
                         states=skyline_mapping[1][i], forest_stats=sub_forest_stats)
            skyline_models.append(m)
        sky_model = SkylineModel(models=skyline_models, dates=skyline,
                                 skyline_mapping=skyline_mapping[0], forest_stats=forest_stats)

        C = next(t for t in forest[0] if 'C' == t.name)
        sky_p_ij = sky_model.get_p_ij_child(C)
        sky_p_ji = sky_model.get_p_ji_child(C)

        # nosky_p = skyline_models[1].get_p_ij_child(C)
        self.assertEqual((2, 3), sky_p_ij.shape)
        self.assertEqual((3, 2), sky_p_ji.shape)

        freq0 = skyline_models[0].get_frequencies()
        freq1 = skyline_models[1].get_frequencies()
        # Check reversibility
        for i in range(len(freq0)):
            for j in range(len(freq1)):
                print(i, j, freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])
                self.assertAlmostEqual(freq0[i] * sky_p_ij[i, j], freq1[j] * sky_p_ji[j, i])
