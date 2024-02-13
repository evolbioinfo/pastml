import unittest

import numpy as np

from pastml.acr.maxlikelihood.models.CustomRatesModel import CustomRatesModel
from pastml.acr.maxlikelihood.models.F81Model import F81Model
from pastml.acr.maxlikelihood.models.HKYModel import HKYModel, A, G, T, C, HKY_STATES
from pastml.acr.maxlikelihood.models.JCModel import JCModel


class PijTest(unittest.TestCase):

    def test_pij_f81(self):
        for _ in range(10):
            t = 10 * np.random.rand(1)[0]
            freqs = np.random.rand(10)
            freqs /= freqs.sum()
            model = CustomRatesModel(sf=1, states=np.array(list('ABCDEFGHIJ')), forest_stats=None, frequencies=freqs,
                                     rate_matrix=np.ones(shape=(10, 10), dtype=np.float64) - np.eye(10))
            p_ij = model.get_Pij_t(t)
            model = F81Model(sf=1, forest_stats=None, frequencies=freqs, states=np.array(list('ABCDEFGHIJ')))
            p_ij_f81 = model.get_Pij_t(t)

            self.assertTrue(np.allclose(p_ij, p_ij_f81),
                            msg='F81 P_ij calculation failed for time {} and frequencies {}'.format(t, freqs))

    def test_pij_jc(self):
        freqs = np.ones(10)
        freqs /= freqs.sum()
        for _ in range(10):
            t = 10 * np.random.rand(1)[0]
            model = CustomRatesModel(sf=1, states=np.array(list('ABCDEFGHIJ')), forest_stats=None, frequencies=freqs,
                                     rate_matrix=np.ones(shape=(10, 10), dtype=np.float64) - np.eye(10))
            p_ij = model.get_Pij_t(t)
            model = JCModel(sf=1, forest_stats=None, states=np.array(list('ABCDEFGHIJ')))
            p_ij_jc = model.get_Pij_t(t)
            self.assertTrue(np.allclose(p_ij, p_ij_jc),
                            msg='JC P_ij calculation failed for time {} and frequencies {}'.format(t, freqs))

    def test_pij_hky(self):
        n = 4
        for _ in range(10):
            t = 10 * np.random.rand(1)[0]
            kappa = 20 * np.random.rand(1)[0]
            freqs = np.random.rand(n)
            freqs /= freqs.sum()
            rate_matrix = np.ones(shape=(n, n), dtype=np.float64) - np.eye(n)
            rate_matrix[A, G] = kappa
            rate_matrix[C, T] = kappa
            rate_matrix[G, A] = kappa
            rate_matrix[T, C] = kappa

            model = CustomRatesModel(sf=1, states=HKY_STATES, forest_stats=None, frequencies=freqs,
                                     rate_matrix=rate_matrix)
            p_ij = model.get_Pij_t(t)
            model = HKYModel(sf=1, forest_stats=None, kappa=kappa, frequencies=freqs)
            p_ij_hky = model.get_Pij_t(t)
            self.assertTrue(np.allclose(p_ij, p_ij_hky),
                            msg='HKY P_ij calculation failed for time {} and frequencies {}'.format(t, freqs))
