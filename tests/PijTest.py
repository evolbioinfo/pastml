import unittest

import numpy as np

from pastml.models.hky import A, G, T, C, get_hky_pij
from pastml.ml import get_f81_pij, get_mu
from pastml.models.generator import get_pij_matrix, get_diagonalisation


class PijTest(unittest.TestCase):

    def test_pij_f81(self):
        for _ in range(10):
            t = 10 * np.random.rand(1)[0]
            freqs = np.random.rand(10)
            freqs /= freqs.sum()
            p_ij = get_pij_matrix(t, *get_diagonalisation(freqs))
            mu = get_mu(freqs)
            p_ij_f81 = get_f81_pij(t, freqs, mu)
            self.assertTrue(np.allclose(p_ij, p_ij_f81),
                            msg='F81 P_ij calculation failed for time {} and frequencies {}'.format(t, freqs))

    def test_pij_jc(self):
        freqs = np.ones(10)
        freqs /= freqs.sum()
        for _ in range(10):
            t = 10 * np.random.rand(1)[0]
            p_ij = get_pij_matrix(t, *get_diagonalisation(freqs))
            mu = get_mu(freqs)
            p_ij_jc = get_f81_pij(t, freqs, mu)
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

            p_ij = get_pij_matrix(t, *get_diagonalisation(freqs, rate_matrix))
            p_ij_hky = get_hky_pij(t, freqs, kappa)
            self.assertTrue(np.allclose(p_ij, p_ij_hky),
                            msg='HKY P_ij calculation failed for time {} and frequencies {}'.format(t, freqs))
