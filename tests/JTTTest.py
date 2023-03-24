import unittest

import numpy as np

from pastml.models.generator import get_normalised_generator
from pastml.models.jtt import JTT_RATE_MATRIX, JTT_A_INV, JTT_A, JTT_D_DIAGONAL, JTT_FREQUENCIES


class JTTTest(unittest.TestCase):

    def test_rate_matrix_is_symmetric(self):
        self.assertTrue(np.allclose(JTT_RATE_MATRIX, JTT_RATE_MATRIX.T),
                        msg='JTT rate matrix was supposed to be symmetric')

    def test_jtt_diagonalisation(self):
        self.assertTrue(np.allclose(get_normalised_generator(JTT_FREQUENCIES, JTT_RATE_MATRIX),
                                    JTT_A.dot(np.diag(JTT_D_DIAGONAL)).dot(JTT_A_INV)),
                        msg='JTT generator diagonalisation failed')
