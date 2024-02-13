import unittest

import numpy as np

from pastml.acr.maxlikelihood.models.JTTModel import JTT_RATE_MATRIX


class JTTTest(unittest.TestCase):

    def test_rate_matrix_is_symmetric(self):
        self.assertTrue(np.allclose(JTT_RATE_MATRIX, JTT_RATE_MATRIX.T),
                        msg='JTT rate matrix was supposed to be symmetric')
