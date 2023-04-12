import os
import unittest

import numpy as np

from pastml.annotation import ForestStats
from pastml.models.GLMModel import GLMModel, GLM_MATRICES
from pastml.models.generator import save_matrix
from pastml.tree import read_tree

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'data')

# TODO: use your favourite tree instead
TREE_NWK = os.path.join(DATA_DIR, 'Albanian.tree.152tax.tre')


class HKYJCTest(unittest.TestCase):

    def test_rate_matrix(self):
        tree = read_tree(TREE_NWK)
        states = np.array('Africa EastEurope WestEurope Albania Greece'.split(' '))
        n = len(states)

        # Let's create a matrix where all rates are 1, but Albania-East Europe is 2
        matrix_1 = np.ones(shape=(n, n)) - np.eye(n)
        matrix_1[1, 3] = 2
        matrix_1[3, 1] = 2
        m1_file = os.path.join(DATA_DIR, 'm1.txt')
        save_matrix(states=states, matrix=matrix_1, outfile=m1_file)

        # Let's create another matrix where all rates are 1, but Albania-Greece is 10
        matrix_2 = np.ones(shape=(n, n)) - np.eye(n)
        matrix_2[3, 4] = 10
        matrix_2[4, 3] = 10
        m2_file = os.path.join(DATA_DIR, 'm2.txt')
        save_matrix(states=states, matrix=matrix_2, outfile=m2_file)

        # Let's save them into an input parameter file for GLM
        param_file = os.path.join(DATA_DIR, 'glm_params.tab')
        with open(param_file, 'w+') as f:
            f.write('parameter\tvalue\n')
            f.write('{}\t{}'.format(GLM_MATRICES, '; '.join((m1_file, m2_file))))

        # let's repeat the test 10 times with different coefficients:
        for _ in range(10):
            # Let's create random coefficients between 0 (exclusive) and 1 (inclusive) each:
            # numpy random creates a value v: 0 <= v < 1, hence 0 < 1 - v <= 1.
            coefficients = 1 - np.random.random(2)

            model = GLMModel(states=states,
                             forest_stats=ForestStats([tree]), parameter_file=param_file,
                             coefficients=coefficients)

            expected_rate_matrix = coefficients[0] * matrix_1 + coefficients[1] * matrix_2
            # The GLM matrix will be sorted by the lexicographical orger of its states,
            # so let's sort the expected matrix too:
            new_order = np.argsort(states)
            expected_rate_matrix = expected_rate_matrix[:, new_order][new_order, :]

            rate_matrix = model.get_rate_matrix()

            self.assertTrue(np.all(expected_rate_matrix == rate_matrix),
                                   msg='The GLM rate matrix calculation failed:\n\tcoefficients are: {},\n'
                                       '\tmatrix1 is \n{},\n'
                                       '\tmatrix2 is \n{},\n'
                                       '\t expected result is \n{},\n'
                                       '\t model result is \n{}.'
                            .format(coefficients, matrix_1, matrix_2, expected_rate_matrix, rate_matrix))

        # remove temporary GLM files
        try:
            os.remove(m1_file)
            os.remove(m2_file)
            os.remove(param_file)
        except:
            pass
