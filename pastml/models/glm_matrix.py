import logging

import numpy as np
import os

from pastml.models.generator import get_pij_matrix, get_diagonalisation



def save_custom_rates(states, rate_matrix, outfile):
    np.savetxt(outfile, rate_matrix, delimiter=' ', fmt='%.18e', header=' '.join(states))


def load_glm_matrix(infile):
    glm_matrix = np.loadtxt(infile, dtype=np.float64, comments='#', delimiter=' ')
    glm_matrix_name = os.path.basename(os.path.normpath(infile))
    #glm_matrix.shape returns the dimensions of the matrix, for example 5x5 (5,5) so [0] is 5
    if not len(glm_matrix.shape) == 2 or not glm_matrix.shape[0] == glm_matrix.shape[1]:
        raise ValueError('The input factor matrix must be squared, but yours is {}.'.format('x'.join(glm_matrix.shape)))
    if not np.all(glm_matrix == glm_matrix.transpose()):
        raise ValueError('The input factor matrix must be symmetric, but yours is not.')
    #puts zeros in the diagonal
    np.fill_diagonal(glm_matrix, 0)
    n = len(glm_matrix)
    if np.count_nonzero(glm_matrix) != n * (n - 1):
        logging.getLogger('pastml').warning('The factor matrix contains zero rates (apart from the diagonal).')
    with open(infile, 'r') as f:
        states = f.readlines()[0]
        if not states.startswith('#'):
            raise ValueError('The factor matrix file should start with state names, '
                             'separated by whitespaces and preceded by # .')
        states = np.array(states.strip('#').strip('\n').strip().split(' '), dtype=str)
        if len(states) != n:
            raise ValueError(
                'The number of specified state names ({}) does not correspond to the factor matrix dimensions ({}x{}). Please check '
                'specific state names including white spaces between them'
                    .format(len(states), *glm_matrix.shape))
    return states, glm_matrix, glm_matrix_name
