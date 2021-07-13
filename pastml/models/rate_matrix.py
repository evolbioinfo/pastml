import logging

import numpy as np

from pastml.models.generator import get_pij_matrix, get_diagonalisation

CUSTOM_RATES = 'CUSTOM_RATES'


def get_custom_rate_pij(rate_matrix, frequencies):
    """
    Returns a function of t that calculates the probability matrix of substitutions i->j over time t,
    with the given rate matrix.

    :return: a function of t that calculates the probability matrix of substitutions i->j over time t.
    :rtype: lambda t: np.array
    """
    D_DIAGONAL, A, A_INV = get_diagonalisation(frequencies, rate_matrix)

    def get_pij(t):
        return get_pij_matrix(t, D_DIAGONAL, A, A_INV)

    return get_pij


def save_custom_rates(states, rate_matrix, outfile):
    np.savetxt(outfile, rate_matrix, delimiter=' ', fmt='%.18e', header=' '.join(states))


def load_custom_rates(infile):
    rate_matrix = np.loadtxt(infile, dtype=np.float64, comments='#', delimiter=' ')
    if not len(rate_matrix.shape) == 2 or not rate_matrix.shape[0] == rate_matrix.shape[1]:
        raise ValueError('The input rate matrix must be squared, but yours is {}.'.format('x'.join(rate_matrix.shape)))
    if not np.all(rate_matrix == rate_matrix.transpose()):
        raise ValueError('The input rate matrix must be symmetric, but yours is not.')
    np.fill_diagonal(rate_matrix, 0)
    n = len(rate_matrix)
    if np.count_nonzero(rate_matrix) != n * (n - 1):
        logging.getLogger('pastml').warning('The rate matrix contains zero rates (apart from the diagonal).')
    with open(infile, 'r') as f:
        states = f.readlines()[0]
        if not states.startswith('#'):
            raise ValueError('The rate matrix file should start with state names, '
                             'separated by whitespaces and preceded by # .')
        states = np.array(states.strip('#').strip('\n').strip().split(' '), dtype=str)
        if len(states) != n:
            raise ValueError(
                'The number of specified state names ({}) does not correspond to the rate matrix dimensions ({}x{}).'
                    .format(len(states), *rate_matrix.shape))
    return states, rate_matrix
