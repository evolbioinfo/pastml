import logging

import numpy as np

from pastml.models import ModelWithFrequencies
from pastml.models.generator import get_diagonalisation, get_pij_matrix

CUSTOM_RATES = 'CUSTOM_RATES'


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
    new_order = np.argsort(states)
    return states[new_order], np.array(rate_matrix)[:, new_order][new_order, :]


class CustomRatesModel(ModelWithFrequencies):

    def __init__(self, forest_stats, sf=None, frequencies=None, rate_matrix_file=None, states=None, rate_matrix=None, tau=0,
                 optimise_tau=False, frequency_smoothing=False, parameter_file=None, reoptimise=False, **kwargs):
        ModelWithFrequencies.__init__(self, states=states, forest_stats=forest_stats,
                                      sf=sf, tau=tau, frequencies=frequencies, optimise_tau=optimise_tau,
                                      frequency_smoothing=frequency_smoothing, reoptimise=reoptimise,
                                      parameter_file=parameter_file, **kwargs)
        self.name = CUSTOM_RATES
        if rate_matrix_file is None and (rate_matrix is None or states is None):
            raise ValueError('Either the rate matrix file '
                             'or the rate matrix plus the states must be specified for {} model'.format(CUSTOM_RATES))
        if rate_matrix_file is None:
            # States are already set in the super constructor
            self._rate_matrix = rate_matrix
        else:
            self._states, self._rate_matrix = load_custom_rates(rate_matrix_file)
        self.D_DIAGONAL, self.A, self.A_INV = get_diagonalisation(self.frequencies, self._rate_matrix)

    @property
    def rate_matrix(self):
        return self._rate_matrix

    @rate_matrix.setter
    def rate_matrix(self, rate_matrix):
        raise NotImplementedError('The rate matrix is preset and cannot be changed.')

    @ModelWithFrequencies.frequencies.setter
    def frequencies(self, frequencies):
        if self._optimise_frequencies or self._frequency_smoothing:
            self._frequencies = frequencies
        else:
            raise NotImplementedError('The frequencies are preset and cannot be changed.')
        self.D_DIAGONAL, self.A, self.A_INV = get_diagonalisation(frequencies, self.rate_matrix)

    def get_Pij_t(self, t, *args, **kwargs):
        """
        Returns a function of t that calculates the probability matrix of substitutions i->j over time t,
        with the given rate matrix.

        :return: a function of t that calculates the probability matrix of substitutions i->j over time t.
        :rtype: lambda t: np.array
        """
        return get_pij_matrix(self.transform_t(t), self.D_DIAGONAL, self.A, self.A_INV)

