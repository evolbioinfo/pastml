import logging

import numpy as np

from pastml.models import ModelWithFrequencies, Model
from pastml.models.generator import get_diagonalisation, get_pij_matrix

GLM = 'GLM'
GLM_MATRICES = 'MATRICES'
GLM_COEFFICIENTS = 'COEFFICIENTS'
GLM_SELECTORS = 'SELECTORS'


def save_custom_rates(states, rate_matrix, outfile):
    np.savetxt(outfile, rate_matrix, delimiter=' ', fmt='%.18e', header=' '.join(states))


def load_GLM_matrix(infile):
    """
    Read input matrix from a file, and check that it looks ok.
    :param infile: input file path
    :return: tuple(states, matrix) with character states and the read matrix
    """
    try:
        rate_matrix = np.loadtxt(infile, dtype=np.float64, comments='#', delimiter=' ')
    except:
        raise ValueError('Failed to load a GLM matrix from file {}, please check the file.'.format(infile))
    if not len(rate_matrix.shape) == 2 or not rate_matrix.shape[0] == rate_matrix.shape[1]:
        raise ValueError('The input GLM matrix must be squared, but yours is {}.'.format('x'.join(rate_matrix.shape)))
    np.fill_diagonal(rate_matrix, 0)
    n = len(rate_matrix)
    if np.count_nonzero(rate_matrix) != n * (n - 1):
        logging.getLogger('pastml').warning('The GLM matrix contains zero rates (apart from the diagonal).')
    with open(infile, 'r') as f:
        states = f.readlines()[0]
        if not states.startswith('#'):
            raise ValueError('The GLM matrix file should start with state names, '
                             'separated by whitespaces and preceded by # .')
        states = np.array(states.strip('#').strip('\n').strip().split(' '), dtype=str)
        if len(states) != n:
            raise ValueError(
                'The number of specified state names ({}) does not correspond to the GLM matrix dimensions ({}x{}).'
                .format(len(states), *rate_matrix.shape))
    new_order = np.argsort(states)
    return states[new_order], np.array(rate_matrix)[:, new_order][new_order, :]


class GLMModel(ModelWithFrequencies):

    def __init__(self, states, forest_stats, parameter_file, coefficients=None, sf=None, tau=0,
                 optimise_tau=False, reoptimise=False, **kwargs):
        self._coefficients = None
        self._optimise_coefficients = reoptimise

        if 'frequency_smoothing' in kwargs:
            del kwargs['frequency_smoothing']

        # This will initialize the basic model with frequencies and read parameter values (GLM matrices and coefficients)
        ModelWithFrequencies.__init__(self, states=states, forest_stats=forest_stats,
                                      sf=sf, tau=tau, frequencies=np.ones(len(states), dtype=np.float64) / len(states),
                                      optimise_tau=optimise_tau,
                                      frequency_smoothing=False, reoptimise=reoptimise,
                                      parameter_file=parameter_file, **kwargs)
        self.name = GLM
        self._optimise_frequencies = False

        if self._coefficients is None:
            self._coefficients = coefficients if coefficients is not None \
                else np.ones(len(self.matrices), dtype=np.float64) / len(states)
            self._optimise_coefficients = True

        # We precalculate the diagonalization of the Lambda matrix here,
        # to use it for state change probability calculations (get_Pij).
        # These diagonalization need to be updated each time the coefficients are changed
        self.D_DIAGONAL, self.A, self.A_INV = get_diagonalisation(self.frequencies, self.get_rate_matrix())

    def get_rate_matrix(self):
        """
        Calculates the rate matrix from coefficients and input matrices
        :return: np.array containing the rate matrix
        """
        # multiply matrices by their coefficients and selectors
        weighted_matrices = np.array([m * c for (m, c) in zip(self.matrices, self.coefficients)])
        return weighted_matrices.sum(axis=0)

    def parse_parameters(self, params, reoptimise=False):
        # this will parse basic model parameters (scaling factor and smoothing factor)
        # and return a dictionary key->value with other named (by key) parameters and their values
        params = Model.parse_parameters(self, params, reoptimise)

        # We assume input GLM matrix filepaths are specified by key GLM_MATRICES and are semicolon-separated
        if GLM_MATRICES not in params.keys():
            raise ValueError('At least one GLM matrix must be given in the parameter file (parameter name "{}"), '
                             'when the model {} is used.'.format(GLM_MATRICES, GLM))
        matrix_files = params[GLM_MATRICES].split(';')
        states_matrices = [load_GLM_matrix(mf.strip()) for mf in matrix_files]
        if not states_matrices:
            raise ValueError('At least one GLM matrix must be given in the parameter file (parameter name "{}"), '
                             'when the model {} is used.'.format(GLM_MATRICES, GLM))
        self._matrices = []
        for (sts, mx) in states_matrices:
            if len(self.states) != len(sts) or not np.all(self.states == sts):
                raise ValueError('GLM matrices given in the parameter file are incompatible, '
                                 'as they correspond to different states, e.g. "{}" vs "{}"'
                                 .format(', '.join(self.states), ', '.join(sts)))
            self._matrices.append(mx)
        self._matrices = np.array(self._matrices)

        # Let's save the matrix location in order to be able to write it to the output file
        self._glm_matrix_location = params[GLM_MATRICES]

        # We assume input GLM coefficients (if given) are specified by key GLM_COEFFICIENTS and are semicolon-separated
        if GLM_COEFFICIENTS in params.keys():
            try:
                self._coefficients = np.array(params[GLM_COEFFICIENTS].strip().split(';')).astype(np.float64)
                n_coefficients = len(self._coefficients)
                n_matrices = len(self._matrices)
                if n_coefficients != n_matrices:
                    raise ValueError('The number of GLM coefficients ({}) given in the parameter file '
                                     'must correspond to the number of GLM matrices ({}) but it does not.'
                                     .format(n_coefficients, n_matrices))

                # TODO: perform some coefficient checks here.
                # Here we assume that
                # each coefficient c should satisfy: -1 <= c <= 1. TODO: check if this assumption is good
                if np.any(self._coefficients > 1):
                    raise ValueError('Coefficients given in parameters ({}) must all be not greater than 1, '
                                     'but yours are not. Please fix them.'
                                     .format(self._coefficients, self._coefficients.sum()))
                if np.any(self._coefficients < -1):
                    raise ValueError('Coefficients given in parameters ({}) must all be not less than -1, '
                                     'but yours are not. Please fix them.'
                                     .format(self._coefficients, self._coefficients.sum()))
            except:
                raise ValueError('GLM coefficients ({}) given in the parameter file are malformatted:'
                                 'they should be float numbers separated by semicolon (;).'
                                 .format(params[GLM_COEFFICIENTS]))
            self._optimise_coefficients = reoptimise

        # TODO: allow to specify selectors as well?
        return params

    @property
    def coefficients(self):
        return self._coefficients

    @coefficients.setter
    def coefficients(self, coefficients):
        if self._optimise_coefficients:
            self._coefficients = coefficients
        else:
            raise NotImplementedError('The coefficients are preset and cannot be changed.')
        self.D_DIAGONAL, self.A, self.A_INV = get_diagonalisation(self.frequencies, self.get_rate_matrix())

    @property
    def matrices(self):
        return self._matrices

    def get_Pij_t(self, t, *args, **kwargs):
        """
        Returns a function of t that calculates the probability matrix of substitutions i->j over time t,
        with the given rate matrix.

        :return: a function of t that calculates the probability matrix of substitutions i->j over time t.
        :rtype: lambda t: np.array
        """
        return get_pij_matrix(self.transform_t(t), self.D_DIAGONAL, self.A, self.A_INV)

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        # TODO: check and update this method

        # Basic model with frequencies parameters + GLM-specific ones (coefficients)
        return ModelWithFrequencies.get_num_params(self) \
            + (len(self.coefficients) if self._optimise_coefficients else 0)

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        # TODO: check and update this method

        ModelWithFrequencies.set_params_from_optimised(self, ps, **kwargs)
        if not self.extra_params_fixed():
            n_params = ModelWithFrequencies.get_num_params(self)
            # the parameters of the basic model with frequencies are stored in the first n_params potsitions
            # of the ps array, GLM-specific parameters are stored after

            n_coeff = len(self.coefficients)
            if self._optimise_coefficients:
                self.coefficients = ps[n_params: n_params + n_coeff]

    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        # TODO: check this method
        if not self.extra_params_fixed():
            # First put basic model with frequencies parameters, then GLM-specific ones (coefficients)
            return np.hstack((ModelWithFrequencies.get_optimised_parameters(self),
                              self.coefficients if self._optimise_coefficients else []))
        return Model.get_optimised_parameters(self)

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        # TODO: check this method
        if not self.extra_params_fixed():
            extras = []
            if self._optimise_coefficients:
                extras += [np.array([-1, 1], np.float64)] * len(self.coefficients)
            return np.array((*ModelWithFrequencies.get_bounds(self), *extras))
        return Model.get_bounds(self)

    @property
    def _print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tGLM coefficients\t{}\t{}\n' \
            .format(ModelWithFrequencies._print_parameters(self),
                    '; '.join('{:g}'.format(_) for _ in self.coefficients),
                    '(optimised)' if self._optimise_coefficients else '(fixed)')

    def freeze(self):
        """
        Prohibit parameter optimization.

        :return: void
        """
        ModelWithFrequencies.freeze(self)
        self._optimise_coefficients = False

    def save_parameters(self, filehandle):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filehandle: filehandle for the file where the parameter values should be written.
        :return: void
        """
        # Save basic model with frequencies parameters
        ModelWithFrequencies.save_parameters(self, filehandle)
        # Save GLM-specific parameters. # TODO: check and update
        filehandle.write('{}\t{}\n'.format(GLM_COEFFICIENTS, '; '.join('{:g}'.format(_) for _ in self.coefficients)))
        # Save the input GLM matrix locations which we memorized when reading the input parameters
        filehandle.write('{}\t{}\n'.format(GLM_MATRICES, self._glm_matrix_location))
