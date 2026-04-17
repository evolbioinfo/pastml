import logging
import os

import numpy as np
import pandas as pd
from scipy.linalg import expm

from pastml.models import Model
from pastml.models.generator import get_diagonalisation, get_pij_matrix, get_normalised_generator

GLM = 'GLM'
GLM_PREDICTOR_DIR = 'GLM_PREDICTOR_DIR'
GLM_COEFFICIENT = 'GLM_COEFFICIENT'
GLM_INDICATOR = 'GLM_INDICATOR'

EPSILON = 1e-3


def read_predictors(directory, states, sep=',', epsilon=EPSILON):
    """
    Reads user-provided predictors, shifts if there are non-positive values,
    converts them to log scale and returns a tuple, whose first values contains the character states,
    the second one file names, and the third log-scaled (potentially shifted) predictors in the same order.

    :param states: list of possible character states,
        which should be contained in to the row and column names of the predictor files.
    :param directory: User-defined directory containing the predictor files.
        Each file should be a CSV with a header line, where the first column contains character state names,
         and the remaining columns contain predictor values.
         Cell ij should contain the predictor value for the transition from state i to state j.
         The predictor values should be non-negative, but if there are any non-positive values,
         they will be shifted to make them positive before taking the log.
         The diagonal will be set to zero (on the log scale).
    :param sep: separator used in the predictor files (default is a comma).
    :param epsilon: minimal value to consider as positive,
        if the minimal predictor value (before log-scaling) is lower than that,
        all its value will be shifted.
    :return: (states, predictor_names, predictor_log_matrices)
    """
    df_state_set = None
    state_set = set(states)
    names, predictors = [], []
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        df = pd.read_table(file_path, sep=sep, index_col=0, header=0)
        if df_state_set is None:
            df_state_set = set(df.index)
            if state_set - df_state_set:
                raise ValueError(f'The predictor file {file_name} does not contain the following states found at tree nodes: '
                                 + ", ".join(sorted(state_set - df_state_set)))
            if len(df_state_set) > len(state_set):
                logging.getLogger('pastml')\
                    .warning(f'The predictor file {file_name} contains additional states with respect to the annotated tree nodes: '
                             + ", ".join(sorted(df_state_set - state_set)))
                state_set = df_state_set
                states = sorted(state_set | df_state_set)
        # check that the state names in the current file match those in the first file
        if state_set != set(df.index):
            raise ValueError(f'All predictor files must have the same character state names, '
                             f'but {file_name} has different state names in the first column'
                             + ", ".join(sorted((state_set ^ set(df.index)))))

        todo, todo_names = [], []
        if len(df.columns) == 1:
            logging.getLogger('pastml') \
                .warning(f'The predictor file {file_name} contains only one column, will treat it as source/destination.')
            predictor_vector = df.loc[states, df.columns[0]].to_numpy(dtype=float, na_value=0)
            predictor_src = np.tile(predictor_vector.reshape(-1, 1), (1, len(state_set)))
            predictor_tgt = np.tile(predictor_vector, (len(state_set), 1))
            todo = [predictor_src, predictor_tgt]
            todo_names = [f'src_{file_name}', f'tgt_{file_name}']
        else:
            if state_set != set(df.columns):
                raise ValueError(f'All predictor files must have the same character state names, '
                                     f'but {file_name} has different state names in the header: '
                                     + ", ".join(sorted((state_set ^ set(df.columns[1:])))))
            predictor = df.loc[states, states].to_numpy(dtype=float, na_value=0)
            todo = [predictor]
            todo_names = [file_name]
        for predictor, predictor_name in zip(todo, todo_names):
            # if it is a binary matrix we will not log-scale it
            non_binary = np.any(~np.isin(predictor, [0, 1]))
            if non_binary:
                # Let's fill in the diagonal with 1s before shifting the non-positive values, and refill it with zeros after.
                np.fill_diagonal(predictor, 1)
                # Shift the predictor values if there are non-positive values, to make them positive before taking the log.
                min_value = predictor.min()
                if min_value < epsilon:
                    predictor += (epsilon - min_value)

                # normalize the predictor
                np.fill_diagonal(predictor, 1)
                predictor /= predictor.max()

                # Set the diagonal to 1, so that it gets to be zero with the log.
                np.fill_diagonal(predictor, 1)
            names.append(predictor_name)
            predictors.append(np.log(predictor) if non_binary else predictor)
    return states, names, predictors

class GLMModel(Model):

    def __init__(self, states, forest_stats, sf=None, tau=0, parameter_file=None, coefficients=None, indicators=None,
                 optimise_sf=True, optimise_tau=False, reoptimise=False, predictors=None, optimise_coefficients=True,
                 optimise_indicators=True, **kwargs):
        self._optimise_coefficients = optimise_coefficients
        self._optimise_indicators = optimise_indicators
        self._predictor_names = np.array(predictors[0]) if predictors is not None else np.array([]) # shape: (n_predictors,)
        self._predictors = np.stack(predictors[1]) if predictors is not None else np.array([]) # shape: (n_predictors, n_states, n_states)
        n_predictors = len(self._predictors)
        self._coefficients = np.array(coefficients, dtype=np.float64) if coefficients is not None \
            else np.ones(n_predictors, dtype=np.float64) / (n_predictors if n_predictors > 0 else 1)
        self._indicators = np.array(indicators, dtype=bool) if indicators is not None \
            else np.ones(n_predictors, dtype=bool)
        # This will initialize the basic model
        Model.__init__(self, states=states, forest_stats=forest_stats,
                       sf=sf, tau=tau, optimise_sf=optimise_sf,
                       optimise_tau=optimise_tau, reoptimise=reoptimise,
                       parameter_file=parameter_file, **kwargs)
        self.name = GLM

        # We precalculate the diagonalization of the Lambda matrix here,
        # to use it for state change probability calculations (get_Pij).
        # These diagonalization need to be updated each time the coefficients are changed
        self.Q = get_normalised_generator(rate_matrix=self.get_rate_matrix())

    def get_rate_matrix(self):
        """
        Calculates the rate matrix from coefficients and input matrices

        :return: np.array containing the rate matrix
        """
        logL = np.einsum('i,ijk->jk', self.coefficients[self.indicators], self._predictors[self.indicators])
        np.fill_diagonal(logL, -np.inf)
        L = np.exp(logL)
        np.fill_diagonal(L, 0)
        return L

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        For a GLM model, apart from the basic parameters (scaling factor and smoothing factor, see pastml.models.Model),
        the input might contain:
        (1) the input matrices (mandatory). The key for this parameter is pastml.models.GLMModel.GLM_MATRICES,
            and the value contains semicolon-separated paths to the files with matrices;
        (2) GLM coefficient values (optional).  The key for this parameter is pastml.models.GLMModel.GLM_COEFFICIENTS,
            and the value contains semicolon-separated coefficients (in order of the corresponding input matrices);

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """

        # this will parse basic model parameters (scaling factor and smoothing factor)
        # and return a dictionary key->value with other named (by key) parameters and their values
        params = Model.parse_parameters(self, params, reoptimise)

        if GLM_PREDICTOR_DIR in params.keys():
            states, names, predictors = read_predictors(params[GLM_PREDICTOR_DIR], self.states)
            self.states = np.array(states)
            self._predictor_names = np.array(names)
            self._predictors = np.stack(predictors)
            n_predictors = len(self._predictors)
            self._coefficients = np.ones(n_predictors, dtype=np.float64) / (n_predictors if n_predictors > 0 else 1)
            self._indicators = np.ones(n_predictors, dtype=bool)

        # We assume input GLM coefficients (if given) are specified by key GLM_COEFFICIENTS and are semicolon-separated
        glm_coefficient_keys = [_ for _ in params.keys() if _.startswith(f'{GLM_COEFFICIENT}:')]
        if glm_coefficient_keys:
            n_predictors = len(self._predictors)
            self._coefficients = np.ones(n_predictors, dtype=np.float64) / (n_predictors if n_predictors > 0 else 1)
            for i, name in enumerate(self._predictor_names):
                if f'{GLM_COEFFICIENT}:{name}' not in params.keys():
                    raise ValueError(f'Some GLM coefficients are given in the parameter file, '
                                     f'but the one for predictor "{name}" is missing. '
                                     f'It is required as the predictor is given. Please fix the parameter file.')
                try:
                    self._coefficients[i] = float(params[f'{GLM_COEFFICIENT}:{name}'])
                except:
                    raise ValueError(f'GLM coefficient for predictor "{name}" given in the parameter file is malformatted:'
                                     f'it should be a float number, but it is not. Please fix it.')
                if self._coefficients[i] > 1 or self._coefficients[i] < -1:
                    raise ValueError('GLM coefficients given in parameters must all be between -1 and 1, '
                                     f'but the coefficient for {name} is {self._coefficients[i]}. Please fix it.')
            self._optimise_coefficients = reoptimise

        # We assume input GLM coefficients (if given) are specified by key GLM_COEFFICIENTS and are semicolon-separated
        glm_indicator_keys = [_ for _ in params.keys() if _.startswith(f'{GLM_INDICATOR}:')]
        if glm_indicator_keys:
            n_predictors = len(self._predictors)
            self._indicators = np.ones(n_predictors, dtype=bool)
            for i, name in enumerate(self._predictor_names):
                if f'{GLM_INDICATOR}:{name}' not in params.keys():
                    raise ValueError(f'Some GLM indicators are given in the parameter file, '
                                     f'but the one for predictor "{name}" is missing. '
                                     f'It is required as the predictor is given. Please fix the parameter file.')
                try:
                    self._indicators[i] = bool(int(params[f'{GLM_INDICATOR}:{name}']))
                except:
                    raise ValueError(f'GLM indicator for predictor "{name}" given in the parameter file is malformatted:'
                                     f'it should be either 0 or 1. Please fix it.')
        return params

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def indicators(self):
        return self._indicators

    @indicators.setter
    def indicators(self, inds):
        if self._optimise_indicators:
            self._indicators = inds
        else:
            raise NotImplementedError('The indicators are preset and cannot be changed.')
        # If the coefficients just got changed, we need to update our precomputed generator
        self.Q = get_normalised_generator(rate_matrix=self.get_rate_matrix())

    @coefficients.setter
    def coefficients(self, coefficients):
        if self._optimise_coefficients:
            self._coefficients = np.array(coefficients, dtype=np.float64)
        else:
            raise NotImplementedError('The coefficients are preset and cannot be changed.')
        # If the coefficients just got changed, we need to update our precomputed generator
        self.Q = get_normalised_generator(rate_matrix=self.get_rate_matrix())

    @property
    def predictors(self):
        return self._predictors

    def get_Pij_t(self, t, *args, **kwargs):
        """
        Returns a function of t that calculates the probability matrix of substitutions i->j over time t,
        with the given rate matrix.

        :return: a function of t that calculates the probability matrix of substitutions i->j over time t.
        :rtype: lambda t: np.array
        """
        return expm(self.Q * self.transform_t(t))
        # return get_pij_matrix(self.transform_t(t), self.D_DIAGONAL, self.A, self.A_INV)

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        # TODO: check and update this method

        # Basic model with frequencies parameters + GLM-specific ones
        # (as many coefficients as the indicators allow us to pick, if we optimize them)
        return Model.get_num_params(self) \
            + (sum(self.indicators) if self._optimise_coefficients else 0)

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        # TODO: check and update this method

        Model.set_params_from_optimised(self, ps, **kwargs)
        if not self.extra_params_fixed():
            n_params = Model.get_num_params(self)
            # the parameters of the basic model are stored in the first n_params positions
            # of the ps array, GLM-specific parameters are stored after

            # only set the coefficients that are selected by the indicators
            n_coeff = sum(self.indicators)
            if self._optimise_coefficients:
                coefficients = np.array(self.coefficients)
                coefficients[self.indicators] = ps[n_params: n_params + n_coeff]
                self.coefficients = coefficients



    def set_parameters_with_optuna(self, trial):
        """
        Set model parameters during optuna optimization

        :return: void, update this model
        """
        # Model.set_parameters_with_optuna(self, trial)
        if not self.extra_params_fixed():
            n_predictors = len(self.predictors)
            if self._optimise_indicators:
                self.indicators = np.array([trial.suggest_categorical(f"GLM_ind_{i}", [0, 1]) \
                                            for i in range(n_predictors)], dtype=bool)
            # if self._optimise_coefficients:
            #     coeffs = np.array(self.coefficients)
            #     coeffs[self.indicators] = np.array([trial.suggest_float(f"GLM_coeff_{i}", -1, 1) \
            #                                         for i in np.arange(0, n_predictors)[self.indicators]],
            #                                        dtype=np.float64)
            #     self.coefficients = coeffs



    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        if not self.extra_params_fixed():
            # First put basic model with frequencies parameters,
            # then GLM-specific ones (coefficients that are selected by indicators, if we optimize them)
            return np.hstack((Model.get_optimised_parameters(self),
                              self.coefficients[self.indicators] \
                                  if self._optimise_coefficients else []))
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
                # putting 1e-6 as a very small number close to zero, in order not to allow for zero itself
                extras += [np.array([-1, 1], np.float64)] * sum(self.indicators)
            return np.array((*Model.get_bounds(self), *extras))
        return Model.get_bounds(self)

#    @property
    def _print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tGLM coefficients\t{}\n' \
               '{}\n'.format(Model._print_parameters(self),
                             '(optimised)' if self._optimise_coefficients else '(fixed)',
                             '\n'.join(f'\t\t{name}:\t{coeff:g}' for (name, coeff) \
                                       in zip(self._predictor_names[self.indicators], self.coefficients[self.indicators])))

    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        Model.freeze(self)
        self._optimise_coefficients = False

    def save_parameters(self, filehandle):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filehandle: filehandle for the file where the parameter values should be written.
        :return: void
        """
        # Save basic model with frequencies parameters
        Model.save_parameters(self, filehandle)
        # Save GLM-specific parameters.
        for (name, coeff) in zip(self._predictor_names[self.indicators], self.coefficients[self.indicators]):
            filehandle.write('{}:{}\t{:g}\n'.format(GLM_COEFFICIENT, name, coeff))
            filehandle.write('{}:{}\t{}\n'.format(GLM_INDICATOR, name, 1))
        for name in self._predictor_names[~self.indicators]:
            filehandle.write('{}:{}\t{:g}\n'.format(GLM_COEFFICIENT, name, 0))
            filehandle.write('{}:{}\t{}\n'.format(GLM_INDICATOR, name, 0))

