import logging

import numpy as np

from pastml.acr.maxlikelihood.models.SimpleModel import SimpleModel

FREQUENCIES = 'frequencies'


class ModelWithFrequencies(SimpleModel):

    def __init__(self, states, forest_stats, sf=None, frequencies=None, tau=0,
                 optimise_tau=False, frequency_smoothing=False, parameter_file=None, reoptimise=False, **kwargs):
        self._frequencies = None
        self._optimise_frequencies = not frequency_smoothing
        self._frequency_smoothing = frequency_smoothing
        SimpleModel.__init__(self, states, forest_stats=forest_stats,
                             sf=sf, tau=tau, optimise_tau=optimise_tau, reoptimise=reoptimise,
                             parameter_file=parameter_file, **kwargs)
        if self._frequencies is None:
            self._frequencies = frequencies if frequencies is not None \
                else np.ones(len(states), dtype=np.float64) / len(states)
        self._observed_frequencies = None

    def get_frequencies(self, node=None, **kwargs):
        return self._frequencies

    @property
    def observed_frequencies(self):
        if self._observed_frequencies is None:
            self._observed_frequencies = np.zeros(len(self._states), dtype=np.float64)
            for i, state in enumerate(self._states):
                self._observed_frequencies[i] = self.forest_stats.observed_frequencies[state]
            if np.any(self._observed_frequencies <= 0):
                logging.getLogger('pastml').debug('Some of the model states were not observed at tips, '
                                                  'we are updating state frequencies '
                                                  'as if we observed each state in at least 1/2 of a case.')
                self._observed_frequencies *= self.forest_stats.n_tips
                self._observed_frequencies = np.maximum(self._observed_frequencies, 1 / 2)
                self._observed_frequencies /= self._observed_frequencies.sum()
        return self._observed_frequencies

    def set_frequencies(self, frequencies):
        if self._optimise_frequencies or self._frequency_smoothing:
            self._frequencies = frequencies
        else:
            raise NotImplementedError('The frequencies are preset and cannot be changed.')

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        n_basic_params = SimpleModel.get_num_params(self)
        if self.extra_params_fixed():
            return n_basic_params
        return n_basic_params + ((len(self._frequencies) - 1)
                                 if self._optimise_frequencies else (1 if self._frequency_smoothing else 0))

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        SimpleModel.set_params_from_optimised(self, ps, **kwargs)
        if not self.extra_params_fixed():
            n_freq = len(self._frequencies)
            n_params = SimpleModel.get_num_params(self)

            freqs = self._frequencies

            if self._optimise_frequencies:
                freqs = np.hstack((ps[n_params: n_params + (n_freq - 1)], [1.]))
                freqs /= freqs.sum()
                self._frequencies = freqs
            elif self._frequency_smoothing:
                freqs = freqs * self.forest_stats.n_tips
                freqs += ps[n_params]
                freqs /= freqs.sum()
                self._frequencies = freqs

    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        if not self.extra_params_fixed():
            return np.hstack((SimpleModel.get_optimised_parameters(self),
                              self._frequencies[:-1] / self._frequencies[-1] if self._optimise_frequencies
                              else ([0] if self._frequency_smoothing else [])))
        return SimpleModel.get_optimised_parameters(self)

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        if not self.extra_params_fixed():
            extras = []
            if self._optimise_frequencies:
                extras += [np.array([1e-6, 10e6], np.float64)] * (len(self._states) - 1)
            if self._frequency_smoothing:
                extras.append(np.array([0, self.forest_stats.n_nodes]))
            return np.array((*SimpleModel.get_bounds(self), *extras))
        return SimpleModel.get_bounds(self)

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        For a model with frequencies, apart from the basic parameters
        (scaling factor and smoothing factor, see pastml.models.Model),
        the input might contain the frequency values:
        the key for each frequency value is the name of the corresponding character state.

        :param params: dict {key->value}
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values (same as input)
        """
        params = SimpleModel.parse_parameters(self, params, reoptimise)
        logger = logging.getLogger('pastml')
        known_freq_states = set(self.get_states()) & set(params.keys())
        if known_freq_states:
            unknown_freq_states = [state for state in self.get_states() if state not in params.keys()]
            if unknown_freq_states and not reoptimise:
                logger.error('Frequencies for some of the states ({}) are missing, '
                             'ignoring the specified frequencies.'.format(', '.join(unknown_freq_states)))
            else:
                self._frequencies = np.array([params[state] if state in params.keys() else 0
                                              for state in self.get_states()])
                try:
                    self._frequencies = self._frequencies.astype(np.float64)
                    if np.round(self._frequencies.sum() - 1, 2) != 0 and not reoptimise:
                        logger.error('Frequencies given in parameters ({}) do not sum up to one ({}),'
                                     'ignoring them.'.format(self._frequencies, self._frequencies.sum()))
                        self._frequencies = None
                    else:
                        if np.any(self._frequencies < 0) and not reoptimise:
                            logger.error('Some of the frequencies given in parameters ({}) are negative,'
                                         'ignoring them.'.format(self._frequencies))
                            self._frequencies = None
                        else:
                            min_freq = \
                                min(1 / self.forest_stats.n_tips,
                                    min(float(params[state]) for state in known_freq_states
                                        if float(params[state]) > 0)) / 2
                            if unknown_freq_states:
                                logger.error('Frequencies for some of the states ({}) are missing from parameters, '
                                             'setting them to {}.'.format(', '.join(unknown_freq_states), min_freq))
                            frequencies = np.maximum(self._frequencies, min_freq)
                            frequencies /= frequencies.sum()
                            self._optimise_frequencies = reoptimise and not self._frequency_smoothing
                except:
                    logger.error('Could not convert the frequencies given in parameters ({}) to float, '
                                 'ignoring them.'.format(self._frequencies))
                    self._frequencies = None
        return params

    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used for logging).

        :return: str representing parameter values
        """
        return '{}' \
               '\tfrequencies\t{}\n' \
               '{}\n'.format(SimpleModel.print_parameters(self),
                             '(optimised)' if self._optimise_frequencies
                             else '(smoothed)' if self._frequency_smoothing else '(fixed)',
                             '\n'.join('\t\t{}:\t{:g}'.format(state, freq)
                                       for (state, freq) in zip(self.get_states(), self._frequencies)))

    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        SimpleModel.freeze(self)
        self._optimise_frequencies = False
        self._frequency_smoothing = False

    def extra_params_fixed(self):
        return self._extra_params_fixed or not self._optimise_frequencies and not self._frequency_smoothing

    def basic_params_fixed(self):
        return not SimpleModel.get_num_params(self)

    def save_parameters(self, filepath, **kwargs):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filepath: path to the file where the parameter values should be written.
        :return: void
        """
        SimpleModel.save_parameters(self, filepath, **kwargs)
        with open(filepath, 'a') as filehandle:
            for state, frequency in zip(self.get_states(), self.get_frequencies()):
                filehandle.write('{}\t{}\n'.format(state, frequency))

    def set_initial_values(self, iteration=0, **kwargs):
        """
        Sets initial values for parameter optimization.

        :param iteration: parameter iteration round (assuming some non-random parameters are to be tried first)
        :return: void, modifies this model by setting its parameters to values used as initial ones
        for parameter optimization
        """
        if 0 == iteration:
            # use this model's values (i.e. equivalent frequencies)
            return
        if 1 == iteration and self._optimise_frequencies:
            self.set_frequencies(self.observed_frequencies)
            return
        bounds = self.get_bounds()
        self.set_params_from_optimised(np.random.uniform(bounds[:, 0], bounds[:, 1]))
