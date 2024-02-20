import logging

import numpy as np

from pastml.acr.maxlikelihood.models.SimpleModel import SimpleModel

FREQUENCIES = 'frequencies'


class FrequencyBlockError(Exception):
    """ Error related to inappropriate model specification wrt frequency block constraints. """

    def __init__(self, *args, **kwargs):
        Exception.__init__(self, *args, **kwargs)


def islist(a):
    return isinstance(a, (list, np.ndarray))


class ModelWithFrequencies(SimpleModel):

    def __init__(self, states, forest_stats=None, sf=None, frequencies=None, tau=0,
                 optimise_tau=False, frequency_smoothing=False, parameter_file=None, reoptimise=False,
                 **kwargs):
        self._frequencies = None
        self._optimise_frequencies = not frequency_smoothing
        self._frequency_smoothing = frequency_smoothing
        SimpleModel.__init__(self, states, forest_stats=forest_stats,
                             sf=sf, tau=tau, optimise_tau=optimise_tau, reoptimise=reoptimise,
                             parameter_file=parameter_file, **kwargs)
        if frequencies is not None:
            self._frequencies = frequencies
        self._observed_frequencies = None

        # frequency block is an optional way to constraint some frequencies.
        # It is a list of sorted sublists, such that the union of the sublists corresponds to [0, ..., n-1],
        # where n is the number of model states, the sublists do not intersect, and each sublist element is unique.
        # Having a sublist [i_1, ..., i_m] (1 <= m < n) means that the corresponding frequencies are optimized together
        # as a block, with individual lower-bound constraints (i_j >= b_j, where b_j > 0)
        # and a constraint of their sum i_1 + ... + i_m = B (where B <= 1).
        # Having more than one sublist hence reduces the number of free parameters.
        # Moreover, having a one-element sublist [i] means that the i-th frequency is fixed.
        n = len(self.get_states())
        self._frequency_blocks = np.array([np.array(range(n))])
        self._frequency_block_sums = np.array([1])
        self._frequency_block_min_values = np.array([np.zeros(n, dtype=np.float64)])

    def get_frequencies(self, node=None, **kwargs):
        if self._frequencies is None:
            # If no frequencies are given, let's initialize them by setting minimal values
            # + equally distributing anything left to top up to the frequency block sum
            self._frequencies = np.zeros(len(self.get_states()), dtype=np.float64)
            for block, block_sum, block_min_vs \
                    in zip(self._frequency_blocks, self._frequency_block_sums, self._frequency_block_min_values):
                self._frequencies[block] = block_min_vs + (block_sum - sum(block_min_vs)) / len(block)
        return self._frequencies

    def get_frequency_blocks(self):
        return self._frequency_blocks

    def set_frequency_blocks(self, frequency_blocks):
        """
        Frequency blocks is an optional way to constraint some frequencies.
        It is a list of sorted sublists, such that the union of the sublists corresponds to [0, ..., n-1],
        where n is the number of model states, the sublists do not intersect, and each sublist element is unique.
        Having a sublist [i_1, ..., i_m] (1 <= m < n) means that the corresponding frequencies are optimized together
        as a block, with individual lower-bound constraints (i_j >= b_j, where 1 >= b_j > 0,
            which can be set via set_frequency_block_min_values method)
        and a constraint of their sum i_1 + ... + i_m = B (where 0 < B <= 1,
            which can be set via set_frequency_block_sums method).
        Having more than one sublist hence reduces the number of free parameters.
        Moreover, having a one-element sublist [i] means that the i-th frequency is fixed
            (and is equal to its block sum).

        :param frequency_blocks: frequency blocks to set
        :return: void
        """
        if not islist(frequency_blocks) \
                or not all(islist(_) for _ in frequency_blocks) \
                or not all(all(isinstance(v, (int, np.int64)) for v in _) for _ in frequency_blocks):
            raise FrequencyBlockError('frequency_blocks must be a list of lists, '
                                      'where each sublist contains indices of the frequencies '
                                      'that belong to the corresponding block.')
        if sorted([_ for block in frequency_blocks for _ in block]) != list(range(len(self._states))):
            raise FrequencyBlockError('The union of frequency blocks must include every state index exactly once')
        self._frequency_blocks = frequency_blocks

    def check_frequency_block_consistency(self):
        """
        Frequency blocks is an optional way to constraint some frequencies.
        It is a list of sorted sublists, such that the union of the sublists corresponds to [0, ..., n-1],
        where n is the number of model states, the sublists do not intersect, and each sublist element is unique.
        Having a sublist [i_1, ..., i_m] (1 <= m < n) means that the corresponding frequencies are optimized together
        as a block, with individual lower-bound constraints (i_j >= b_j, where 1 >= b_j > 0,
            which can be set via set_frequency_block_min_values method)
        and a constraint of their sum i_1 + ... + i_m = B (where 0 < B <= 1,
            which can be set via set_frequency_block_sums method).
        Having more than one sublist hence reduces the number of free parameters.
        Moreover, having a one-element sublist [i] means that the i-th frequency is fixed
            (and is equal to its block sum).

        This method checks is the model's frequencies are consistent with its frequency block constraints.

        :return: True if the model is consistent otherwise raises a FrequencyBlockError
        """
        for block, block_sum, block_min_vs \
                in zip(self._frequency_blocks, self._frequency_block_sums, self._frequency_block_min_values):
            subfreqs = self.get_frequencies()[block]
            if np.round(subfreqs.sum() - block_sum, 3):
                raise FrequencyBlockError('Model frequencies {} '
                                          'do not satisfy the constraint on their sum {:g} '
                                          '(specified via frequency blocks)'
                                          .format(', '.join('{}={:g}'.format(*_)
                                                            for _ in zip(self.get_states()[block], subfreqs)),
                                                  block_sum))
            if np.any(np.round(subfreqs - block_min_vs, 3) < 0):
                raise FrequencyBlockError('Model frequencies {} '
                                          'do not satisfy the constraint on their min values {} '
                                          '(specified via frequency blocks)'
                                          .format(', '.join('{}={:g}'.format(*_)
                                                            for _ in zip(self.get_states()[block], subfreqs)),
                                                  ', '.join('{}>={:g}'.format(*_)
                                                            for _ in zip(self.get_states()[block], block_min_vs))))

    def set_frequency_block_sums(self, frequency_block_sums):
        """
        Frequency blocks is an optional way to constraint some frequencies.
        It is a list of sorted sublists, such that the union of the sublists corresponds to [0, ..., n-1],
        where n is the number of model states, the sublists do not intersect, and each sublist element is unique.
        Having a sublist [i_1, ..., i_m] (1 <= m < n) means that the corresponding frequencies are optimized together
        as a block, with a constraint of their sum i_1 + ... + i_m = B (where 0 < B <= 1).
        This method sets these sum constraints.

        :param frequency_block_sums: frequency block sum constraints to set
        :return: void
        """
        if not islist(frequency_block_sums) \
                or not all(isinstance(_, (float, np.float64)) for _ in frequency_block_sums):
            raise FrequencyBlockError('frequency_block_sums must be a list of sum constraints '
                                      'of the frequencies in the corresponding frequency block.')
        if np.round(sum(frequency_block_sums), 3) != 1:
            raise FrequencyBlockError('The sum of frequency block sums must be 1')
        if len(frequency_block_sums) != len(self._frequency_blocks):
            raise FrequencyBlockError('frequency_block_sums must be a list of sum constraints '
                                      'of the frequencies in the corresponding frequency block. '
                                      'It should hence contain as many elements as there are frequency blocks '
                                      '(which is not the case here).')
        if next((v for v in frequency_block_sums if 0 >= v or 1 < v), False):
            raise FrequencyBlockError('All frequency_block_min_sums must be between 0 and 1')
        self._frequency_block_sums = frequency_block_sums

    def set_frequency_block_min_values(self, frequency_block_min_values):
        """
        Frequency blocks is an optional way to constraint some frequencies.
        It is a list of sorted sublists, such that the union of the sublists corresponds to [0, ..., n-1],
        where n is the number of model states, the sublists do not intersect, and each sublist element is unique.
        Having a sublist [i_1, ..., i_m] (1 <= m < n) means that the corresponding frequencies are optimized together
        as a block, with individual lower-bound constraints (i_j >= b_j, where 1 >= b_j > 0).
        This method sets these lower-bound constraints.

        :param frequency_block_min_values: frequency block lower bound constraints.
        :return: void
        """
        if not islist(frequency_block_min_values) \
                or not all(islist(_) for _ in frequency_block_min_values) \
                or not all(all(isinstance(v, (float, int, np.float64, np.int64)) for v in _)
                           for _ in frequency_block_min_values):
            raise FrequencyBlockError('frequency_block_min_values must be a list of lists, '
                                      'where each sublist contains lower bounds on the values '
                                      'of the frequencies in the corresponding frequency block')
        if sum(len(_) for _ in frequency_block_min_values) != len(self._states):
            raise FrequencyBlockError('frequency_block_min_values must be a list of lists, '
                                      'where each sublist contains lower bounds on the values '
                                      'of the frequencies in the corresponding frequency block. '
                                      'It should hence contain in total as many elements as there are model states '
                                      '(which is not the case here).')

        if next((next((v for v in _ if 0 > v or 1 < v), False) for _ in frequency_block_min_values), False):
            raise FrequencyBlockError('All frequency_block_min_values must be between 0 and 1')
        self._frequency_block_min_values = frequency_block_min_values

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
        if self._optimise_frequencies:
            return n_basic_params + len(self._frequencies) - len(self._frequency_blocks)
        return n_basic_params + (1 if self._frequency_smoothing else 0)

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
            n_params = SimpleModel.get_num_params(self)
            if self._optimise_frequencies:
                for block, block_sum, block_min_vs \
                        in zip(self._frequency_blocks, self._frequency_block_sums, self._frequency_block_min_values):
                    n_block = len(block)
                    if n_block > 1:
                        # Frequencies in a block (f_1, ..., f_m) have lower limits (l_i): f_i >= b_i
                        # and a constraint B on their sum: \sum_i f_i = B
                        # While what we've just optimized are added values (v_i),
                        # proportional to the last frequency's added value (v_m = 1) ,
                        # such that f_i = b_i + v_i * (B - \sum_i b_i) / \sum_i v_i.
                        # Hence (since v_m = 1) \sum_i v_i = (B - \sum_i b_i) / (f_m - b_m),
                        # v_i = (f_i - b_i) / (B - \sum_i b_i) * \sum_i v_i.
                        B_minus_sum_b_i = block_sum - sum(block_min_vs)
                        vs = np.hstack((ps[n_params: n_params + (n_block - 1)], [1.]))
                        self._frequencies[block] = block_min_vs + vs * B_minus_sum_b_i / vs.sum()
                    else:
                        self._frequencies[block[0]] = block_sum
            elif self._frequency_smoothing:
                self._frequencies *= self.forest_stats.n_tips
                self._frequencies += ps[n_params]
                self._frequencies /= self._frequencies.sum()

    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """

        simple_params = SimpleModel.get_optimised_parameters(self)
        if not self.extra_params_fixed():
            freq_params = []
            if self._optimise_frequencies:
                for block, block_sum, block_min_vs \
                        in zip(self._frequency_blocks, self._frequency_block_sums, self._frequency_block_min_values):
                    if len(block) > 1:
                        # Frequencies in a block (f_1, ..., f_m) have lower limits (l_i): f_i >= b_i
                        # and a constraint B on their sum: \sum_i f_i = B
                        # While what we've just optimized are added values (v_i),
                        # proportional to the last frequency's added value (v_m = 1) ,
                        # such that f_i = b_i + v_i * (B - \sum_i b_i) / \sum_i v_i.
                        # Hence (since v_m = 1) \sum_i v_i = (B - \sum_i b_i) / (f_m - b_m),
                        # v_i = (f_i - b_i) / (B - \sum_i b_i) * \sum_i v_i.
                        subfreqs = self._frequencies[block]
                        B_minus_sum_b_i = block_sum - sum(block_min_vs)
                        sum_v_i = B_minus_sum_b_i / (subfreqs[-1] - block_min_vs[-1])
                        freq_params.extend((subfreqs[:-1] - block_min_vs[:-1]) / B_minus_sum_b_i * sum_v_i)
            else:
                freq_params = [0] if self._frequency_smoothing else []
            if freq_params:
                return np.hstack((simple_params, freq_params))
        return simple_params

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        simple_bounds = SimpleModel.get_bounds(self)
        if not self.extra_params_fixed():
            extras = []
            if self._optimise_frequencies:
                for block in self._frequency_blocks:
                    n_block = len(block)
                    if n_block > 1:
                        extras += [np.array([1e-6, 10e6], np.float64)] * (n_block - 1)
            if self._frequency_smoothing:
                extras.append(np.array([0, self.forest_stats.n_nodes]))
            if extras:
                return np.array((*simple_bounds, *extras))
        return simple_bounds

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
                                              and len(self._frequency_blocks) < len(self._frequencies)
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
        return self._extra_params_fixed \
            or not self._optimise_frequencies and not self._frequency_smoothing

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
        if 1 == iteration and self._optimise_frequencies and len(self._frequency_blocks) == 1 \
                and all(self._frequency_block_min_values == 0):
            self.set_frequencies(self.observed_frequencies)
            return
        bounds = self.get_bounds()
        self.set_params_from_optimised(np.random.uniform(bounds[:, 0], bounds[:, 1]))
