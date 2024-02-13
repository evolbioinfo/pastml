import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pastml import quote

MODEL = 'model'
CHANGES_PER_AVG_BRANCH = 'state_changes_per_avg_branch'
SCALING_FACTOR = 'scaling_factor'
SMOOTHING_FACTOR = 'smoothing_factor'
FREQUENCIES = 'frequencies'


class Model(ABC):

    def __init__(self, forest_stats):
        self._forest_stats = forest_stats

    @abstractmethod
    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used for logging).

        :return: str representing parameter values
        """
        pass

    @abstractmethod
    def get_states(self, node=None, **kwargs):
        """
        Returns model states.

        :param node: (optional) ete3.TreeNode for whose ACR these states are to be considered
        :return: np.array of sorted states
        """
        pass

    @abstractmethod
    def likelihood2marginal_probability_array(self, lh, node, **kwargs):
        """
        Converts a likelihood array to a marginal probability array.

        :param lh: likelihood array to be converted
        :param node: (optional) ete3.TreeNone, whose likelihood is being converted
        :return: np.array of marginal probabilities in the same order as model states
        """
        pass

    @abstractmethod
    def print_basic_parameters(self):
        """
        Constructs a string representing basic parameter values
        (those that are pre-optimised during the first optimization run).

        :return: str representing basic parameter values
        """
        pass

    @abstractmethod
    def save_parameters(self, filepath, **kwargs):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filepath: path to the file where the parameter values should be written.
        :return: the actual filepath used
        """
        pass

    @abstractmethod
    def extra_params_fixed(self):
        """
        Checks if non-basic (those that are not optimized during the pre-optimization round)
        model parameter values are fixed.
        :return:
        """
        pass

    @abstractmethod
    def fix_extra_params(self):
        """
        Fixed the non-basic (those that are not optimized during the pre-optimization round)
        model parameters.

        :return: void
        """
        pass

    @abstractmethod
    def unfix_extra_params(self):
        """
        Frees the non-basic (those that are not optimized during the pre-optimization round)
        model parameters.

        :return: void
        """
        pass

    @property
    @abstractmethod
    def name(self):
        """
        Returns the model name.

        :return: str
        """
        pass

    @abstractmethod
    def get_p_ij_child(self, node):
        """
        Returns the probability matrix of substitutions i->j over the branch of the given node.

        :return: probability matrix
        :rtype: np.array
        """
        pass

    @abstractmethod
    def get_p_ji_child(self, node):
        """
        Returns the probability matrix of substitutions i(parent state)<-j(child state)
        over the branch of the given node (going backward in time).

        :return: probability matrix
        :rtype: np.array
        """
        pass

    @abstractmethod
    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        pass

    @abstractmethod
    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        pass

    @abstractmethod
    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        pass

    @abstractmethod
    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        pass

    @abstractmethod
    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        pass

    @abstractmethod
    def basic_params_fixed(self):
        """
        Checks if the basic (those that are optimized during the pre-optimization round)
        model parameters are all fixed.

        :return: bool
        """
        pass

    @abstractmethod
    def refine_states(self, node, feature, **kwargs):
        """
        Checks if tree node states defined in the feature annotation correspond to this models' state,
        and, if needed, adjusts the annotation.

        :param node: tree's node whose states are to be checked
        :type node: ete.TreeNode
        :param feature: name of the annotation feature containing node states
        :type feature: str
        :return: void, if needed adjusts the node's annotation
        """
        pass

    @abstractmethod
    def get_allowed_states(self, node, feature, **kwargs):
        """
        Returns a vector with zeros for states that are not allowed for this node (based on annotations)
        and ones otherwise.

        :param node: tree's node whose states are to be checked
        :type node: ete.TreeNode
        :param feature: name of the annotation feature containing node states
        :type feature: str
        :return: np.array with a mask for allowed states
        """
        pass

    @abstractmethod
    def get_tau(self, node, **kwargs):
        """
        Returns the smoothing factor associated with this node.

        :param node: ete3.TreeNode
        :return: the value of the smoothing factor
        """
        pass

    @abstractmethod
    def get_frequencies(self, node, **kwargs):
        """
        Returns the equilibrium frequencies associated with this node.

        :param node: ete3.TreeNode
        :return: the np.array with the equilibrium frequencies
        """
        pass

    @property
    def forest_stats(self):
        return self._forest_stats

    @forest_stats.setter
    def forest_stats(self, forest_stats):
        self._forest_stats = forest_stats


class SimpleModel(Model, ABC):

    def refine_states(self, node, feature, **kwargs):
        """
        Checks if tree node states defined in the feature annotation correspond to this models' state,
        and, if needed, adjusts the annotation to contain only the allowed states.

        :param node: tree's node whose states are to be checked
        :type node: ete.TreeNode
        :param feature: name of the annotation feature containing node states
        :type feature: str
        :return: void, if needed adjusts the node's annotation
        """
        if hasattr(node, feature):
            node_states = getattr(node, feature)
            new_node_states = self._state_set & node_states
            node.add_feature(feature, new_node_states)
            if new_node_states != node_states:
                logging.getLogger('pastml') \
                    .warning('Tree node {} contained state(s) {} that are not allowed under the model {}, '
                             'we removed them.'.format(node.name, quote(node_states - new_node_states), self.name))

    def get_allowed_states(self, node, feature, **kwargs):
        """
        Returns a vector with zeros for states that are not allowed for this node (based on annotations)
        and ones otherwise.

        :param node: tree's node whose states are to be checked
        :type node: ete.TreeNode
        :param feature: name of the annotation feature containing node states
        :type feature: str
        :return: np.array with a mask for allowed states
        """
        node_states = getattr(node, feature, set())
        n = len(self._states)
        if not node_states:
            return np.ones(n, dtype=int)
        else:
            allowed_states = np.zeros(n, dtype=int)
            for state in node_states:
                allowed_states[self._state2index[state]] = 1
            return allowed_states

    def __init__(self, states, forest_stats, sf=None, tau=0, optimise_tau=False, parameter_file=None, reoptimise=False,
                 **kwargs):
        Model.__init__(self, forest_stats)
        self._name = None
        self._states = np.sort(states)
        self._state_set = set(self._states)
        self._state2index = dict(zip(self._states, range(len(self._states))))
        self._optimise_tau = optimise_tau
        self._optimise_sf = True
        self._sf = None
        self._tau = None
        self.parse_parameters(parameter_file, reoptimise)
        if self._sf is None:
            self._sf = sf if sf is not None else 1. / forest_stats.avg_dist
        if self._tau is None:
            self._tau = tau if tau else 0

        self.calc_tau_factor(**kwargs)

        self._extra_params_fixed = False

    def calc_tau_factor(self, **kwargs):
        self._tau_factor = \
            self.forest_stats.length / (self.forest_stats.length + self._tau * self.forest_stats.n_nodes) \
                if self._tau else 1

    def __str__(self):
        return \
            'Model {} with parameter values:\n' \
            '{}'.format(self.name,
                        self.print_parameters())

    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used to logging).

        :return: str representing parameter values
        """
        return self.print_basic_parameters()

    def print_basic_parameters(self):
        return '\tscaling factor:\t{:.6f}, i.e. {:.6f} changes per avg branch\t{}\n' \
               '\tsmoothing factor:\t{:.6f}\t{}\n' \
            .format(self.get_sf(), self.forest_stats.avg_dist * self.get_sf(),
                    '(optimised)' if self._optimise_sf else '(fixed)',
                    self.get_tau(), '(optimised)' if self._optimise_tau else '(fixed)')

    def save_parameters(self, filepath, **kwargs):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filepath: path to the file where the parameter values should be written.
        :return: the actual filepath used
        """
        # Not using DataFrames to speed up document writing
        with open(filepath, 'w+') as filehandle:
            filehandle.write('parameter\tvalue\n')
            filehandle.write('{}\t{}\n'.format(MODEL, self.name))
            # filehandle.write('{}\t{}\n'.format(NUM_NODES, self.forest_stats.n_nodes))
            # filehandle.write('{}\t{}\n'.format(NUM_TIPS, self.forest_stats.n_tips))
            filehandle.write('{}\t{}\n'.format(SCALING_FACTOR, self.get_sf()))
            filehandle.write('{}\t{}\n'.format(CHANGES_PER_AVG_BRANCH, self.get_sf() * self.forest_stats.avg_dist))
            filehandle.write('{}\t{}\n'.format(SMOOTHING_FACTOR, self.get_tau()))
        return filepath

    def extra_params_fixed(self):
        return self._extra_params_fixed

    def fix_extra_params(self):
        self._extra_params_fixed = True

    def unfix_extra_params(self):
        self._extra_params_fixed = False

    @property
    def forest_stats(self):
        return self._forest_stats

    @forest_stats.setter
    def forest_stats(self, forest_stats):
        self._forest_stats = forest_stats
        self.calc_tau_factor()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def get_states(self, node=None, **kwargs):
        return self._states

    def set_states(self, states, **kwargs):
        self._states = np.sort(states)
        self._state_set = set(states)

    def get_sf(self, **kwargs):
        return self._sf

    def set_sf(self, sf, **kwargs):
        if self._optimise_sf:
            self._sf = sf
        else:
            raise NotImplementedError('The scaling factor is preset and cannot be changed.')

    def get_tau(self, node=None, **kwargs):
        return self._tau

    def set_tau(self, tau, **kwargs):
        if self._optimise_tau:
            self._tau = tau
            self.calc_tau_factor()
        else:
            raise NotImplementedError('Tau is preset and cannot be changed.')

    def get_Pij_t(self, t, **kwargs):
        """
        Returns a function for calculation of probability matrix of substitutions i->j over time t.

        :return: probability matrix
        :rtype: function
        """
        raise NotImplementedError("Please implement this method in the Model subclass")

    def get_p_ij_child(self, node):
        """
        Returns the probability matrix of substitutions i->j over the branch of the given node.

        :return: probability matrix
        :rtype: np.array
        """
        return self.get_Pij_t(node.dist)

    def get_p_ji_child(self, node):
        """
        Returns the probability matrix of substitutions i(parent state)<-j(child state)
        over the branch of the given node (going backward in time).

        :return: probability matrix
        :rtype: np.array
        """
        return self.get_Pij_t(node.dist)

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update this model parameter values from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        if self._optimise_sf:
            self._sf = ps[0]
        if self._optimise_tau:
            self._tau = ps[1 if self._optimise_sf else 0]

    def get_optimised_parameters(self):
        """
        Converts this model parameters to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        return np.hstack(([self.get_sf()] if self._optimise_sf else [],
                          [self.get_tau()] if self._optimise_tau else []))

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        bounds = []
        if self._optimise_sf:
            bounds += [np.array([0.001 / self.forest_stats.avg_dist,
                                 10. / self.forest_stats.avg_dist])]
        if self._optimise_tau:
            bounds += [np.array([0, self.forest_stats.avg_dist])]
        return np.array(bounds, np.float64)

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        return (1 if self._optimise_sf else 0) + (1 if self._optimise_tau else 0)

    def parse_parameters(self, params, reoptimise=False):
        """
        Update this model's values from the input parameters.
        The input might contain:
        (1) the scaling factor, by which each branch length will be multiplied (optional).
            The key for this parameter is pastml.models.SCALING_FACTOR;
        (2) the smoothing factor, which will be added to each branch length
            before the branches are renormalized to keep the initial tree length (optional).
            The key for this parameter is pastml.models.SMOOTHING_FACTOR;

        :param params: dict {key->value}
            or a path to the file containing a tab-delimited table with the first column containing keys
                and the second (named 'value') containing values.
        :param reoptimise: whether these model parameters should be treated as starting values (True)
            or as fixed values (False)
        :return: dict with parameter values
        """

        logger = logging.getLogger('pastml')
        frequencies, sf, kappa, tau = None, None, None, None
        if params is None:
            return {}
        if not isinstance(params, str) and not isinstance(params, dict):
            raise ValueError('Parameters must be specified either as a dict or as a path to a csv file, not as {}!'
                             .format(type(params)))
        if isinstance(params, str):
            if not os.path.exists(params):
                raise ValueError('The specified parameter file ({}) does not exist.'.format(params))
            try:
                param_dict = pd.read_csv(params, header=0, index_col=0, sep='\t')
                if 'value' not in param_dict.columns:
                    raise ValueError('Could not find the "value" column in the parameter file {}. '
                                     'It should be a tab-delimited file with two columns, '
                                     'the first one containing parameter names, '
                                     'and the second, named "value", containing parameter values.')
                param_dict = param_dict.to_dict()['value']
                params = param_dict
            except:
                raise ValueError('The specified parameter file {} is malformed, '
                                 'should be a tab-delimited file with two columns, '
                                 'the first one containing parameter names, '
                                 'and the second, named "value", containing parameter values.'.format(params))
        params = {str(k.encode('ASCII', 'replace').decode()): v for (k, v) in params.items()}
        if SCALING_FACTOR in params:
            self._sf = params[SCALING_FACTOR]
            try:
                self._sf = np.float64(self._sf)
                if self._sf <= 0:
                    logger.error('Scaling factor cannot be negative, ignoring the value given in parameters ({}).'
                                 .format(sf))
                    self._sf = None
                else:
                    self._optimise_sf = reoptimise
            except:
                logger.error('Scaling factor ({}) given in parameters is not float, ignoring it.'.format(sf))
                self._sf = None
        if SMOOTHING_FACTOR in params:
            self._tau = params[SMOOTHING_FACTOR]
            try:
                self._tau = np.float64(self._tau)
                if self._tau < 0:
                    logger.error(
                        'Smoothing factor cannot be negative, ignoring the value given in parameters ({}).'.format(tau))
                    self._tau = None
            except:
                logger.error('Smoothing factor ({}) given in parameters is not float, ignoring it.'.format(tau))
                self._tau = None
        return params

    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        self._optimise_sf = False
        self._optimise_tau = False

    def transform_t(self, t):
        return (t + self.get_tau()) * self._tau_factor * self.get_sf()

    def basic_params_fixed(self):
        return not self._optimise_tau and not self._optimise_sf

    def likelihood2marginal_probability_array(self, lh, node, **kwargs):
        """
        Converts a likelihood array to a marginal probability array.

        :param lh: likelihood array to be converted
        :param node: (optional) ete3.TreeNone, whose likelihood is being converted
        :return: np.array of marginal probabilities in the same order as model states
        """
        return lh / lh.sum()


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
