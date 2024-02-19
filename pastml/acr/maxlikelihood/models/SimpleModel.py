import logging
import os
from abc import ABC

import numpy as np
import pandas as pd

from pastml import quote
from pastml.acr.maxlikelihood.models import Model, MODEL


CHANGES_PER_AVG_BRANCH = 'state_changes_per_avg_branch'
SCALING_FACTOR = 'scaling_factor'
SMOOTHING_FACTOR = 'smoothing_factor'


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
