from abc import ABC, abstractmethod

import numpy as np

MODEL = 'model'


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

    def set_initial_values(self, iteration=0, **kwargs):
        """
        Sets initial values for parameter optimization.

        :param iteration: parameter iteration round (assuming some non-random parameters are to be tried first)
        :return: void, modifies this model by setting its parameters to values used as initial ones
        for parameter optimization
        """
        if 0 == iteration:
            # use this model's values
            return
        bounds = self.get_bounds()
        self.set_params_from_optimised(np.random.uniform(bounds[:, 0], bounds[:, 1]))
