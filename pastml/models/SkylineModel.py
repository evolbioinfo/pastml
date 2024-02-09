import os
from collections import defaultdict

import numpy as np

from pastml import MODEL_ID
from pastml.models import Model
from pastml.tree import DATE


class SkylineModel(Model):
    """
    This class represents a metamodel that operates on a tree,
    for which the model can change with time in a piecewise manner.

    The skyline model assumes that the tree nodes are pre-annotated with model ids, according to the time intervals
    (where the model with id 0 is the oldest one, e.g. at the root).

    If the original tree has a branch from a parent node P at time 1988 to a child node C at time 2000,
    while the skyline intervals are ]-inf, 1991] and ]1991, inf[, then this branch will be split in three parts
    during state transition probability matrix calculation:
    (1) a branch from C to (an imaginary timepoint) PC_up, of length 1991 - 1988 = 3 years, where PC_up is at time 1991
    (2) a branch from (an imaginary timepoint) PC_down, of length 2000 - 1991 = 9 years, where PC_down is at time 1991
    (2) a zero-branch PC_up-PC_down, where the state mapping between the two models happens.

    In the example above, since the skyline has two intervals,
    each tree node will be annotated either with model id 0 (for the top interval ]-inf, 1991])
    or model id 1 (]1991, inf[)).
    In the example above node P will be annotated with model id 0,
    while C will be annotated with model id 1.

    Hence, the skyline models applies model 0
    when calculating the probabilities of state changes along the branch C-PC_up,
    model 1 for the branch PC_down-C,
    and converts the states between the two models along the branch PC_up-PC_down
    (according to state mapping between models 0 and 1).
    """

    def __init__(self, models, dates, skyline_mapping, forest_stats, **kwargs):
        """
        The skyline model is initiated with an array of models that will be applied to each skyline interval.

        :param models: an array of models that will be applied to each skyline interval
        :type models: np.array(pastml.models.Model)
        :param dates: an array of dates corresponding to skyline interval changes
        :type dates: np.array(float)
        """
        if len(models) != len(dates) + 1:
            raise ValueError('Skyline dates should represent moments of skyline interval change, '
                             'hence they should be all different and as many as the number of model minus one')
        self._models = np.array(models)
        self._dates = np.array(sorted(set(dates)))
        if skyline_mapping is not None:
            self._skyline_mapping = skyline_mapping
        else:
            id_mapping = np.eye(len(self._models[0].get_states()))
            self._skyline_mapping = defaultdict(lambda: id_mapping)
        self._all_states = np.array(sorted(set.union(*(m._state_set for m in self._models))))
        state2index = {s: i for (i, s) in enumerate(self._all_states)}
        self._model2state_mask = [[state2index[s] for s in m.get_states()] for m in self._models]
        Model.__init__(self, forest_stats)

    def get_interval(self, model_id=0):
        """Returns the time interval to which the model_id's model applies.
        The interval is closed on the start date and open on the end date: [start, end[

        :return: tuple(interval start date, interval end date)
        """
        return (-np.inf if model_id == 0 else self._dates[model_id - 1]), \
            (self._dates[model_id] if model_id < (len(self._models) - 1) else np.inf)

    def get_num_params(self):
        """
        Returns the number of optimized parameters for this model.

        :return: the number of optimized parameters
        """
        return sum(_.get_num_params() for _ in self._models)

    def set_params_from_optimised(self, ps, **kwargs):
        """
        Update the parameter values of underlying models from a vector representing parameters
        for the likelihood optimization algorithm.

        :param ps: np.array containing parameters of the likelihood optimization algorithm
        :param kwargs: dict of eventual other arguments
        :return: void, update this model
        """
        start = 0
        for model in self._models:
            n = model.get_num_params()
            model.set_params_from_optimised(ps[start: n], **kwargs)
            start += n

    def get_optimised_parameters(self):
        """
        Converts the parameters of underlying models to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        return np.hstack((model.get_optimised_parameters() for model in self._models))

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        return np.hstack((model.get_bounds() for model in self._models))

    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used for logging).

        :return: str representing parameter values
        """
        return '\n'.join('----skyline interval {}:\n{}'
                         .format(i, model.print_parameters()) for (i, model) in enumerate(self._models))

    def freeze(self):
        """
        Prohibit parameter optimization by setting all optimization flags to False.

        :return: void
        """
        for model in self._models:
            model.freeze()

    def extra_params_fixed(self):
        return all((model.extra_params_fixed() for model in self._models))

    def basic_params_fixed(self):
        return all((model.basic_params_fixed() for model in self._models))

    def save_parameters(self, filehandle):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filehandle: filehandle for the file where the parameter values should be written.
        :return: void
        """
        filename, file_extension = os.path.splitext(filehandle)
        for i, model in enumerate(self._models):
            model.save_parameters('{}_interval{}.{}'.format(filename, i, file_extension))

    def __str__(self):
        return \
            'Skyline Model:\n' \
            '\n'.join('----skyline interval {} ({}-{}):\n{}'.format(i, *self.get_interval(i), model)
                      for (i, model) in enumerate(self._models))

    def print_basic_parameters(self):
        return '\n'.join('----skyline interval {} ({}-{}):\n{}'
                         .format(i, *self.get_interval(i), model.print_basic_parameters())
                         for (i, model) in enumerate(self._models))

    def fix_extra_params(self):
        for model in self._models:
            model.fix_extra_params()

    def unfix_extra_params(self):
        for model in self._models:
            model.unfix_extra_params()

    @property
    def name(self):
        return 'Skyline Model: {}'.format('>'.join('{}({}-{})'.format(_.name, *self.get_interval(i))
                                                   for (i, _) in enumerate(self._models)))

    def get_p_ij_child(self, node):
        """
        Returns the probability matrix of substitutions i(parent state)->j(child state)
        over the branch of the given node.

        :return: probability matrix
        :rtype: np.array
        """
        model_down = getattr(node, MODEL_ID, 0)
        child_date = getattr(node, DATE)
        parent_date = child_date - node.dist
        model_up = getattr(node.up, MODEL_ID, 0) if not node.is_root() else self.get_model_id_at_time(parent_date)
        if model_up == model_down:
            return self._models[model_up].get_Pij_t(node.dist)

        p_ij = None
        while model_up != model_down:
            model_change_date = self.get_interval(model_up)[1]
            cur_p_ij = self._models[model_up].get_Pij_t(model_change_date - parent_date)
            p_ij = cur_p_ij if p_ij is None else p_ij.dot(cur_p_ij)
            p_ij = p_ij.dot(self._skyline_mapping[(model_up, model_up + 1)])

            parent_date = model_change_date
            model_up += 1
        return p_ij.dot(self._models[model_up].get_Pij_t(child_date - parent_date))

    def get_p_ji_child(self, node):
        """
        Returns the probability matrix of substitutions i(parent state)<-j(child state)
        over the branch of the given node (going backward in time).

        :return: probability matrix
        :rtype: np.array
        """
        model_down = getattr(node, MODEL_ID, 0)
        child_date = getattr(node, DATE)
        parent_date = child_date - node.dist
        model_up = getattr(node.up, MODEL_ID, 0) if not node.is_root() else self.get_model_id_at_time(parent_date)
        if model_up == model_down:
            return np.transpose(self._models[model_up].get_Pij_t(node.dist))

        p_ji = None
        while model_down != model_up:
            model_change_date = self.get_interval(model_down)[0]
            cur_p_ji = np.transpose(self._models[model_down].get_Pij_t(child_date - model_change_date))
            p_ji = cur_p_ji if p_ji is None else p_ji.dot(cur_p_ji)
            p_ji = p_ji.dot(self._skyline_mapping[(model_down, model_down - 1)])

            child_date = model_change_date
            model_down -= 1
        return p_ji.dot(np.transpose(self._models[model_down].get_Pij_t(child_date - parent_date)))

    def get_states(self, node=None, **kwargs):
        if node:
            return self._models[getattr(node, MODEL_ID, 0)].get_states(node)
        else:
            return self._all_states

    def get_model_id_at_time(self, date, start_id=0, stop_id=np.inf):
        stop_id = min(len(self._models), stop_id)
        cur_id = int((start_id + stop_id)/ 2)
        start, stop = self.get_interval(cur_id)
        if start <= date < stop:
            return cur_id
        elif start > date:
            return self.get_model_id_at_time(date, start_id, cur_id)
        else:
            return self.get_model_id_at_time(date, cur_id, stop_id)

    def refine_states(self, node, feature, **kwargs):
        """
        Checks if tree node states defined in the feature annotation
        correspond to the corresponding interval's models' state,
        and, if needed, adjusts the annotation to contain only the allowed states.

        :param node: tree's node whose states are to be checked
        :type node: ete.TreeNode
        :param feature: name of the annotation feature containing node states
        :type feature: str
        :return: void, if needed adjusts the node's annotation
        """
        self._models[getattr(node, MODEL_ID, 0)].refine_states(node, feature, **kwargs)

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
        return self._models[getattr(node, MODEL_ID, 0)].get_allowed_states(node, feature, **kwargs)

    def get_tau(self, node, **kwargs):
        return self._models[getattr(node, MODEL_ID, 0)].get_tau()

    def get_frequencies(self, node=None, **kwargs):
        return self._models[getattr(node, MODEL_ID, 0)].get_frequencies()

    def likelihood2marginal_probability_array(self, lh, node, **kwargs):
        """
        Converts a likelihood array to a marginal probability array.
        The marginal probability array is of the size of all states (across all models).

        :param lh: likelihood array to be converted
        :param node: tree's node, whose likelihood is being converted
        :return: np.array of marginal probabilities in the same order as model states
        """
        res = np.zeros(len(self._all_states), dtype=np.float64)
        res[self._model2state_mask[getattr(node, MODEL_ID, 0)]] = lh / lh.sum()
        return res
