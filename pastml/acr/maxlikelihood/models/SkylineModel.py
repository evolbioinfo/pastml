import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from pastml import col_name2cat
from pastml.acr.maxlikelihood.models import Model
from pastml.tree import DATE

MODEL_ID = 'MODEL_ID'


def annotate_skyline(forest, skyline, character, skyline_mapping=None):
    """
    Verifies that the skyline points are well-defined, annotates the tree with skyline model ids,
    and parses the state mapping table if the states changes between skyline intervals.

    :param forest: list of trees to be annotated (with DATE annotations on their nodes)
    :param skyline: dates of model/state changes
    :param character: character of interest
    :param skyline_mapping: an optional file, containing skyline state mapping
    :return: skyline (sorted array of model/state change dates),
        a tuple (skyline mapping table, a list of lists of states (for each time interval)).
        The latter is None if the mapping was not specified.
    """
    min_date, max_date = min(getattr(tree, DATE) - tree.dist for tree in forest), \
        max(max(getattr(_, DATE) for _ in tree) for tree in forest)
    skyline = sorted(skyline)
    if skyline[0] >= max_date or skyline[-1] <= min_date:
        raise ValueError('You have specified dates of model/state changes ({}), '
                         'however they are outside of your tree date interval: {}-{}.'
                         .format(', '.join(str(_) for _ in skyline), min_date, max_date))

    logging.getLogger('pastml').debug('The tree(s) cover the period between {} and {}, '
                                      'the models/states change at the following dates: {}.'
                                      .format(min_date, max_date, ', '.join(str(_) for _ in skyline)))

    if skyline_mapping:
        skyline_mapping = parse_skyline_mapping(character, skyline, skyline_mapping)
    else:
        skyline_mapping = None

    def annotate_node_skyline(node, i):
        # Skyline contains the times when the model changes,
        # hence the root nodes (model 0) should be before the first skyline point,
        # and the most recent leaf should be after the last
        for j in range(i, len(skyline) + 1):
            if j == len(skyline):
                break
            n_date = getattr(node, DATE)
            if skyline[j] > n_date and (j == 0 or n_date >= skyline[j - 1]):
                break
        node.add_feature(MODEL_ID, j)
        for child in node.children:
            annotate_node_skyline(child, j)

    for tree in forest:
        annotate_node_skyline(tree, 0)

    return skyline, skyline_mapping


def parse_skyline_mapping(character, skyline, skyline_mapping):
    df = pd.read_csv(skyline_mapping, sep='\t', index_col=None)

    def convert_col(c):
        cc = col_name2cat(c)
        if cc == character:
            return cc
        try:
            return float(c)
        except:
            return c

    df.columns = [convert_col(_) for _ in df.columns]
    try:
        df = df[[character] + skyline]
    except KeyError:
        raise ValueError('Skyline mapping is specified in {} but instead of containing columns {} it contains {}'
                         .format(skyline_mapping, ', '.join([character] + list(reversed([str(_) for _ in skyline]))),
                                 df.columns))

    def get_states(source_state, source_col, target_col):
        target_states = {str(_) for _ in df.loc[df[source_col] == source_state, target_col].unique()
                         if not pd.isna(_)}
        if not target_states:
            raise ValueError('Could not find the states corresponding to {} of {} in {}'
                             .format(source_state, source_col, target_col))
        return target_states

    mapping = {}
    prev_states = {}
    prev_col = None
    all_states = []
    # character corresponds to column names at the most recent time interval
    for i, col in enumerate(skyline + [character], start=0):
        states = {str(_) for _ in df[col].unique() if not pd.isna(_)}
        all_states.append(np.array(sorted(states)))
        if i > 0:
            mapping[(i - 1, i)] = {prev_state: get_states(prev_state, prev_col, col) for prev_state in prev_states}
            mapping[(i, i - 1)] = {state: get_states(state, col, prev_col) for state in states}
        prev_col, prev_states = col, states

    skyline_mapping = {}
    for (i, j), state2states in mapping.items():
        mapping_ij = np.zeros(shape=(len(all_states[i]), len(all_states[j])), dtype=np.float64)
        skyline_mapping[(i, j)] = mapping_ij
        for (from_i, from_state) in enumerate(all_states[i]):
            to_states = state2states[from_state]
            for (to_j, to_state) in enumerate(all_states[j]):
                if to_state in to_states:
                    mapping_ij[from_i, to_j] = 1
    return skyline_mapping, all_states


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
            model.set_params_from_optimised(ps[start: start + n], **kwargs)
            start += n

    def get_optimised_parameters(self):
        """
        Converts the parameters of underlying models to a vector representing parameters
        for the likelihood optimization algorithm.

        :return: np.array containing parameters of the likelihood optimization algorithm
        """
        return np.hstack([model.get_optimised_parameters() for model in self._models])

    def get_bounds(self):
        """
        Get bounds for parameters for likelihood optimization algorithm.

        :return: np.array containing lower and upper (potentially infinite) bounds for each parameter
        """
        bounds = []
        for model in self._models:
            bounds.extend(model.get_bounds())
        return np.array(bounds)

    def print_parameters(self):
        """
        Constructs a string representing parameter values (to be used for logging).

        :return: str representing parameter values
        """
        return ''.join('\t----skyline interval {} ({}):\n{}'
                       .format(i, self.get_interval_string(i), model.print_parameters())
                       for (i, model) in enumerate(self._models))

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

    def save_parameters(self, filepath, **kwargs):
        """
        Writes this model parameter values to the parameter file (in the same format as the input parameter file).

        :param filepath: path to the file where the parameter values should be written.
        :return: the actual filepaths used
        """
        filename, file_extension = os.path.splitext(filepath)
        filepaths = []
        for i, model in enumerate(self._models):
            fp = '{}_submodel_{}_{}{}'.format(filename, model.name, self.get_interval_string(i), file_extension)
            filepaths.append(fp)
            model.save_parameters(fp, **kwargs)
        return filepaths

    def get_interval_string(self, i):
        """
        Returns a string representation of the time interval corresponding to the i-th model.
        :param i: model number
        :return: a string representation of the time interval corresponding to the i-th model
        """
        return '{:g}-{:g}'.format(*self.get_interval(i))

    def __str__(self):
        return 'Skyline Model:\n' + \
            '\n'.join('\t----skyline interval {} ({}):\n{}'.format(i, self.get_interval_string(i), model)
                      for (i, model) in enumerate(self._models))

    def print_basic_parameters(self):
        return ''.join('\t----skyline interval {} ({}):\n{}'
                       .format(i, self.get_interval_string(i), model.print_basic_parameters())
                       for (i, model) in enumerate(self._models))

    def fix_extra_params(self):
        for model in self._models:
            model.fix_extra_params()

    def unfix_extra_params(self):
        for model in self._models:
            model.unfix_extra_params()

    @property
    def name(self):
        result = 'Skyline'
        for i, m in enumerate(self._models):
            result += '-{}'.format(m.name)
            if i < len(self._models) - 1:
                date = self.get_interval(i)[1]
                result += '-{:g}'.format(date)
        return result

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
            return self._models[model_up].get_p_ij_child(node)

        p_ij = None
        while model_up != model_down:
            model_change_date = self.get_interval(model_up)[1]
            cur_p_ij = self._models[model_up].get_Pij_t(model_change_date - parent_date)
            p_ij = cur_p_ij if p_ij is None else p_ij.dot(cur_p_ij)
            mapping = self.get_mapping(model_up, model_up + 1)
            p_ij = p_ij.dot(mapping)

            parent_date = model_change_date
            model_up += 1
        return p_ij.dot(self._models[model_up].get_Pij_t(child_date - parent_date))

    def get_mapping(self, source_model, target_model):
        """
        Returns the state mapping matrix from the states of the source model to the states of the target model.
        If a state A (with frequency f_A) of the source model corresponds to
        states A1 or A2 (with frequencies f_A1 and f_A2) of the target model,
        then the mapping matrix at row A contains all zeros, except for columns A1 and A2,
        # whose values are (f_A1 + f_A)/(2f_A) and (f_A2 + f_A)/(2f_A).
        # This allows to have a one-to-one mapping
        # if the source and target models and their parameters and states are identical,
        # while ensuring the time reversibility (\pi_i^(src) P_ij(b) = \pi_j^(tgt) P_ji(b),
        # where b is a branch that starts in the src_model time interval and ends in the target model time interval:
        # \pi_i^(src) P_ij(b) = \pi_i^(src) \sum_kl P_ik^(src) M_kl^(src->tgt) P_lj^(tgt) =
        # = \sum_kl P_ki^(src) \pi_k^(src) M_kl^(src->tgt) P_jl^(tgt) \pi_j^(tgt) / \pi_l^(tgt) =
        # = \sum_kl \pi_j^(tgt) P_jl^(tgt) M_lk^(tgt->src) P_ik^(src) M_kl^(src->tgt) / M_lk^(tgt->src) * \pi_k^(src) / \pi_l^(tgt) =
        # = ( if M_ab(x->y) = (\pi_a^(x) + \pi_b^(y))/ (2\pi_a^(x))
        #     and only pairs a,b (or k,l) where there is a mapping between a and b (or k and l) are considered) =
        # = \pi_j^(tgt) \sum_kl P_jl^(tgt) M_lk^(tgt->src) P_ik^(src) = \pi_j^(tgt) P_ji(b).
        whose values are f_A1/(f_A1 + f_A2) and f_A2/(f_A1 + f_A2).
        # TODO: the model should impose that f_A1 + f_A2 = f_A

        :param source_model: source model
        :param target_model: target model
        :return: the mapping matrix
        """
        # result = np.array(self._skyline_mapping[(source_model, target_model)], dtype=np.float64)
        # for i, freq_source in enumerate(self._models[source_model].get_frequencies()):
        #     result[i, :] *= (freq_source + self._models[target_model].get_frequencies()) / freq_source / 2
        # return result
        result = np.array(self._skyline_mapping[(source_model, target_model)], dtype=np.float64)
        for i, freq_source in enumerate(self._models[source_model].get_frequencies()):
            result[i, :] *= self._models[target_model].get_frequencies()
            # result[i, :] /= max(result[i, :].sum(), freq_source)
            result[i, :] /= result[i, :].sum()
        return result

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
            return self._models[model_up].get_p_ji_child(node)

        p_ji = None
        while model_down != model_up:
            model_change_date = self.get_interval(model_down)[0]
            cur_p_ji = self._models[model_down].get_Pij_t(child_date - model_change_date)
            p_ji = cur_p_ji if p_ji is None else p_ji.dot(cur_p_ji)
            # remap the frequencies in a way that if a state A (with frequency f_A)
            # is mapped to A1 or A2 (with frequencies f_A1 and f_A2),
            # then its mapping becomes f_A1/f_A for A1 and f_A2/f_A for A2.
            mapping = self.get_mapping(model_down, model_down - 1)
            p_ji = p_ji.dot(mapping)

            child_date = model_change_date
            model_down -= 1
        return p_ji.dot(self._models[model_down].get_Pij_t(child_date - parent_date))

    def get_states(self, node=None, **kwargs):
        if node:
            return self._models[getattr(node, MODEL_ID, 0)].get_states(node)
        else:
            return self._all_states

    def get_model_id_at_time(self, date, start_id=0, stop_id=np.inf):
        stop_id = min(len(self._models), stop_id)
        cur_id = int((start_id + stop_id) / 2)
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

    def set_initial_values(self, iteration=0, **kwargs):
        """
        Sets initial values for parameter optimization.

        :param iteration: parameter iteration round (assuming some non-random parameters are to be tried first)
        :return: void, modifies this model by setting its parameters to values used as initial ones
        for parameter optimization
        """
        for model in self._models:
            model.set_initial_values(iteration, **kwargs)
