import logging
import os
import warnings

import numpy as np

from pastml import STATES, METHOD, CHARACTER, PASTML_VERSION
from pastml.annotation import annotate_forest, ForestStats
from pastml.file import get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file, get_pastml_work_dir, get_ancestral_state_file
from pastml.logger import set_up_pastml_logger
from pastml.ml import MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, ml_acr, \
    ML_METHODS, MAP, JOINT, MARGINAL_ML_METHODS
from pastml.models import MODEL, SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.models.CustomRatesModel import CustomRatesModel, CUSTOM_RATES
from pastml.models.EFTModel import EFTModel, EFT
from pastml.models.F81Model import F81Model, F81
from pastml.models.HKYModel import HKYModel, HKY, HKY_STATES
from pastml.models.JCModel import JCModel, JC
from pastml.models.JTTModel import JTTModel, JTT, JTT_STATES
from pastml.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS
from pastml.tree import read_forest, save_tree, refine_states

model2class = {F81: F81Model, JC: JCModel, CUSTOM_RATES: CustomRatesModel, HKY: HKYModel, JTT: JTTModel, EFT: EFTModel}

warnings.filterwarnings("ignore", append=True)


def serialize_acr(args):
    acr_result, work_dir = args
    out_param_file = \
        os.path.join(work_dir,
                     get_pastml_parameter_file(method=acr_result[METHOD],
                                               model=acr_result[MODEL].name if MODEL in acr_result else None,
                                               column=acr_result[CHARACTER]))

    # Not using DataFrames to speed up document writing
    with open(out_param_file, 'w+') as f:
        f.write('parameter\tvalue\n')
        f.write('pastml_version\t{}\n'.format(PASTML_VERSION))
        for name in sorted(acr_result.keys()):
            if name not in [STATES, MARGINAL_PROBABILITIES, METHOD, MODEL]:
                f.write('{}\t{}\n'.format(name, acr_result[name]))
        f.write('{}\t{}\n'.format(METHOD, acr_result[METHOD]))
        if is_ml(acr_result[METHOD]):
            acr_result[MODEL].save_parameters(f)
    logging.getLogger('pastml').debug('Serialized ACR parameters and statistics for {} to {}.'
                                      .format(acr_result[CHARACTER], out_param_file))

    if is_marginal(acr_result[METHOD]):
        out_mp_file = \
            os.path.join(work_dir,
                         get_pastml_marginal_prob_file(method=acr_result[METHOD], model=acr_result[MODEL].name,
                                                       column=acr_result[CHARACTER]))
        acr_result[MARGINAL_PROBABILITIES].to_csv(out_mp_file, sep='\t', index_label='node')
        logging.getLogger('pastml').debug('Serialized marginal probabilities for {} to {}.'
                                          .format(acr_result[CHARACTER], out_mp_file))


def acr(forest, character, states, prediction_method=MPPA, model=F81,
        parameters=None, rate_file=None, force_joint=True,
        reoptimise=False, tau=0, frequency_smoothing=False):
    """
    Reconstructs ancestral states for the given tree(s) and the given character.
    The tree tip/node states should be preannotated with this character.

    :param forest: tree or list of trees whose ancestral states are to be reconstructed.
    :type forest: ete3.Tree or list(ete3.Tree)
    :param character: character for which ancestral states are to be reconstructed.
    :type character: str
    :param states: array of possible character states
    :type states: np.array(str)
    :param model: (optional, default is F81) model to be used by PASTML.
    :type model: str
    :param prediction_method: (optional, default is MPPA) ancestral state prediction method(s) to be used by PASTML.
    :type prediction_method: str
    :param parameters: an optional way to fix some parameters,
        must be in a form {param: value},
        where param can be a character state (then the value should specify its frequency between 0 and 1),
        or pastml.ml.SCALING_FACTOR (then the value should be the scaling factor for three branches,
        e.g. set to 1 to keep the original branches). Could also be represent a path to parameter file.
    :type parameters: dict or str
    :param rate_file: an optional way to fix rates for CUSTOM_RATES model: path to the file containing the rate matrix.
    :type rate_file: str
    :param reoptimise: (False by default) if set to True and the parameters are specified,
        they will be considered as an optimisation starting point instead, and the parameters will be optimised.
    :type reoptimise: bool
    :param force_joint: (optional, default is True) whether the JOINT state should be added to the MPPA prediction
        even when not selected by the Brier score (only used if the prediction method is set to MPPA)
    :type force_joint: bool
    :param tau: a smoothing factor to apply to branch lengths during likelihood calculation.
        If set to zero (default), zero internal branches will be collapsed instead.
    :type tau: float

    :return: an ACR result dictionary.
    :rtype: dict
    """
    forest_stats = ForestStats(forest)
    logger = logging.getLogger('pastml')
    logger.debug('ACR settings for {}:\n\tMethod:\t{}{}.'
                 .format(character, prediction_method,
                         '\n\tModel:\t{}'.format(model) if model and is_ml(prediction_method) else ''))
    if is_parsimonious(prediction_method):
        return parsimonious_acr(forest=forest, character=character, prediction_method=prediction_method,
                                states=states, num_nodes=forest_stats.num_nodes, num_tips=forest_stats.num_tips)
    elif is_ml(prediction_method):
        optimise_tau = tau is None or reoptimise
        if tau is None:
            tau = 0
        missing_data, observed_frequencies, state2index = calculate_observed_freqs(character, forest, states)
        logger.debug('Observed frequencies for {}:{}{}.'
                     .format(character,
                             ''.join('\n\tfrequency of {}:\t{:.6f}'
                                     .format(state, observed_frequencies[state2index[state]]) for state in states),
                             '\n\tfraction of missing data:\t{:.6f}'
                             .format(missing_data) if missing_data else '')
                     )

        if is_ml(prediction_method) and model in {HKY, JTT}:
            initial_states = states
            states = HKY_STATES if HKY == model else JTT_STATES
            if not set(initial_states) & set(states):
                raise ValueError('The allowed states for model {} are {}, '
                                 'but your annotation file/tree specifies {} as states for character {}.'
                                 .format(model, ', '.join(states), ', '.join(initial_states), character))
            if not np.all(states == initial_states):
                refine_states(forest, character, states)

        model_instance = model2class[model](parameter_file=parameters, rate_matrix_file=rate_file,
                                            reoptimise=reoptimise,
                                            frequency_smoothing=frequency_smoothing, tau=tau,
                                            optimise_tau=optimise_tau, states=states, forest_stats=forest_stats,
                                            observed_frequencies=observed_frequencies)
        return ml_acr(forest=forest, character=character, prediction_method=prediction_method,
                      model=model_instance, force_joint=force_joint, observed_frequencies=observed_frequencies)
    else:
        raise ValueError('Method {} is unknown, should be one of ML ({}) or one of MP ({})'
                         .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS)))


def calculate_observed_freqs(character, forest, states):
    n = len(states)
    missing_data = 0.
    state2index = dict(zip(states, range(n)))
    observed_frequencies = np.zeros(n, np.float64)
    for tree in forest:
        for _ in tree:
            state = getattr(_, character, set())
            if state:
                num_node_states = len(state)
                for _ in state:
                    observed_frequencies[state2index[_]] += 1. / num_node_states
            else:
                missing_data += 1
    total_count = observed_frequencies.sum() + missing_data
    observed_frequencies /= observed_frequencies.sum()
    missing_data /= total_count
    return missing_data, observed_frequencies, state2index


def acr_pipeline(tree, data=None, data_sep='\t', id_index=0,
                 column=None, prediction_method=MPPA, model=F81,
                 parameters=None, rate_matrix=None,
                 out_data=None, work_dir=None,
                 verbose=False, forced_joint=False,
                 reoptimise=False, smoothing=False, frequency_smoothing=False):
    """
    Applies PastML to the given tree(s) with the specified states and visualises the result (as html maps).

    :param tree: path to the input tree(s) in newick format (must be rooted).
    :type tree: str

    :param data: (optional) path to the annotation file in tab/csv format with the first row containing the column names.
        If not given, the annotations should be contained in the tree file itself.
    :type data: str
    :param data_sep: (optional, by default '\t') column separator for the annotation table.
        By default, it is set to tab, i.e. for tab-delimited file. Set it to ',' if your file is csv.
    :type data_sep: char
    :param id_index: (optional, by default is 0) index of the column in the annotation table
        that contains the tree tip names, indices start from zero.
    :type id_index: int

    :param column: (optional) name of the annotation table column that contains the character
        to be analysed. If not specified the first columns will be considered.
    :type column: str
    :param prediction_method: (optional, default is pastml.ml.MPPA) ancestral character reconstruction method,
        can be one of the max likelihood (ML) methods: pastml.ml.MPPA, pastml.ml.MAP, pastml.ml.JOINT,
        or one of the max parsimony (MP) methods: pastml.parsimony.ACCTRAN, pastml.parsimony.DELTRAN,
        pastml.parsimony.DOWNPASS.
    :type prediction_method: str
    :param forced_joint: (optional, default is False) add JOINT state to the MPPA state selection
        even if it is not selected by Brier score.
    :type forced_joint: bool
    :param model: (optional, default is pastml.models.f81_like.F81) evolutionary model for ML methods
        (ignored by MP methods).
    :type model: str
    :param parameters: optional way to fix some of the ML-method parameters.
        Could be specified as
        (1) a dict {param: value},
        or (2) as a path to parameter file.
        The file should be tab-delimited, with two columns: the first one containing parameter names,
        and the second, named "value", containing parameter values.
        Parameters can include character state frequencies (parameter name should be the corresponding state,
        and parameter value - the float frequency value, between 0 and 1),
        tree branch scaling factor (parameter name pastml.ml.SCALING_FACTOR),
        and tree branch smoothing factor (parameter name pastml.ml.SMOOTHING_FACTOR).
    :type parameters: str or dict
    :param rate_matrix: (only for pastml.models.rate_matrix.CUSTOM_RATES model) path to the file
        specifying the rate matrix.
        Should be specified as a path to rate matrix file.
        The rate matrix file should specify character states in its first line, preceded by '# ' and separated by spaces.
        The following lines should contain a symmetric squared rate matrix with positive rates
        (and zeros on the diagonal), separated by spaces,
        in the same order at the character states specified in the first line.
        For example for four states, A, C, G, T and the rates A<->C 1, A<->G 4, A<->T 1, C<->G 1, C<->T 4, G<->T 1,
        the rate matrix file would look like:
        # A C G T
        0 1 4 1
        1 0 1 4
        4 1 0 1
        1 4 1 0
    :type rate_matrix: str
    :param reoptimise: (False by default) if set to True and the parameters are specified,
        they will be considered as an optimisation starting point instead, and optimised.
    :type reoptimise: bool
    :param smoothing: (optional, default is False) apply a smoothing factor (optimised) to branch lengths
        during likelihood calculation.
    :type smoothing: bool
    :param frequency_smoothing: (optional, default is False) apply a smoothing factor (optimised) to state frequencies
        (given as input parameters, see parameters argument) during likelihood calculation.
        If the selected model (model argument) does not allow for frequency optimisation, this option will be ignored.
        If reoptimise argument is also set to True, the frequencies will only be smoothed but not reoptimised.
    :type frequency_smoothing: bool
    :param out_data: path to the output annotation file with the reconstructed ancestral character states.
    :type out_data: str
    :param work_dir: (optional) path to the folder where pastml parameter, named tree
        and marginal probability (for marginal ML methods (pastml.ml.MPPA, pastml.ml.MAP) only) files are to be stored.
        Default is <path_to_input_file>/<input_file_name>_pastml. If the folder does not exist, it will be created.
    :type work_dir: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :return: void
    """
    logger = set_up_pastml_logger(verbose)
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    roots = read_forest(tree, columns=[column] if data is None else None)
    _, column2states = annotate_forest(roots, columns=column, data=data, data_sep=data_sep, id_index=id_index,
                                       unknown_treshold=.9, state_threshold=.75)
    column = next(iter(column2states.keys()))
    if parameters:
        if not isinstance(parameters, str) and not isinstance(parameters, dict):
            raise ValueError('Parameters should be either a path to the parameter file or a dict, got {}.'
                             .format(type(parameters)))
    else:
        parameters = None
    if rate_matrix:
        if not isinstance(rate_matrix, str):
            raise ValueError('Rate matrix should be specified as a path to the file, got {}.'.format(type(rate_matrix)))
    else:
        rate_matrix = None
    logger.debug('Finished input validation.')

    acr_result = acr(forest=roots, character=column, states=column2states[column],
                     prediction_method=prediction_method, model=model,
                     parameters=parameters, rate_file=rate_matrix, force_joint=forced_joint,
                     reoptimise=reoptimise,
                     tau=None if smoothing else 0, frequency_smoothing=frequency_smoothing)

    logger.debug('\n=============SAVING RESULTS=============')
    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)
    if not out_data:
        out_data = os.path.join(work_dir, get_ancestral_state_file(column))

    serialize_predicted_states([column], out_data, roots)
    save_tree(roots, columns=[column], nwk=os.path.join(work_dir, get_named_tree_file(tree)))
    serialize_acr((acr_result, work_dir))


def serialize_predicted_states(columns, out_data, roots):
    ids, data = [], []
    # Not using DataFrames to speed up document writing
    with open(out_data, 'w+') as f:
        f.write('node\t{}\n'.format('\t'.join(columns)))
        for root in roots:
            for node in root.traverse():
                vs = []
                column2values = {}
                for column in columns:
                    value = getattr(node, column, set())
                    vs.append(value)
                    if value:
                        column2values[column] = sorted(value, reverse=True)
                data.append(vs)
                ids.append(node.name)
                while column2values:
                    f.write('{}'.format(node.name))
                    for column in columns:
                        if column in column2values:
                            values = column2values[column]
                            value = values.pop()
                            if not values:
                                del column2values[column]
                        else:
                            value = ''
                        f.write('\t{}'.format(value))
                    f.write('\n')
    logging.getLogger('pastml').debug('Serialized reconstructed states to {}.'.format(out_data))


def main():
    """
    Entry point, calling :py:func:`pastml.acr.acr_pipeline` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ancestral character reconstruction "
                                                 "for rooted phylogenetic trees.", prog='pastml_acr')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree', help="input tree(s) in newick format (must be rooted).",
                            type=str, required=True)

    annotation_group = parser.add_argument_group('annotation-file-related arguments')
    annotation_group.add_argument('-d', '--data', required=False, type=str, default=None,
                                  help="annotation file in tab/csv format with the first row "
                                       "containing the column names. "
                                       "If not given, the annotations should be contained in the tree file itself.")
    annotation_group.add_argument('-s', '--data_sep', required=False, type=str, default='\t',
                                  help="column separator for the annotation table. "
                                       "By default is set to tab, i.e. for a tab-delimited file. "
                                       "Set it to ',' if your file is csv.")
    annotation_group.add_argument('-i', '--id_index', required=False, type=int, default=0,
                                  help="index of the annotation table column containing tree tip names, "
                                       "indices start from zero (by default is set to 0).")

    acr_group = parser.add_argument_group('ancestral-character-reconstruction-related arguments')
    acr_group.add_argument('-c', '--column',
                           help="name of the annotation table column that contain characters "
                                "to be analysed. "
                                "If not specified, the first column will be considered.",
                           type=str)
    acr_group.add_argument('--prediction_method',
                           choices=[MPPA, MAP, JOINT, DOWNPASS, ACCTRAN, DELTRAN, COPY],
                           type=str, default=MPPA,
                           help='ancestral character reconstruction (ACR) method, '
                                'can be one of the max likelihood (ML) methods: {ml}, '
                                'or one of the max parsimony (MP) methods: {mp}. '
                           .format(ml=', '.join(ML_METHODS), mp=', '.join(MP_METHODS), copy=COPY, default=MPPA))
    acr_group.add_argument('--forced_joint', action='store_true',
                           help='add {joint} state to the {mppa} state selection '
                                'even if it is not selected by Brier score.'.format(joint=JOINT, mppa=MPPA))
    acr_group.add_argument('-m', '--model', default=F81,
                           choices=[JC, F81, EFT, HKY, JTT, CUSTOM_RATES],
                           type=str,
                           help='evolutionary model for ML methods (ignored by MP methods). ')
    acr_group.add_argument('--parameters', type=str,
                           help='optional way to fix some of the ML-method parameters '
                                'by specifying a file that contain them. '
                                'The file should be tab-delimited, with two columns: '
                                'the first one containing parameter names, '
                                'and the second, named "value", containing parameter values. '
                                'Parameters can include character state frequencies '
                                '(parameter name should be the corresponding state, '
                                'and parameter value - the float frequency value, between 0 and 1),'
                                'tree branch scaling factor (parameter name {}),'.format(SCALING_FACTOR) +
                                'and tree branch smoothing factor (parameter name {}),'.format(SMOOTHING_FACTOR))
    acr_group.add_argument('--rate_matrix', type=str,
                           help='(only for {} model) path to the file containing the rate matrix. '
                                'The rate matrix file should specify character states in its first line, '
                                'preceded by #  and separated by spaces. '
                                'The following lines should contain a symmetric squared rate matrix with positive rates'
                                '(and zeros on the diagonal), separated by spaces, '
                                'in the same order at the character states specified in the first line.'
                                'For example, for four states, A, C, G, T '
                                'and the rates A<->C 1, A<->G 4, A<->T 1, C<->G 1, C<->T 4, G<->T 1,'
                                'the rate matrix file would look like:\n'
                                '# A C G T\n'
                                '0 1 4 1\n'
                                '1 0 1 4\n'
                                '4 1 0 1\n'
                                '1 4 1 0'.format(CUSTOM_RATES))
    acr_group.add_argument('--reoptimise', action='store_true',
                           help='if the parameters are specified, they will be considered as an optimisation '
                                'starting point instead and optimised.')
    acr_group.add_argument('--smoothing', action='store_true',
                           help='Apply a smoothing factor (optimised) to branch lengths during likelihood calculation.')
    acr_group.add_argument('--frequency_smoothing', action='store_true',
                           help='Apply a smoothing factor (optimised) to state frequencies '
                                '(given as input parameters, see --parameters) '
                                'during likelihood calculation. '
                                'If the selected model (--model) does not allow for frequency optimisation,'
                                ' this option will be ignored. '
                                'If --reoptimise is also specified, '
                                'the frequencies will only be smoothed but not reoptimised. ')
    acr_group.add_argument('--resolve_polytomies', action='store_true',
                           help='When specified, the polytomies with a state change '
                                '(i.e. a parent node, P, in state A has more than 2 children, '
                                'including m > 1 children, C_1, ..., C_m, in state B) are resolved '
                                'by grouping together same-state (different from the parent state) nodes '
                                '(i.e. a new internal node N in state B is created and becomes the child of P '
                                'and the parent of C_1, ..., C_m).')

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_data', required=False, type=str,
                           help="path to the output annotation file with the reconstructed ancestral character states.")
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where pastml parameter, named tree "
                                "and marginal probability (for marginal ML methods ({}) only) files are to be stored. "
                                "Default is <path_to_input_tree>/<input_tree_name>_pastml. "
                                "If the folder does not exist, it will be created."
                           .format(', '.join(MARGINAL_ML_METHODS)))
    out_group.add_argument('-v', '--verbose', action='store_true',
                           help="print information on the progress of the analysis (to console)")

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    params = parser.parse_args()

    acr_pipeline(**vars(params))


if '__main__' == __name__:
    main()
