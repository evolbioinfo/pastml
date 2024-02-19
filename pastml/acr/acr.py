import logging
import os
import warnings

import numpy as np

from pastml import PASTML_VERSION, value2list
from pastml.acr import METHOD, CHARACTER, save_acr_stats
from pastml.acr.maxlikelihood.ml import ml_acr
from pastml.annotation import annotate_forest, ForestStats
from pastml.file import get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file, get_pastml_work_dir, get_ancestral_state_file, get_pastml_stats_file
from pastml.logger import set_up_pastml_logger
from pastml.acr.maxlikelihood import MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, \
    ML_METHODS, MAP, JOINT, MARGINAL_ML_METHODS
from pastml.acr.maxlikelihood.models import MODEL
from pastml.acr.maxlikelihood.models.SimpleModel import SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.acr.maxlikelihood.models.CustomRatesModel import CustomRatesModel, CUSTOM_RATES
from pastml.acr.maxlikelihood.models.EFTModel import EFTModel, EFT
from pastml.acr.maxlikelihood.models.F81Model import F81Model, F81
from pastml.acr.maxlikelihood.models.HKYModel import HKYModel, HKY
from pastml.acr.maxlikelihood.models.JCModel import JCModel, JC
from pastml.acr.maxlikelihood.models.JTTModel import JTTModel, JTT
from pastml.acr.maxlikelihood.models.SkylineModel import SkylineModel, annotate_skyline
from pastml.acr.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS
from pastml.tree import read_forest, save_tree

model2class = {F81: F81Model, JC: JCModel, CUSTOM_RATES: CustomRatesModel, HKY: HKYModel, JTT: JTTModel, EFT: EFTModel}

warnings.filterwarnings("ignore", append=True)


def serialize_acr(args):
    logger = logging.getLogger('pastml')
    acr_result, work_dir = args
    out_stats_file = os.path.join(work_dir,
                                  get_pastml_stats_file(model=acr_result[MODEL].name if MODEL in acr_result else None,
                                                        method=acr_result[METHOD], column=acr_result[CHARACTER]))

    save_acr_stats(acr_result, out_stats_file)
    logger.debug('Serialized ACR statistics for {} to {}.'
                 .format(acr_result[CHARACTER], out_stats_file))

    if is_ml(acr_result[METHOD]):
        out_param_file = \
            os.path.join(work_dir,
                         get_pastml_parameter_file(model=acr_result[MODEL].name if MODEL in acr_result else None,
                                                   column=acr_result[CHARACTER]))
        out_param_file = acr_result[MODEL].save_parameters(out_param_file)

        logger.debug('Serialized {} model parameters for {} to {}.'
                     .format(acr_result[MODEL].name, acr_result[CHARACTER], out_param_file))

        if is_marginal(acr_result[METHOD]):
            out_mp_file = \
                os.path.join(work_dir,
                             get_pastml_marginal_prob_file(model=acr_result[MODEL].name,
                                                           column=acr_result[CHARACTER]))
            acr_result[MARGINAL_PROBABILITIES].to_csv(out_mp_file, sep='\t', index_label='node')
            logger.debug('Serialized marginal probabilities for {} under model {} to {}.'
                         .format(acr_result[CHARACTER], acr_result[MODEL].name, out_mp_file))


def acr(forest, character, states, prediction_method=MPPA, model=F81,
        parameters=None, rate_file=None, force_joint=True,
        reoptimise=False, tau=0, frequency_smoothing=False,
        skyline=None, skyline_mapping=None):
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
    :type model: str or list(str)
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
    logger = logging.getLogger('pastml')
    logger.debug('ACR settings for {}:\n\tMethod:\t{}{}.'
                 .format(character, prediction_method,
                         '\n\tModel:\t{}'.format(model) if model and is_ml(prediction_method) else ''))
    if is_parsimonious(prediction_method):
        forest_stats = ForestStats(forest, character)
        logger.debug(forest_stats)
        return parsimonious_acr(forest=forest, character=character, prediction_method=prediction_method,
                                states=states, num_nodes=forest_stats.n_nodes, num_tips=forest_stats.n_tips)
    elif is_ml(prediction_method):
        n_sky = (len(skyline) if skyline else 0) + 1
        model = value2list(n_sky, model, F81)
        parameters = value2list(n_sky, parameters, None)
        rate_file = rate_matrix2list(model, rate_file)

        optimise_tau = tau is None or reoptimise
        if tau is None:
            tau = 0

        forest_stats = ForestStats(forest, character)

        if skyline:
            start_date = -np.inf
            skyline_models = []
            for i in range(n_sky):
                end_date = skyline[i] if i < len(skyline) else np.inf
                sub_forest_stats = ForestStats(forest, character, start_date, end_date)
                logger.debug(sub_forest_stats)
                m = model2class[model[i]](parameter_file=parameters[i],
                                          rate_matrix_file=rate_file[i],
                                          reoptimise=reoptimise, frequency_smoothing=frequency_smoothing, tau=tau,
                                          optimise_tau=optimise_tau,
                                          states=states if not skyline_mapping else skyline_mapping[1][i],
                                          forest_stats=sub_forest_stats)
                skyline_models.append(m)
                start_date = end_date
            model_instance = SkylineModel(models=skyline_models, dates=skyline,
                                          skyline_mapping=skyline_mapping[0] if skyline_mapping else None,
                                          forest_stats=forest_stats)
        else:
            logger.debug(forest_stats)
            model_instance = model2class[model[0]](parameter_file=parameters[0], rate_matrix_file=rate_file[0],
                                                   reoptimise=reoptimise,
                                                   frequency_smoothing=frequency_smoothing, tau=tau,
                                                   optimise_tau=optimise_tau, states=states, forest_stats=forest_stats)
        return ml_acr(forest=forest, character=character, prediction_method=prediction_method,
                      model=model_instance, force_joint=force_joint)
    else:
        raise ValueError('Method {} is unknown, should be one of ML ({}) or one of MP ({})'
                         .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS)))


def rate_matrix2list(models, rate_file):
    rate_matrices = []
    i = 0
    if not rate_file:
        rate_file = []
    if isinstance(rate_file, str):
        rate_file = [rate_file]
    for m in models:
        if CUSTOM_RATES == m:
            if i > len(rate_file):
                raise ValueError('A rate matrix must be specified for each {} model.'
                                 .format(CUSTOM_RATES))
            rate_matrices.append(rate_file[i])
            i += 1
        else:
            rate_matrices.append(None)
    if i < len(rate_file):
        raise ValueError('Extra rate matrices are specified '
                         '(as many rate matrices must be given as intervals with the {} model).'
                         .format(CUSTOM_RATES))
    rate_file = rate_matrices
    return rate_file


def acr_pipeline(tree, data=None, data_sep='\t', id_index=0,
                 column=None, prediction_method=MPPA, model=None,
                 parameters=None, rate_matrix=None,
                 out_data=None, work_dir=None,
                 verbose=False, forced_joint=False,
                 reoptimise=False, smoothing=False, frequency_smoothing=False,
                 skyline=None, skyline_mapping=None):
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

    :param model: evolutionary model for ML methods (ignored by MP methods).
        If a skyline with M intervals is used, then up to M models can be specified.
        If there are M skyline intervals, and N < M models,
        they will be associated with the first (oldest, closest to the root) N intervals,
        and the other M-N intervals will get the default model (pastml.models.f81_like.F81).
    :type model: str or list(str)

    :param parameters: optional way to fix (some of) the ML-method parameters.
        Could be specified as
        (1) a dict {param: value},
        or (2) as a path to parameter file,
        or (3) a list of (1) or (2).
        The file should be tab-delimited, with two columns: the first one containing parameter names,
        and the second, named "value", containing parameter values.
        Parameters depend on the selected model. Common parameter examples include
        character state frequencies (parameter name should be the corresponding state,
        and parameter value - the float frequency value, between 0 and 1),
        tree branch scaling factor (parameter name pastml.ml.SCALING_FACTOR),
        and tree branch smoothing factor (parameter name pastml.ml.SMOOTHING_FACTOR).
        If a skyline with M intervals is used, then up to M parameter files/dictionaries can be specified. '
        If there are M skyline intervals, and N < M parameter files/dictionaries,
        they will be associated with the first (oldest, closest to the root) N intervals.
        If needed, some of these files can be empty.
    :type parameters: str or dict or list(str) or list(dict)

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
        1 4 1 0.
        If a skyline is used with several intervals using the pastml.models.rate_matrix.CUSTOM_RATES model,
        then as many paths to the corresponding matrix files should be specified.
    :type rate_matrix: str or list(str)

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

    :param skyline: date(s) of the changes in model parameters
        (for non-dated tree(s) should be expressed as distance to root), if specified,
        character states, models and their parameters will be allowed to change between the time intervals.
    :type skyline: list(float)

    :param skyline_mapping: optional way to specify the mappings between different skyline states if they change over time.
        Should be specified as a path to the mapping file (only for the first character).
        The file should be tab-delimited, with length of the skyline + 1 columns:
        the first one containing character states at the date of the latest tip and named as the character,
        and the others, named by skyline dates, containing the corresponding states at those years, e.g.
        country 1991    1917
        Russia  USSR    Russian Empire
        Belarus  USSR    Russian Empire
        France  France  France
        ... ... ...
    :type skyline_mapping: str

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
        if isinstance(parameters, str) or isinstance(parameters, dict):
            parameters = [parameters]
        if not isinstance(parameters, list):
            raise ValueError('Parameters should be either a path to the parameter file or a dict '
                             '(or several paths/dictionaries if the skyline is used), got {} instead.'
                             .format(type(parameters)))
    else:
        parameters = []

    if rate_matrix:
        if isinstance(rate_matrix, str):
            rate_matrix = [rate_matrix]
        if not isinstance(rate_matrix, list):
            raise ValueError('Rate matrix should be be specified as a path to the file '
                             '(or several paths if the skyline is used), got {} instead.'
                             .format(type(rate_matrix)))
    else:
        rate_matrix = []

    if skyline:
        skyline, skyline_mapping = annotate_skyline(roots, skyline, column, skyline_mapping)

    logger.debug('Finished input validation.')

    acr_result = acr(forest=roots, character=column, states=column2states[column],
                     prediction_method=prediction_method, model=model,
                     parameters=parameters, rate_file=rate_matrix, force_joint=forced_joint,
                     reoptimise=reoptimise,
                     tau=None if smoothing else 0, frequency_smoothing=frequency_smoothing,
                     skyline_mapping=skyline_mapping, skyline=skyline)

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
                           choices=[MPPA, MAP, JOINT, DOWNPASS, ACCTRAN, DELTRAN],
                           type=str, default=MPPA,
                           help='ancestral character reconstruction (ACR) method, '
                                'can be one of the max likelihood (ML) methods: {ml}, '
                                'or one of the max parsimony (MP) methods: {mp}. '
                           .format(ml=', '.join(ML_METHODS), mp=', '.join(MP_METHODS), default=MPPA))
    acr_group.add_argument('--forced_joint', action='store_true',
                           help='add {joint} state to the {mppa} state selection '
                                'even if it is not selected by Brier score.'.format(joint=JOINT, mppa=MPPA))
    acr_group.add_argument('-m', '--model',
                           choices=[JC, F81, EFT, HKY, JTT, CUSTOM_RATES],
                           type=str, nargs='*',
                           help='evolutionary model for ML methods (ignored by MP methods).'
                                'If a skyline with M intervals is used, '
                                'then up to M models can be specified. '
                                'If there are M skyline intervals, and N < M models, '
                                'they will be associated with the first (oldest, closest to the root) N intervals, '
                                'and the other M-N intervals will get the default model ({}).'.format(F81)
                           )
    acr_group.add_argument('--parameters', type=str, nargs='*',
                           help='optional way to fix some of the ML-method parameters '
                                'by specifying a file that contain them. '
                                'The file should be tab-delimited, with two columns: '
                                'the first one containing parameter names, '
                                'and the second, named "value", containing parameter values. '
                                'Parameters depend on the selected model. '
                                'Common parameter examples include character state frequencies '
                                '(parameter name should be the corresponding state, '
                                'and parameter value - the float frequency value, between 0 and 1),'
                                'tree branch scaling factor (parameter name {}),'.format(SCALING_FACTOR) +
                                'and tree branch smoothing factor (parameter name {}).\n'.format(SMOOTHING_FACTOR) +
                                'If a skyline with M intervals is used, '
                                'then up to M parameter files can be specified. '
                                'If there are M skyline intervals, and N < M parameter files, '
                                'they will be associated with the first (oldest, closest to the root) N intervals. '
                                'If needed, some of these files can be empty.')
    acr_group.add_argument('--rate_matrix', type=str, nargs='*',
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
                                '1 4 1 0\n'
                                'If a skyline is used with several intervals using the {} model, '
                                'then as many paths to the corresponding matrix files should be specified.'
                           .format(CUSTOM_RATES, CUSTOM_RATES))
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
                                'the frequencies will only be smoothed but not reoptimised.')
    acr_group.add_argument('--skyline', required=False, default=None,
                           help="date(s) of the changes in model parameters "
                                "(for non-dated tree(s) should be expressed as distance to root), "
                                "if specified, character states, models and their parameters "
                                "will be allowed to change between the time intervals.",
                           type=float, nargs='*')
    acr_group.add_argument('--skyline_mapping', required=False, default=None,
                           help="optional way to specify the mappings between different skyline states, "
                                "if they change over time. "
                                "Should contain the paths to the mapping file, "
                                "which should be tab-delimited, with length of the skyline + 1 columns:"
                                "the first one containing character states at the date of the latest tip "
                                "and named as the character,"
                                "and the others, named by skyline dates, "
                                "containing the corresponding states at those years, e.g.\n"
                                "country\t1991\t1917\n"
                                "Russia\tUSSR\tRussian Empire\n"
                                "Belarus\tUSSR\tRussian Empire\n"
                                "France\tFrance\tFrance\n"
                                "...\t...\t...",
                           type=str)

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

# TODO: add root date parameter