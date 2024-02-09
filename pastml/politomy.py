import logging
import os
from collections import defaultdict

from pastml import col_name2cat, PASTML_VERSION, value2list
from pastml.acr import model2class, serialize_predicted_states
from pastml.annotation import ForestStats, annotate_forest, annotate_dates
from pastml.file import get_pastml_work_dir, get_combined_ancestral_state_file, get_named_tree_file
from pastml.logger import set_up_pastml_logger
from pastml.ml import MPPA, ML_METHODS, is_ml, ml_acr, MAP, JOINT
from pastml.models import SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.models.CustomRatesModel import CUSTOM_RATES
from pastml.models.EFTModel import EFT
from pastml.models.F81Model import F81
from pastml.models.HKYModel import HKY
from pastml.models.JCModel import JC
from pastml.models.JTTModel import JTT
from pastml.parsimony import MP_METHODS, is_parsimonious, parsimonious_acr, DOWNPASS, ACCTRAN, DELTRAN
from pastml.tree import resolve_trees, IS_POLYTOMY, unresolve_trees, read_forest, save_tree, DATE

COPY = 'COPY'


def polytomy_pipeline(tree, data=None, data_sep='\t', id_index=0,
                      columns=None, prediction_method=MPPA, model=F81,
                      parameters=None, rate_matrix=None,
                      out_data=None, work_dir=None,
                      verbose=False, forced_joint=False):
    """
    Resolves polytomies on the given tree(s) based on the ACR for given characters and methods/models.

    :param tree: path to the input tree(s) in newick format (must be rooted).
    :type tree: str

    :param data: (optional) path to the annotation file(s) in tab/csv format with the first row containing the column names.
        If not given, the annotations should be contained in the tree file itself.
    :type data: str
    :param data_sep: (optional, by default '\t') column separator for the annotation table.
        By default, it is set to tab, i.e. for tab-delimited file. Set it to ',' if your file is csv.
    :type data_sep: char
    :param id_index: (optional, by default is 0) index of the column in the annotation table
        that contains the tree tip names, indices start from zero.
    :type id_index: int

    :param columns: (optional) name(s) of the annotation table column(s) that contains the character(s)
        to be analysed. If not specified all the columns will be considered.
    :type columns: str or list(str)
    :param prediction_method: (optional, default is pastml.ml.MPPA) ancestral character reconstruction method(s),
        can be one of the max likelihood (ML) methods: pastml.ml.MPPA, pastml.ml.MAP, pastml.ml.JOINT,
        one of the max parsimony (MP) methods: pastml.parsimony.ACCTRAN, pastml.parsimony.DELTRAN,
        pastml.parsimony.DOWNPASS; or pastml.acr.COPY to keep the annotated character states as-is without inference.
        When multiple ancestral characters are specified (with ``columns`` argument),
        the same method can be used for all of them (if only one method is specified),
        or different methods can be used (specified in the same order as ``columns``).
        If multiple methods are given, but not for all the characters,
        for the rest of them the default method (pastml.ml.MPPA) is chosen.'
    :type prediction_method: str or list(str)
    :param forced_joint: (optional, default is False) add JOINT state to the MPPA state selection
        even if it is not selected by Brier score.
    :type forced_joint: bool
    :param model: (optional, default is pastml.models.f81_like.F81) evolutionary model(s) for ML methods
        (ignored by MP and COPY methods).
        When multiple ancestral characters are specified (with ``columns`` argument),
        the same model can be used for all of them (if only one model is specified),
        or different models can be used (specified in the same order as ``columns``).
        If multiple models are given, but not for all the characters,
        for the rest of them the default model (pastml.models.f81_like.F81) is chosen.
    :type model: str or list(str)
    :param parameters: optional way to fix some of the ML-method parameters.
        Could be specified as
        (1a) a dict {column: {param: value}},
        where column corresponds to the character for which these parameters should be used,
        or (1b) in a form {column: path_to_param_file};
        or (2) as a list of paths to parameter files
        (in the same order as ``columns`` argument that specifies characters)
        possibly given only for the first few characters;
        or (3) as a path to parameter file (only for the first character).
        Each file should be tab-delimited, with two columns: the first one containing parameter names,
        and the second, named "value", containing parameter values.
        Parameters can include character state frequencies (parameter name should be the corresponding state,
        and parameter value - the float frequency value, between 0 and 1),
        tree branch scaling factor (parameter name pastml.ml.SCALING_FACTOR),
        and tree branch smoothing factor (parameter name pastml.ml.SMOOTHING_FACTOR).
    :type parameters: str or list(str) or dict
    :param rate_matrix: (only for pastml.models.rate_matrix.CUSTOM_RATES model) path to the file(s)
        specifying the rate matrix(ces).
        Could be specified as
        (1) a dict {column: path_to_file},
        where column corresponds to the character for which this rate matrix should be used,
        or (2) as a list of paths to rate matrix files
        (in the same order as ``columns`` argument that specifies characters)
        possibly given only for the first few characters;
        or (3) as a path to rate matrix file (only for the first character).
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
    :type rate_matrix: str or list(str) or dict

    :param out_data: path to the output annotation file with the reconstructed ancestral character states.
    :type out_data: str
    :param work_dir: (optional) path to the folder where the named tree and ACR files are to be stored.
        Default is <path_to_input_tree>/<input_tree_name>_pastml. If the folder does not exist, it will be created.
    :type work_dir: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :return: void
    """
    logger = set_up_pastml_logger(verbose)
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    if isinstance(columns, str):
        columns = [columns]
    roots = read_forest(tree, columns=columns if data is None else None)
    columns, column2states = annotate_forest(roots, columns=columns, data=data, data_sep=data_sep, id_index=id_index,
                                             unknown_treshold=.9, state_threshold=.75)
    logger.debug('Finished input validation.')

    if parameters:
        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(parameters, list):
            parameters = dict(zip(columns, parameters))
        elif isinstance(parameters, dict):
            parameters = {col_name2cat(col): params for (col, params) in parameters.items()}
        else:
            raise ValueError('Parameters should be either a list or a dict, got {}.'.format(type(parameters)))
    else:
        parameters = {}

    if rate_matrix:
        if isinstance(rate_matrix, str):
            rate_matrix = [rate_matrix]
        if isinstance(rate_matrix, list):
            rate_matrix = dict(zip(columns, rate_matrix))
        elif isinstance(rate_matrix, dict):
            rate_matrix = {col_name2cat(col): rs for (col, rs) in rate_matrix.items()}
        else:
            raise ValueError('Rate matrices should be either a list or a dict, got {}.'.format(type(rate_matrix)))
    else:
        rate_matrix = {}

    resolve_polytomies_based_on_acr(roots, columns, column2states=column2states, prediction_method=prediction_method,
                                    model=model, column2parameters=parameters, column2rates=rate_matrix,
                                    force_joint=forced_joint)

    logger.debug('\n=============SAVING RESULTS=============')
    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)
    if not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file())

    serialize_predicted_states(sorted(column2states.keys()), out_data, roots)
    save_tree(roots, columns=column2states.keys(), nwk=os.path.join(work_dir, get_named_tree_file(tree)))


def resolve_polytomies_based_on_acr(forest, columns, column2states=None, prediction_method=MPPA, model=F81,
                                    column2parameters=None, column2rates=None, force_joint=True):
    """
    Resolves polytomies in the given forest based on ACR.

    :param forest: tree or list of trees whose ancestral states are to be reconstructed.
    :type forest: ete3.Tree or list(ete3.Tree)
    :param model: (optional, default is F81) model(s) to be used by PASTML,
        can be either one model to be used for all the characters,
        or a list of different models (in the same order as the annotation dataframe columns)
    :type model: str or list(str)
    :param prediction_method: (optional, default is MPPA) ancestral state prediction method(s) to be used by PASTML,
        can be either one method to be used for all the characters,
        or a list of different methods (in the same order as the annotation dataframe columns)
    :type prediction_method: str or list(str)
    :param column2parameters: an optional way to fix some parameters,
        must be in a form {column: {param: value}},
        where param can be a character state (then the value should specify its frequency between 0 and 1),
        or pastml.ml.SCALING_FACTOR (then the value should be the scaling factor for three branches,
        e.g. set to 1 to keep the original branches). Could also be in a form {column: path_to_param_file}.
    :type column2parameters: dict
    :param reoptimise: (False by default) if set to True and the parameters are specified,
        they will be considered as an optimisation starting point instead, and the parameters will be optimised.
    :type reoptimise: bool
    :param force_joint: (optional, default is True) whether the JOINT state should be added to the MPPA prediction
        even when not selected by the Brier score
    :type force_joint: bool
    :param tau: a smoothing factor to apply to branch lengths during likelihood calculation.
        If set to zero (default), zero internal branches will be collapsed instead.
    :type tau: float

    :param threads: (optional, default is 0, which stands for automatic) number of threads PastML can use for parallezation.
        By default, detected automatically based on the system. Note that PastML will at most use as many threads
        as the number of characters (-c option) being analysed plus one.
    :type threads: int

    :return: list of ACR result dictionaries, one per character.
    :rtype: list(dict)
    """

    logger = logging.getLogger('pastml')

    logger.debug('\n=============POLYTOMY RESOLUTION===================')

    column2parameters = column2parameters if column2parameters else defaultdict(lambda: None)
    column2rates = column2rates if column2rates else defaultdict(lambda: None)

    prediction_methods = value2list(len(column2states), prediction_method, MPPA)
    models = value2list(len(column2states), model, F81)

    # If we are going to resolve polytomies we might need to get back to the initial states so let's memorise them
    n2c2states = defaultdict(dict)
    for root in forest:
        for n in root.traverse():
            for c in column2states.keys():
                vs = getattr(n, c, set())
                if vs:
                    n2c2states[n][c] = vs

    # method, model, states, params, optimise
    character2settings = {}
    for (character, prediction_method, model) in zip(columns, prediction_methods, models):
        logging.getLogger('pastml') \
            .debug('ACR settings for {}:\n\tMethod:\t{}{}.'
                   .format(character, prediction_method,
                           '\n\tModel:\t{}'.format(model) if model and is_ml(prediction_method) else ''))
        if COPY == prediction_method or is_parsimonious(prediction_method):
            character2settings[character] = [prediction_method, None]
        elif is_ml(prediction_method):
            model_instance = model2class[model](parameter_file=column2parameters[character],
                                                rate_matrix_file=column2rates[character],
                                                reoptimise=False,
                                                frequency_smoothing=False, states=column2states[character],
                                                forest_stats=ForestStats(forest, character))
            if model_instance.get_num_params():
                raise ValueError('All the parameters must be fixed for polytomy resolution, '
                                 'but it is not the case for character {} (model {}). '
                                 'Please, check you parameter file {}.'
                                 .format(character, model, column2parameters[character]))
            character2settings[character] = [prediction_method, model_instance]
        else:
            raise ValueError('Method {} is unknown, should be one of ML ({}), one of MP ({}) or {}'
                             .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS), COPY))

    def acr():
        for character, (prediction_method, model) in character2settings.items():
            if is_ml(prediction_method):
                model.forest_stats = ForestStats(forest, character)
                ml_acr(forest=forest, character=character, prediction_method=prediction_method,
                       model=model, force_joint=force_joint)
            if is_parsimonious(prediction_method):
                parsimonious_acr(forest=forest, character=character, prediction_method=prediction_method,
                                 states=column2states[character],
                                 num_nodes=sum(sum(1 for _ in tree.traverse()) for tree in forest),
                                 num_tips=sum(len(tree) for tree in forest))

    acr()

    if resolve_trees(column2states, forest):
        level = logger.level
        logger.setLevel(logging.ERROR)
        # we have selected states before, so now need to reset them
        for root in forest:
            for n in root.traverse():
                c2states = n2c2states[n]
                for c in columns:
                    if c in c2states:
                        n.add_feature(c, c2states[c])
                    # if it is a copy method we just need to keep the polytomy state
                    # as there is no way to calculate a state
                    elif not getattr(n, IS_POLYTOMY, False) or not character2settings[c][0] == COPY:
                        n.del_feature(c)

        acr()
        logger.setLevel(level)
        while unresolve_trees(column2states, forest):
            logger.setLevel(logging.ERROR)
            acr()
            logger.setLevel(level)
        logger.setLevel(level)


def main():
    """
    Entry point, calling :py:func:`pastml.polytomy.polytomy_pipeline` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Polytomy resolution based on ancestral characters "
                                                 "in rooted phylogenetic trees.", prog='pastml_polytomy')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree', help="input tree(s) in newick format (must be rooted).",
                            type=str, required=True)

    annotation_group = parser.add_argument_group('annotation-file-related arguments')
    annotation_group.add_argument('-d', '--data', required=False, type=str, nargs='*', default=None,
                                  help="annotation file(s) in tab/csv format with the first row "
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
    acr_group.add_argument('-c', '--columns', nargs='*',
                           help="names of the annotation table columns that contain characters "
                                "to be analysed. "
                                "If not specified, all columns are considered.",
                           type=str)
    acr_group.add_argument('--prediction_method',
                           choices=[MPPA, MAP, JOINT, DOWNPASS, ACCTRAN, DELTRAN, COPY],
                           type=str, nargs='*', default=MPPA,
                           help='ancestral character reconstruction (ACR) method, '
                                'can be one of the max likelihood (ML) methods: {ml}, '
                                'one of the max parsimony (MP) methods: {mp}; '
                                'or {copy} to keep the annotated character states as-is without inference. '
                                'When multiple ancestral characters are specified (see -c, --columns), '
                                'the same method can be used for all of them (if only one method is specified), '
                                'or different methods can be used (specified in the same order as -c, --columns). '
                                'If multiple methods are given, but not for all the characters, '
                                'for the rest of them the default method ({default}) is chosen.'
                           .format(ml=', '.join(ML_METHODS), mp=', '.join(MP_METHODS), copy=COPY, default=MPPA))
    acr_group.add_argument('--forced_joint', action='store_true',
                           help='add {joint} state to the {mppa} state selection '
                                'even if it is not selected by Brier score.'.format(joint=JOINT, mppa=MPPA))
    acr_group.add_argument('-m', '--model', default=F81,
                           choices=[JC, F81, EFT, HKY, JTT, CUSTOM_RATES],
                           type=str, nargs='*',
                           help='evolutionary model for ML methods (ignored by MP methods). '
                                'When multiple ancestral characters are specified (see -c, --columns), '
                                'the same model can be used for all of them (if only one model is specified), '
                                'or different models can be used (specified in the same order as -c, --columns). '
                                'If multiple models are given, but not for all the characters, '
                                'for the rest of them the default model ({}) is chosen.'.format(F81))
    acr_group.add_argument('--parameters', type=str, nargs='*',
                           help='if the specified ML model has parameters, they should be fixed '
                                'by specifying files that contain them. '
                                'Should be in the same order '
                                'as the ancestral characters (see -c, --columns). '
                                'Could be given only for the first few characters. '
                                'Each file should be tab-delimited, with two columns: '
                                'the first one containing parameter names, '
                                'and the second, named "value", containing parameter values. '
                                'Parameters can include character state frequencies '
                                '(parameter name should be the corresponding state, '
                                'and parameter value - the float frequency value, between 0 and 1),'
                                'tree branch scaling factor (parameter name {}),'.format(SCALING_FACTOR) +
                                'and tree branch smoothing factor (parameter name {}),'.format(SMOOTHING_FACTOR))
    acr_group.add_argument('--rate_matrix', type=str, nargs='*',
                           help='(only for {} model) path to the file(s) containing the rate matrix(ces). '
                                'Should be in the same order '
                                'as the ancestral characters (see -c, --columns). '
                                'Could be given only for the first few characters. '
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

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_data', required=False, type=str,
                           help="path to the output annotation file with the reconstructed ancestral character states.")
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where a named tree with (partially) resolved polytomies "
                                "and ACR files are to be stored. "
                                "Default is <path_to_tree_file>/<input_tree_name>_pastml. "
                                "If the folder does not exist, it will be created.")
    out_group.add_argument('-v', '--verbose', action='store_true',
                           help="print information on the progress of the analysis (to console)")

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    params = parser.parse_args()

    polytomy_pipeline(**vars(params))


if '__main__' == __name__:
    main()
