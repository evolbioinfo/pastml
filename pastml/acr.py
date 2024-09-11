import logging
import os
import sys
import warnings
from collections import defaultdict, Counter
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from Bio.Phylo import NewickIO, write
from Bio.Phylo.NewickIO import StringIO
from ete3 import Tree

from pastml import col_name2cat, value2list, STATES, METHOD, CHARACTER, get_personalized_feature_name, numeric2datetime, \
    PASTML_VERSION, _set_up_pastml_logger
from pastml.annotation import preannotate_forest, ForestStats
from pastml.file import get_combined_ancestral_state_file, get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file, get_pastml_work_dir
from pastml.ml import MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, ml_acr, \
    ML_METHODS, MAP, JOINT, ALL, ML, META_ML_METHODS, MARGINAL_ML_METHODS, get_default_ml_method
from pastml.models import MODEL, SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.models.CustomRatesModel import CustomRatesModel, CUSTOM_RATES
from pastml.models.EFTModel import EFTModel, EFT
from pastml.models.F81Model import F81Model, F81
from pastml.models.HKYModel import HKYModel, HKY, HKY_STATES
from pastml.models.JCModel import JCModel, JC
from pastml.models.JTTModel import JTTModel, JTT, JTT_STATES
from pastml.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS, MP, \
    get_default_mp_method
from pastml.tree import name_tree, annotate_dates, DATE, read_forest, DATE_CI, resolve_trees, IS_POLYTOMY, \
    unresolve_trees, clear_extra_features, parse_date
from pastml.visualisation import get_formatted_date
from pastml.visualisation.cytoscape_manager import visualize, TIMELINE_SAMPLED, TIMELINE_NODES, TIMELINE_LTT, \
    DIST_TO_ROOT_LABEL, DATE_LABEL
from pastml.visualisation.itol_manager import generate_itol_annotations
from pastml.visualisation.tree_compressor import REASONABLE_NUMBER_OF_TIPS, VERTICAL, HORIZONTAL, TRIM

model2class = {F81: F81Model, JC: JCModel, CUSTOM_RATES: CustomRatesModel, HKY: HKYModel, JTT: JTTModel, EFT: EFTModel}

warnings.filterwarnings("ignore", append=True)

COPY = 'COPY'


def _serialize_acr(args):
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


def acr(forest, df=None, columns=None, column2states=None, prediction_method=MPPA, model=F81,
        column2parameters=None, column2rates=None,
        force_joint=True, threads=0,
        reoptimise=False, tau=0, resolve_polytomies=False, frequency_smoothing=False):
    """
    Reconstructs ancestral states for the given tree and
    all the characters specified as columns of the given annotation dataframe.

    :param df: dataframe indexed with tree node names
        and containing characters for which ACR should be performed as columns.
    :type df: pandas.DataFrame
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

    if isinstance(forest, Tree):
        forest = [forest]

    if columns is None:
        if df is None:
            raise ValueError('Either the tree should be preannotated with character values '
                             'and columns and column2states specified, '
                             'or an annotation dataframe provided!')
        columns = df.columns
        column2states = {column: np.array(sorted([_ for _ in df[column].unique() if not pd.isna(_) and '' != _]))
                         for column in columns}
        preannotate_forest(forest, df=df)

    forest_stats = ForestStats(forest)

    logging.getLogger('pastml').debug('\n=============ACR===============================')

    column2parameters = column2parameters if column2parameters else {}
    column2rates = column2rates if column2rates else {}

    prediction_methods = value2list(len(columns), prediction_method, MPPA)
    models = value2list(len(columns), model, F81)

    def get_states(method, model, column):
        initial_states = column2states[column]
        if not is_ml(method) or model not in {HKY, JTT}:
            return initial_states
        states = HKY_STATES if HKY == model else JTT_STATES
        if not set(initial_states) & set(states):
            raise ValueError('The allowed states for model {} are {}, '
                             'but your annotation file specifies {} as states in column {}.'
                             .format(model, ', '.join(states), ', '.join(initial_states), column))
        state_set = set(states)
        for root in forest:
            for n in root.traverse():
                if hasattr(n, column):
                    n.add_feature(column, state_set & getattr(n, column))
        return states

    # If we gonna resolve polytomies we might need to get back to the initial states so let's memorise them
    n2c2states = defaultdict(dict)
    for root in forest:
        for n in root.traverse():
            for c in columns:
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
            states = get_states(prediction_method, model, character)
            character2settings[character] = [prediction_method, states]
        elif is_ml(prediction_method):
            params = column2parameters[character] if character in column2parameters else None
            rate_file = column2rates[character] if character in column2rates else None
            optimise_tau = tau is None or reoptimise
            if tau is None:
                tau = 0
            states = get_states(prediction_method, model, character)

            missing_data, observed_frequencies, state2index = calculate_observed_freqs(character, forest, states)

            logger = logging.getLogger('pastml')
            logger.debug('Observed frequencies for {}:{}{}.'
                         .format(character,
                                 ''.join('\n\tfrequency of {}:\t{:.6f}'
                                         .format(state, observed_frequencies[state2index[state]]) for state in states),
                                 '\n\tfraction of missing data:\t{:.6f}'
                                 .format(missing_data) if missing_data else '')
                         )

            model_instance = model2class[model](parameter_file=params, rate_matrix_file=rate_file, reoptimise=reoptimise,
                                                frequency_smoothing=frequency_smoothing, tau=tau,
                                                optimise_tau=optimise_tau, states=states, forest_stats=forest_stats,
                                                observed_frequencies=observed_frequencies)
            character2settings[character] = [prediction_method, model_instance]
        else:
            raise ValueError('Method {} is unknown, should be one of ML ({}), one of MP ({}) or {}'
                             .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS), COPY))

    if threads < 1:
        threads = max(os.cpu_count(), 1)

    def _work(character):
        prediction_method, model_or_states = character2settings[character]
        if COPY == prediction_method:
            return {CHARACTER: character, STATES: model_or_states, METHOD: prediction_method}
        if is_ml(prediction_method):
            return ml_acr(forest=forest, character=character, prediction_method=prediction_method,
                          model=model_or_states,
                          force_joint=force_joint, observed_frequencies=observed_frequencies)
        if is_parsimonious(prediction_method):
            return parsimonious_acr(forest=forest, character=character, prediction_method=prediction_method,
                                    states=model_or_states,
                                    num_nodes=forest_stats.num_nodes, num_tips=forest_stats.num_tips)

    if threads > 1:
        with ThreadPool(processes=threads - 1) as pool:
            acr_results = \
                pool.map(func=_work, iterable=character2settings.keys())
    else:
        acr_results = [_work(character) for character in character2settings.keys()]

    acr_results = flatten_lists(acr_results)

    column2states = {acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results}
    column2copy = {acr_result[CHARACTER]: acr_result[METHOD] == COPY for acr_result in acr_results}
    if resolve_polytomies and resolve_trees(column2states, forest):
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
                    elif not getattr(n, IS_POLYTOMY, False) or not column2copy[c]:
                        n.del_feature(c)

        forest_stats = ForestStats(forest)
        for acr_res in acr_results:
            character = acr_res[CHARACTER]
            method = acr_res[METHOD]
            if is_ml(method):
                if character not in character2settings:
                    character = character[:character.rfind('_{}').format(method)]
                character2settings[character][1].freeze()
                character2settings[character][1].forest_stats = forest_stats
        if threads > 1:
            with ThreadPool(processes=threads - 1) as pool:
                acr_results = \
                    pool.map(func=_work, iterable=character2settings.keys())
        else:
            acr_results = [_work(character) for character in character2settings.keys()]
        logger.setLevel(level)
        while unresolve_trees(column2states, forest):
            logger.setLevel(logging.ERROR)
            if threads > 1:
                with ThreadPool(processes=threads - 1) as pool:
                    acr_results = \
                        pool.map(func=_work, iterable=character2settings.keys())
            else:
                acr_results = [_work(character) for character in character2settings.keys()]
            logger.setLevel(level)
        logger.setLevel(level)
        acr_results = flatten_lists(acr_results)
    return acr_results


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


def flatten_lists(lists):
    result = []
    for _ in lists:
        if isinstance(_, list):
            result.extend(_)
        else:
            result.append(_)
    return result


def _quote(str_list):
    return ', '.join('"{}"'.format(_) for _ in str_list) if str_list is not None else ''


def pastml_pipeline(tree, data=None, data_sep='\t', id_index=0,
                    columns=None, prediction_method=MPPA, model=F81,
                    parameters=None, rate_matrix=None,
                    name_column=None, root_date=None, timeline_type=TIMELINE_SAMPLED,
                    tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, colours=None,
                    out_data=None, html_compressed=None, html=None, html_mixed=None, work_dir=None,
                    verbose=False, forced_joint=False, upload_to_itol=False, itol_id=None, itol_project=None,
                    itol_tree_name=None, offline=False, threads=0, reoptimise=False, focus=None,
                    resolve_polytomies=False, smoothing=False, frequency_smoothing=False,
                    pajek=None, pajek_timing=VERTICAL, recursion_limit=0):
    """
    Applies PastML to the given tree(s) with the specified states and visualises the result (as html maps).

    :param tree: path to the input tree(s) in newick format (must be rooted).
    :type tree: str

    :param data: (optional) path to the annotation file in tab/csv format with the first row containing the column names.
        If not given, the annotations should be contained in the tree file itself.
    :type data: str
    :param data_sep: (optional, by default '\t') column separator for the annotation table.
        By default is set to tab, i.e. for tab-delimited file. Set it to ',' if your file is csv.
    :type data_sep: char
    :param id_index: (optional, by default is 0) index of the column in the annotation table
        that contains the tree tip names, indices start from zero.
    :type id_index: int

    :param columns: (optional) name(s) of the annotation table column(s) that contain character(s)
        to be analysed. If not specified all annotation table columns will be considered.
    :type columns: str or list(str)
    :param prediction_method: (optional, default is pastml.ml.MPPA) ancestral character reconstruction method(s),
        can be one of the max likelihood (ML) methods: pastml.ml.MPPA, pastml.ml.MAP, pastml.ml.JOINT,
        one of the max parsimony (MP) methods: pastml.parsimony.ACCTRAN, pastml.parsimony.DELTRAN,
        pastml.parsimony.DOWNPASS; or pastml.acr.COPY to keep the annotated character states as-is without inference.
        One can also specify one of the meta-methods: pastml.ml.ALL, pastml.ml.ML, pastml.parsimony.MP,
        that would perform ACR with multiple methods (all of them for pastml.ml.ALL,
        all the ML methods for pastml.ml.ML, or all the MP methods for pastml.parsimony.MP)
        and save/visualise the results as multiple characters suffixed with the corresponding method.
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
        (ignored by MP methods).
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
    :param name_column: (optional) name of the annotation table column to be used for node names
        in the compressed map visualisation
        (must be one of those specified in ``columns``, if ``columns`` are specified).
        If the annotation table contains only one column, it will be used by default.
    :type name_column: str
    :param root_date: (optional) date(s) of the root(s) (for dated tree(s) only),
        if specified, used to visualise a timeline based on dates (otherwise it is based on distances to root).
    :type root_date: str or pandas.datetime or float or list
    :param tip_size_threshold: (optional, by default is 15) recursively remove the tips
        of size less than threshold-th largest tip from the compressed map (set to 1e10 to keep all).
        The larger it is the less tips will be trimmed.
    :type tip_size_threshold: int
    :param focus: optional way to put a focus on certain character state values,
        so that the nodes in these states are displayed
        even if they do not pass the trimming threshold (tip_size_threshold argument).
        Should be in the form character:state.
    :type focus: str or list(str)
    :param timeline_type: (optional, by default is pastml.visualisation.cytoscape_manager.TIMELINE_SAMPLED)
        type of timeline visualisation: at each date/distance to root selected on the slider, either
        (pastml.visualisation.cytoscape_manager.TIMELINE_SAMPLED) all the lineages sampled after it are hidden; "
        or (pastml.visualisation.cytoscape_manager.TIMELINE_NODES) all the nodes with a
        more recent date/larger distance to root are hidden;
        or (pastml.visualisation.cytoscape_manager.TIMELINE_LTT) all the nodes whose branch started
        after this date/distance to root are hidden, and the external branches are cut
        to the specified date/distance to root if needed;
    :type timeline_type: str
    :param colours: optional way to specify the colours used for character state visualisation.
        Could be specified as
        (1a) a dict {column: {state: colour}},
        where column corresponds to the character for which these parameters should be used,
        or (1b) in a form {column: path_to_colour_file};
        or (2) as a list of paths to colour files
        (in the same order as ``columns`` argument that specifies characters)
        possibly given only for the first few characters;
        or (3) as a path to colour file (only for the first character).
        Each file should be tab-delimited, with two columns: the first one containing character states,
        and the second, named "colour", containing colours, in HEX format (e.g. #a6cee3).
    :type colours: str or list(str) or dict
    :param resolve_polytomies: (default False) when True, the polytomies with a state change
        (i.e. a parent node, P, in state A has more than 2 children, including m > 1 children, C_1, ..., C_m, in state B)
        are resolved by grouping together same-state (different from the parent state) nodes
        (i.e. a new internal node N in state B is created and becomes the child of P and the parent of C_1, ..., C_m).
    :type resolve_polytomies: bool

    :param out_data: path to the output annotation file with the reconstructed ancestral character states.
    :type out_data: str
    :param html_compressed: path to the output compressed visualisation file (html).
    :type html_compressed: str
    :param pajek: path to the output compressed visualisation file (Pajek NET Format).
        Produced only if html_compressed is specified.
    :type pajek: str
    :param pajek_timing: the type of the compressed visualisation to be saved in Pajek NET Format (if pajek is specified).
        Can be either 'VERTICAL' (default, after the nodes underwent vertical compression),
        'HORIZONTAL' (after the nodes underwent vertical and horizontal compression)
        or 'TRIM' (after the nodes underwent vertical and horizontal compression and minor node trimming).
    :type pajek_timing: str
    :param html: (optional) path to the output tree visualisation file (html).
    :type html: str
    :param html_mixed: (optional) path to the output mostly compressed map visualisation file (html),
        where the nodes in states specified with the focus argument are uncompressed.
    :type html_mixed: str
    :param work_dir: (optional) path to the folder where pastml parameter, named tree
        and marginal probability (for marginal ML methods (pastml.ml.MPPA, pastml.ml.MAP) only) files are to be stored.
        Default is <path_to_input_file>/<input_file_name>_pastml. If the folder does not exist, it will be created.
    :type work_dir: str
    :param offline: (optional, default is False) By default (offline=False) PastML assumes
        that there is an internet connection available,
        which permits it to fetch CSS and JS scripts needed for visualisation online.
        With offline=True, PastML will store all the needed CSS/JS scripts in the folder specified by work_dir,
        so that internet connection is not needed
        (but you must not move the output html files to any location other that the one specified by html/html_compressed.
    :type offline: bool

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :param threads: (optional, default is 0, which stands for automatic) number of threads PastML can use for parallesation.
        By default, detected automatically based on the system. Note that PastML will at most use as many threads
        as the number of characters (-c option) being analysed plus one.
    :type threads: int

    :parameter recursion_limit: Set the recursion limit in python.
        If your tree is very large and pastml produces an overflow error,
        you could try to set the recursion limit to a large number (e.g., 10000).
        By default the standard recursion limit is used and should be sufficient
        for trees of up to several thousand tips.
    :type recursion_limit: int

    :param upload_to_itol: (optional, default is False) whether iTOL annotations
        for the reconstructed characters associated with the named tree (i.e. the one found in work_dir) should be created.
        If additionally itol_id and itol_project are specified,
        the annotated tree will be automatically uploaded to iTOL (https://itol.embl.de/).
    :type upload_to_itol: bool
    :param itol_id: (optional) iTOL user batch upload ID that enables uploading to your iTOL account
        (see https://itol.embl.de/help.cgi#batch).
    :type itol_id: str
    :param itol_project: (optional) iTOL project the annotated tree should be uploaded to
        (must exist, and itol_id must be specified). If not specified, the tree will not be associated to any project.
    :type itol_project: str
    :param itol_tree_name: (optional) name for the tree uploaded to iTOL.
    :type itol_tree_name: str

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    if recursion_limit and recursion_limit > 0:
        sys.setrecursionlimit(recursion_limit)

    copy_only = COPY == prediction_method or (isinstance(prediction_method, list)
                                              and all(COPY == _ for _ in prediction_method))

    roots, columns, column2states, name_column, age_label, parameters, rates = \
        _validate_input(tree, columns, name_column if html_compressed or html_mixed else None, data, data_sep, id_index,
                        root_date if html_compressed or html or html_mixed or upload_to_itol else None,
                        copy_only=copy_only, parameters=parameters, rates=rate_matrix)

    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)

    if threads < 1:
        threads = max(os.cpu_count(), 1)

    acr_results = acr(forest=roots, columns=columns, column2states=column2states,
                      prediction_method=prediction_method, model=model, column2parameters=parameters,
                      column2rates=rates,
                      force_joint=forced_joint, threads=threads, reoptimise=reoptimise, tau=None if smoothing else 0,
                      resolve_polytomies=resolve_polytomies, frequency_smoothing=frequency_smoothing)

    column2states = {acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results}

    if not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file())

    state_df = _serialize_predicted_states(sorted(column2states.keys()), out_data, roots,
                                           dates_are_dates=age_label == DATE_LABEL)

    # a meta-method would have added a suffix to the name feature
    if html_compressed and name_column and name_column not in column2states:
        ml_name_column = get_personalized_feature_name(name_column, get_default_ml_method())
        name_column = ml_name_column if ml_name_column in column2states \
            else get_personalized_feature_name(name_column, get_default_mp_method())

    itol_result = None
    new_tree = os.path.join(work_dir, get_named_tree_file(tree))
    features = [DATE, DATE_CI] + list(column2states.keys())
    clear_extra_features(roots, features)
    nwks = '\n'.join([root.write(format_root_node=True, format=3, features=features) for root in roots])
    with open(new_tree, 'w+') as f:
        f.write(nwks)
    try:
        nexus = new_tree.replace('.nwk', '.nexus')
        if '.nexus' not in nexus:
            nexus = '{}.nexus'.format(nexus)
        write(NewickIO.parse(StringIO(nwks)), nexus, 'nexus')
        with open(nexus, 'r') as f:
            nexus_str = f.read().replace('&&NHX:', '&')
            for feature in features:
                nexus_str = nexus_str.replace(':{}='.format(feature), ',{}='.format(feature))
        with open(nexus, 'w') as f:
            f.write(nexus_str)
    except Exception as e:
        logger.error(
            'Did not manage to save the annotated tree in nexus format due to the following error: {}'.format(e))
        pass

    if upload_to_itol or html or html_compressed:
        if colours:
            if isinstance(colours, str):
                colours = [colours]
            if isinstance(colours, list):
                colours = dict(zip(columns, colours))
            elif isinstance(colours, dict):
                colours = {col_name2cat(col): cls for (col, cls) in colours.items()}
            else:
                raise ValueError('Colours should be either a list or a dict, got {}.'.format(type(colours)))
        else:
            colours = {}

    if threads > 1:
        pool = ThreadPool(processes=threads - 1)
        async_result = pool.map_async(func=_serialize_acr, iterable=((acr_res, work_dir) for acr_res in acr_results))
        if upload_to_itol:
            if DATE_LABEL == age_label:
                try:
                    dates = state_df[DATE].apply(lambda _: numeric2datetime(_).strftime("%d %b %Y"))
                    state_df[DATE] = dates
                except:
                    pass
            itol_result = pool.apply_async(func=generate_itol_annotations,
                                           args=(column2states, work_dir, acr_results, state_df, age_label,
                                                 new_tree, itol_id, itol_project, itol_tree_name, colours))
    else:
        for acr_res in acr_results:
            _serialize_acr((acr_res, work_dir))
        if upload_to_itol:
            if DATE_LABEL == age_label:
                try:
                    dates = state_df[DATE].apply(lambda _: numeric2datetime(_).strftime("%d %b %Y"))
                    state_df[DATE] = dates
                except:
                    pass
            generate_itol_annotations(column2states, work_dir, acr_results, state_df, age_label,
                                      new_tree, itol_id, itol_project, itol_tree_name, colours)

    if html or html_compressed or html_mixed:
        logger.debug('\n=============VISUALISATION=====================')

        if (html_compressed or html_mixed) and focus:
            def parse_col_val(cv):
                cv = str(cv).strip()
                colon_pos = cv.find(':')
                if colon_pos == -1:
                    if len(column2states) == 1 and cv in next(iter(column2states.values())):
                        return next(iter(column2states.keys())), cv
                    else:
                        raise ValueError('Focus values should be in a form character:state, got {} instead.'.format(cv))
                col, state = col_name2cat(cv[:colon_pos]), cv[colon_pos + 1:]
                if col not in column2states:
                    ml_col = get_personalized_feature_name(col, get_default_ml_method())
                    if ml_col in column2states:
                        col = ml_col
                    else:
                        mp_col = get_personalized_feature_name(col, get_default_mp_method())
                        if mp_col in column2states:
                            col = mp_col
                        else:
                            raise ValueError('Character {} specified for focus values is not found in metadata.'.format(
                                cv[:colon_pos]))
                if state not in column2states[col]:
                    raise ValueError(
                        'Character {} state {} not found among possible states in metadata.'.format(cv[:colon_pos],
                                                                                                    state))
                return col, state

            if isinstance(focus, str):
                focus = list(focus)
            if not isinstance(focus, list):
                raise ValueError(
                    'Focus argument should be either a string or a list of strings, got {} instead.'.format(
                        type(focus)))
            focus_cv = [parse_col_val(_) for _ in focus]
            focus = defaultdict(set)
            for c, v in focus_cv:
                focus[c].add(v)

        visualize(roots, column2states=column2states, html=html, html_compressed=html_compressed, html_mixed=html_mixed,
                  name_column=name_column, tip_size_threshold=tip_size_threshold, date_label=age_label,
                  timeline_type=timeline_type, work_dir=work_dir, local_css_js=offline, column2colours=colours,
                  focus=focus, pajek=pajek, pajek_timing=pajek_timing)

    if threads > 1:
        async_result.wait()
        if upload_to_itol:
            itol_result.wait()
        pool.close()


def _validate_input(tree_nwk, columns=None, name_column=None, data=None, data_sep='\t', id_index=0,
                    root_dates=None, copy_only=False, parameters=None, rates=None):
    logger = logging.getLogger('pastml')
    logger.debug('\n=============INPUT DATA VALIDATION=============')

    if not columns and data is None:
        raise ValueError("If you don't provide the metadata file, "
                         "you need to provide an annotated tree and specify the columns argument, "
                         "which will be used to look for character annotations in your input tree.")

    if columns and isinstance(columns, str):
        columns = [columns]

    roots = read_forest(tree_nwk, columns=columns if data is None else None)

    column2annotated = Counter()
    column2states = defaultdict(set)

    if data:
        df = pd.read_csv(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
        df.index = df.index.map(str)
        logger.debug('Read the annotation file {}.'.format(data))
        if columns:
            unknown_columns = set(columns) - set(df.columns)
            if unknown_columns:
                raise ValueError('{} of the specified columns ({}) {} not found among the annotation columns: {}.'
                                 .format('One' if len(unknown_columns) == 1 else 'Some',
                                         _quote(unknown_columns),
                                         'is' if len(unknown_columns) == 1 else 'are',
                                         _quote(df.columns)))
            df = df[columns]
        df.columns = [col_name2cat(column) for column in df.columns]
        if name_column:
            name_column = col_name2cat(name_column)
        columns = df.columns

        node_names = set.union(*[{n.name for n in root.traverse() if n.name} for root in roots])
        df_index_names = set(df.index)
        common_ids = node_names & df_index_names

        # strip quotes if needed
        if not common_ids:
            node_names = {_.strip("'").strip('"') for _ in node_names}
            common_ids = node_names & df_index_names
            if common_ids:
                for root in roots:
                    for n in root.traverse():
                        n.name = n.name.strip("'").strip('"')

        filtered_df = df.loc[list(common_ids), :]
        if not filtered_df.shape[0]:
            tip_name_representatives = []
            for _ in roots[0].iter_leaves():
                if len(tip_name_representatives) < 3:
                    tip_name_representatives.append(_.name)
                else:
                    break
            raise ValueError(
                'Your tree tip names (e.g. {}) do not correspond to annotation id column values (e.g. {}). '
                'Check your annotation file.'
                    .format(', '.join(tip_name_representatives),
                            ', '.join(list(df_index_names)[: min(len(df_index_names), 3)])))
        logger.debug('Checked that (at least some of) tip names correspond to annotation file index.')
        preannotate_forest(roots, df=df)
        for c in df.columns:
            column2states[c] |= {_ for _ in df[c].unique() if pd.notnull(_) and _ != ''}

    num_tips = 0

    column2annotated_states = defaultdict(set)
    for root in roots:
        for n in root.traverse():
            for c in columns:
                vs = getattr(n, c, set())
                column2states[c] |= vs
                column2annotated_states[c] |= vs
                if vs:
                    column2annotated[c] += 1
            if n.is_leaf():
                num_tips += 1

    if column2annotated:
        c, num_annotated = min(column2annotated.items(), key=lambda _: _[1])
    else:
        c, num_annotated = columns[0], 0
    percentage_unknown = (num_tips - num_annotated) / num_tips
    if percentage_unknown >= (.9 if not copy_only else 1):
        raise ValueError('{:.1f}% of tip annotations for character "{}" are unknown, '
                         'not enough data to infer ancestral states. '
                         '{}'
                         .format(percentage_unknown * 100, c,
                                 'Check your annotation file and if its ids correspond to the tree tip/node names.'
                                 if data
                                 else 'You tree file should contain character state annotations, '
                                      'otherwise consider specifying a metadata file.'))

    c, states = min(column2annotated_states.items(), key=lambda _: len(_[1]))
    if len(states) > num_tips * .75 and not copy_only:
        raise ValueError('Character "{}" has {} unique states annotated in this tree: {}, '
                         'which is too much to infer on a {} with only {} tips. '
                         'Make sure the character you are analysing is discrete, and if yes use a larger tree.'
                         .format(c, len(states), states, 'tree' if len(roots) == 1 else 'forest', num_tips))

    if name_column and name_column not in columns:
        raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                         .format(name_column, _quote(columns)))
    elif len(columns) == 1:
        name_column = columns[0]

    # Process root dates
    if root_dates is not None:
        root_dates = [parse_date(d) for d in (root_dates if isinstance(root_dates, list) else [root_dates])]
        if 1 < len(root_dates) < len(roots):
            raise ValueError('{} trees are given, but only {} root dates.'.format(len(roots), len(root_dates)))
        elif 1 == len(root_dates):
            root_dates *= len(roots)
    age_label = DIST_TO_ROOT_LABEL \
        if (root_dates is None and not next((True for root in roots if getattr(root, DATE, None) is not None), False)) \
        else DATE_LABEL
    annotate_dates(roots, root_dates=root_dates)
    logger.debug('Finished input validation.')

    column2states = {c: np.array(sorted(states)) for c, states in column2states.items()}

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

    if rates:
        if isinstance(rates, str):
            rates = [rates]
        if isinstance(rates, list):
            rates = dict(zip(columns, rates))
        elif isinstance(rates, dict):
            rates = {col_name2cat(col): rs for (col, rs) in rates.items()}
        else:
            raise ValueError('Rate matrices should be either a list or a dict, got {}.'.format(type(rates)))
    else:
        rates = {}

    for i, tree in enumerate(roots):
        name_tree(tree, suffix='' if len(roots) == 1 else '_{}'.format(i))

    return roots, columns, column2states, name_column, age_label, parameters, rates


def _serialize_predicted_states(columns, out_data, roots, dates_are_dates=True):
    ids, data = [], []
    # Not using DataFrames to speed up document writing
    with open(out_data, 'w+') as f:
        f.write('node\t{}\n'.format('\t'.join(columns)))
        for root in roots:
            for node in root.traverse():
                vs = [node.dist, get_formatted_date(node, dates_are_dates)]
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
    return pd.DataFrame(index=ids, data=data, columns=['dist', DATE] + columns)


def main():
    """
    Entry point, calling :py:func:`pastml.acr.pastml_pipeline` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ancestral character reconstruction and visualisation "
                                                 "for rooted phylogenetic trees.", prog='pastml')

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
    acr_group.add_argument('-c', '--columns', nargs='*',
                           help="names of the annotation table columns that contain characters "
                                "to be analysed. "
                                "If not specified, all columns are considered.",
                           type=str)
    acr_group.add_argument('--prediction_method',
                           choices=[MPPA, MAP, JOINT, DOWNPASS, ACCTRAN, DELTRAN, COPY, ALL, ML, MP],
                           type=str, nargs='*', default=MPPA,
                           help='ancestral character reconstruction (ACR) method, '
                                'can be one of the max likelihood (ML) methods: {ml}, '
                                'one of the max parsimony (MP) methods: {mp}; '
                                'or {copy} to keep the annotated character states as-is without inference. '
                                'One can also specify one of the meta-methods {meta} that would perform ACR '
                                'with multiple methods (all of them for {meta_all}, '
                                'all the ML methods for {meta_ml}, or all the MP methods for {meta_mp}) '
                                'and save/visualise the results as multiple characters '
                                'suffixed with the corresponding method.'
                                'When multiple ancestral characters are specified (see -c, --columns), '
                                'the same method can be used for all of them (if only one method is specified), '
                                'or different methods can be used (specified in the same order as -c, --columns). '
                                'If multiple methods are given, but not for all the characters, '
                                'for the rest of them the default method ({default}) is chosen.'
                           .format(ml=', '.join(ML_METHODS), mp=', '.join(MP_METHODS), copy=COPY, default=MPPA,
                                   meta=', '.join(META_ML_METHODS | {MP}), meta_ml=ML, meta_mp=MP, meta_all=ALL))
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
                           help='optional way to fix some of the ML-method parameters '
                                'by specifying files that contain them. '
                                'Should be in the same order '
                                'as the ancestral characters (see -c, --columns) '
                                'for which the reconstruction is to be preformed. '
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
                                'as the ancestral characters (see -c, --columns) '
                                'for which the reconstruction is to be preformed. '
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

    vis_group = parser.add_argument_group('visualisation-related arguments')
    vis_group.add_argument('-n', '--name_column', type=str, default=None,
                           help="name of the character to be used for node names "
                                "in the compressed map visualisation "
                                "(must be one of those specified via -c, --columns). "
                                "If the annotation table contains only one column it will be used by default.")
    vis_group.add_argument('--root_date', required=False, default=None,
                           help="date(s) of the root(s) (for dated tree(s) only), "
                                "if specified, used to visualise a timeline based on dates "
                                "(otherwise it is based on distances to root).",
                           type=str, nargs='*')
    vis_group.add_argument('--tip_size_threshold', type=int, default=REASONABLE_NUMBER_OF_TIPS,
                           help="recursively remove the tips of size less than threshold-th largest tip"
                                "from the compressed map (set to 1e10 to keep all tips). "
                                "The larger it is the less tips will be trimmed.")
    vis_group.add_argument('--timeline_type', type=str, default=TIMELINE_SAMPLED,
                           help="type of timeline visualisation: at each date/distance to root selected on the slider "
                                "either ({sampled}) - all the lineages sampled after it are hidden; "
                                "or ({nodes}) - all the nodes with a more recent date/larger distance to root are hidden; "
                                "or ({ltt}) - all the nodes whose branch started after this date/distance to root "
                                "are hidden, and the external branches are cut to the specified date/distance to root "
                                "if needed;".format(sampled=TIMELINE_SAMPLED, ltt=TIMELINE_LTT, nodes=TIMELINE_NODES),
                           choices=[TIMELINE_SAMPLED, TIMELINE_NODES, TIMELINE_LTT])
    vis_group.add_argument('--offline', action='store_true',
                           help="By default (without --offline option) PastML assumes "
                                "that there is an internet connection available, "
                                "which permits it to fetch CSS and JS scripts needed for visualisation online."
                                "With --offline option turned on, PastML will store all the needed CSS/JS scripts "
                                "in the folder specified by --work_dir, so that internet connection is not needed "
                                "(but you must not move the output html files to any location "
                                "other that the one specified by --html/--html_compressed).")
    vis_group.add_argument('--colours', type=str, nargs='*',
                           help='optional way to specify the colours used for character state visualisation. '
                                'Should be in the same order '
                                'as the ancestral characters (see -c, --columns) '
                                'for which the reconstruction is to be preformed. '
                                'Could be given only for the first few characters. '
                                'Each file should be tab-delimited, with two columns: '
                                'the first one containing character states, '
                                'and the second, named "colour", containing colours, in HEX format (e.g. #a6cee3).')
    vis_group.add_argument('--focus', type=str, nargs='*',
                           help='optional way to put a focus on certain character state values, '
                                'so that the nodes in these states are displayed '
                                'even if they do not pass the trimming threshold (--tip_size_threshold). '
                                'Should be in the form character:state.')
    vis_group.add_argument('--resolve_polytomies', action='store_true',
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
                                "Default is <path_to_input_file>/<input_file_name>_pastml. "
                                "If the folder does not exist, it will be created."
                           .format(', '.join(MARGINAL_ML_METHODS)))
    out_group.add_argument('--html_compressed', required=False, default=None, type=str,
                           help="path to the output compressed map visualisation file (html).")
    out_group.add_argument('--pajek', required=False, default=None, type=str,
                           help="path to the output vertically compressed visualisation file (Pajek NET Format). "
                                "Prooduced only if --html_compressed is specified.")
    out_group.add_argument('--pajek_timing', required=False, default=VERTICAL, choices=(VERTICAL, HORIZONTAL, TRIM),
                           type=str,
                           help="the type of the compressed visualisation to be saved in Pajek NET Format "
                                "(if --pajek is specified). "
                                "Can be either {} (default, after the nodes underwent vertical compression), "
                                "{} (after the nodes underwent vertical and horizontal compression) "
                                "or {} (after the nodes underwent vertical and horizontal compression"
                                " and minor node trimming)".format(VERTICAL, HORIZONTAL, TRIM))
    out_group.add_argument('--html', required=False, default=None, type=str,
                           help="path to the output full tree visualisation file (html).")
    out_group.add_argument('--html_mixed', required=False, default=None, type=str,
                           help="path to the output mostly compressed map visualisation file (html), "
                                "where the nodes in states specified with --focus are uncompressed.")
    out_group.add_argument('-v', '--verbose', action='store_true',
                           help="print information on the progress of the analysis (to console)")

    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=PASTML_VERSION))

    parser.add_argument('--threads', required=False, default=0, type=int,
                        help="Number of threads PastML can use for parallesation. "
                             "By default detected automatically based on the system. "
                             "Note that PastML will at most use as many threads "
                             "as the number of characters (-c option) being analysed plus one.")
    parser.add_argument('--recursion_limit', required=False, default=0, type=int,
                        help='Set the recursion limit in python. '
                             'If your tree is very large and pastml produces an overflow error, '
                             'you could try to set the recursion limit to a large number (e.g., 10000). '
                             'By default the standard recursion limit is used and should be sufficient '
                             'for trees of up to several thousand tips.')

    itol_group = parser.add_argument_group('iTOL-related arguments')
    itol_group.add_argument('--upload_to_itol', action='store_true',
                            help="create iTOL annotations for the reconstructed characters "
                                 "associated with the named tree (i.e. the one found in --work_dir). "
                                 "If additionally --itol_id and --itol_project are specified, "
                                 "the annotated tree will be automatically uploaded to iTOL (https://itol.embl.de/).")
    itol_group.add_argument('--itol_id', required=False, default=None, type=str,
                            help="iTOL user batch upload ID that enables uploading to your iTOL account "
                                 "(see https://itol.embl.de/help.cgi#batch).")
    itol_group.add_argument('--itol_project', required=False, default="Sample project", type=str,
                            help="iTOL project the annotated tree should be associated with "
                                 "(must exist, and --itol_id must be specified). By default set to 'Sample project'.")
    itol_group.add_argument('--itol_tree_name', required=False, default=None, type=str,
                            help="name for the tree uploaded to iTOL.")

    params = parser.parse_args()

    pastml_pipeline(**vars(params))


if '__main__' == __name__:
    main()
