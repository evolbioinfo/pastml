import logging
import os
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd
from ete3 import Tree

from pastml import col_name2cat, value2list, STATES, METHOD, CHARACTER, get_personalized_feature_name, NUM_SCENARIOS
from pastml.annotation import preannotate_forest, get_forest_stats
from pastml.file import get_combined_ancestral_state_file, get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file, get_pastml_work_dir
from pastml.ml import SCALING_FACTOR, MODEL, FREQUENCIES, MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, ml_acr, \
    ML_METHODS, MAP, JOINT, ALL, ML, META_ML_METHODS, MARGINAL_ML_METHODS, get_default_ml_method
from pastml.models.f81_like import F81, JC, EFT
from pastml.models.hky import KAPPA, HKY_STATES, HKY
from pastml.models.jtt import JTT_STATES, JTT
from pastml.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS, MP, \
    get_default_mp_method
from pastml.tree import name_tree, collapse_zero_branches, annotate_dates, DATE, read_forest
from pastml.visualisation.cytoscape_manager import visualize, TIMELINE_SAMPLED, TIMELINE_NODES, TIMELINE_LTT
from pastml.visualisation.itol_manager import generate_itol_annotations
from pastml.visualisation.tree_compressor import REASONABLE_NUMBER_OF_TIPS

PASTML_VERSION = '1.9.19'

warnings.filterwarnings("ignore", append=True)

COPY = 'COPY'


def datetime2numeric(d):
    """
    Converts a datetime date to numeric format.
    For example: 2016-12-31 -> 2016.9972677595629; 2016-1-1 -> 2016.0
    :param d: a date to be converted
    :type d: np.datetime
    :return: numeric representation of the date
    :rtype: float
    """
    first_jan_this_year = pd.datetime(year=d.year, month=1, day=1)
    day_of_this_year = d - first_jan_this_year
    first_jan_next_year = pd.datetime(year=d.year + 1, month=1, day=1)
    days_in_this_year = first_jan_next_year - first_jan_this_year
    return d.year + day_of_this_year / days_in_this_year


def _parse_pastml_parameters(params, states):
    logger = logging.getLogger('pastml')
    frequencies, sf, kappa = None, None, None
    if not isinstance(params, str) and not isinstance(params, dict):
        raise ValueError('Parameters must be specified either as a dict or as a path to a csv file, not as {}!'
                         .format(type(params)))
    if isinstance(params, str):
        if not os.path.exists(params):
            raise ValueError('The specified parameter file ({}) does not exist.'
                             .format(params))
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
            raise ValueError('The specified parameter file {} is malformatted, '
                             'should be a tab-delimited file with two columns, '
                             'the first one containing parameter names, '
                             'and the second, named "value", containing parameter values.'.format(params))
    params = {str(k.encode('ASCII', 'replace').decode()): v for (k, v) in params.items()}
    frequencies_specified = set(states) & set(params.keys())
    if frequencies_specified:
        if len(frequencies_specified) < len(states):
            logger.error('Some frequency parameters are specified, but missing the following states: {}, '
                         'ignoring specified frequencies.'.format(', '.join(set(states) - frequencies_specified)))
        else:
            frequencies = np.array([params[state] for state in states])
            try:
                frequencies = frequencies.astype(np.float64)
                if np.round(frequencies.sum() - 1, 3) != 0:
                    logger.error('Specified frequencies ({}) do not sum up to one,'
                                 'ignoring them.'.format(frequencies))
                    frequencies = None
                elif np.any(frequencies < 0):
                    logger.error('Some of the specified frequencies ({}) are negative,'
                                 'ignoring them.'.format(frequencies))
                    frequencies = None
                frequencies /= frequencies.sum()
            except:
                logger.error('Specified frequencies ({}) must not be negative, ignoring them.'.format(frequencies))
                frequencies = None
    if SCALING_FACTOR in params:
        sf = params[SCALING_FACTOR]
        try:
            sf = np.float64(sf)
            if sf <= 0:
                logger.error(
                    'Scaling factor ({}) cannot be negative, ignoring it.'.format(sf))
                sf = None
        except:
            logger.error('Scaling factor ({}) is not float, ignoring it.'.format(sf))
            sf = None
    if KAPPA in params:
        kappa = params[KAPPA]
        try:
            kappa = np.float64(kappa)
            if kappa <= 0:
                logger.error(
                    'Kappa ({}) cannot be negative, ignoring it.'.format(kappa))
                kappa = None
        except:
            logger.error('Kappa ({}) is not float, ignoring it.'.format(kappa))
            kappa = None
    return frequencies, sf, kappa


def _serialize_acr(args):
    acr_result, work_dir = args
    out_param_file = \
        os.path.join(work_dir,
                     get_pastml_parameter_file(method=acr_result[METHOD],
                                               model=acr_result[MODEL] if MODEL in acr_result else None,
                                               column=acr_result[CHARACTER]))

    # Not using DataFrames to speed up document writing
    with open(out_param_file, 'w+') as f:
        f.write('parameter\tvalue\n')
        f.write('pastml_version\t{}\n'.format(PASTML_VERSION))
        for name in sorted(acr_result.keys()):
            if name not in [FREQUENCIES, STATES, MARGINAL_PROBABILITIES]:
                if NUM_SCENARIOS == name:
                    f.write('{}\t{:g}\n'.format(name, acr_result[name]))
                else:
                    f.write('{}\t{}\n'.format(name, acr_result[name]))
        if is_ml(acr_result[METHOD]):
            for state, freq in zip(acr_result[STATES], acr_result[FREQUENCIES]):
                f.write('{}\t{}\n'.format(state, freq))
    logging.getLogger('pastml').debug('Serialized ACR parameters and statistics for {} to {}.'
                                      .format(acr_result[CHARACTER], out_param_file))

    if is_marginal(acr_result[METHOD]):
        out_mp_file = \
            os.path.join(work_dir,
                         get_pastml_marginal_prob_file(method=acr_result[METHOD], model=acr_result[MODEL],
                                                       column=acr_result[CHARACTER]))
        acr_result[MARGINAL_PROBABILITIES].to_csv(out_mp_file, sep='\t', index_label='node')
        logging.getLogger('pastml').debug('Serialized marginal probabilities for {} to {}.'
                                          .format(acr_result[CHARACTER], out_mp_file))


def reconstruct_ancestral_states(forest, character, states, prediction_method=MPPA, model=F81,
                                 params=None, avg_br_len=None, num_nodes=None, num_tips=None,
                                 force_joint=True):
    """
    Reconstructs ancestral states for the given character on the given tree.

    :param character: character whose ancestral states are to be reconstructed.
    :type character: str
    :param forest: trees whose ancestral states are to be reconstructed,
        annotated with the feature specified as `character` containing node states when known.
    :type forest: list(ete3.Tree)
    :param states: possible character states.
    :type states: numpy.array
    :param avg_br_len: (optional) average non-zero branch length for this tree. If not specified, will be calculated.
    :type avg_br_len: float
    :param model: (optional, default is F81) state evolution model to be used by PASTML.
    :type model: str
    :param prediction_method: (optional, default is MPPA) ancestral state prediction method to be used by PASTML.
    :type prediction_method: str
    :param num_nodes: (optional) total number of nodes in the given tree (including tips).
        If not specified, will be calculated.
    :type num_nodes: int
    :param num_tips: (optional) total number of tips in the given tree.
        If not specified, will be calculated.
    :type num_tips: int
    :param params: an optional way to fix some parameters,
        must be in a form {param: value},
        where param can be a state (then the value should specify its frequency between 0 and 1),
        or "scaling factor" (then the value should be the scaling factor for three branches,
        e.g. set to 1 to keep the original branches). Could also be in a form path_to_param_file.
        Only makes sense for ML methods.
    :type params: dict or str

    :return: ACR result dictionary whose values depend on the prediction method.
    :rtype: dict
    """

    logging.getLogger('pastml').debug('ACR settings for {}:\n\tMethod:\t{}{}.'
                                      .format(character, prediction_method,
                                              '\n\tModel:\t{}'.format(model)
                                              if model and is_ml(prediction_method) else ''))
    if COPY == prediction_method:
        return {CHARACTER: character, STATES: states, METHOD: prediction_method}
    if not num_nodes:
        num_nodes = sum(sum(1 for _ in tree.traverse()) for tree in forest)
    if not num_tips:
        num_tips = sum(len(tree) for tree in forest)
    if is_ml(prediction_method):
        if avg_br_len is None:
            dists = []
            for tree in forest:
                dists.extend(n.dist for n in tree.traverse() if n.dist)
            avg_br_len = np.mean(dists)
        freqs, sf, kappa = None, None, None
        if params is not None:
            freqs, sf, kappa = _parse_pastml_parameters(params, states)
        return ml_acr(forest=forest, character=character, prediction_method=prediction_method, model=model, states=states,
                      avg_br_len=avg_br_len, num_nodes=num_nodes, num_tips=num_tips, freqs=freqs, sf=sf, kappa=kappa,
                      force_joint=force_joint)
    if is_parsimonious(prediction_method):
        return parsimonious_acr(forest, character, prediction_method, states, num_nodes, num_tips)

    raise ValueError('Method {} is unknown, should be one of ML ({}), one of MP ({}) or {}'
                     .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS), COPY))


def acr(forest, df, prediction_method=MPPA, model=F81, column2parameters=None, force_joint=True):
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
    :param force_joint: (optional, default is True) whether the JOINT state should be added to the MPPA prediction
        even when not selected by the Brier score
    :type force_joint: bool

    :return: list of ACR result dictionaries, one per character.
    :rtype: list(dict)
    """
    for c in df.columns:
        df[c] = df[c].apply(lambda _: '' if pd.isna(_) else _.encode('ASCII', 'replace').decode())

    if isinstance(forest, Tree):
        forest = [forest]

    columns = preannotate_forest(df, forest)
    for i, tree in enumerate(forest):
        name_tree(tree, suffix='' if len(forest) == 1 else '_{}'.format(i))
    collapse_zero_branches(forest, features_to_be_merged=df.columns)

    avg_br_len, num_nodes, num_tips = get_forest_stats(forest)

    logging.getLogger('pastml').debug('\n=============ACR===============================')

    column2parameters = column2parameters if column2parameters else {}

    def _work(args):
        return reconstruct_ancestral_states(*args, avg_br_len=avg_br_len, num_nodes=num_nodes, num_tips=num_tips,
                                            force_joint=force_joint)

    prediction_methods = value2list(len(columns), prediction_method, MPPA)
    models = value2list(len(columns), model, F81)

    def get_states(method, model, column):
        df_states = [_ for _ in df[column].unique() if pd.notnull(_) and _ != '']
        if not is_ml(method) or model not in {HKY, JTT}:
            return np.sort(df_states)
        states = HKY_STATES if HKY == model else JTT_STATES
        if not set(df_states) & set(states):
            raise ValueError('The allowed states for model {} are {}, '
                             'but your annotation file specifies {} as states in column {}.'
                             .format(model, ', '.join(states), ', '.join(df_states), column))
        state_set = set(states)
        df[column] = df[column].apply(lambda _: _ if _ in state_set else '')
        return states

    with ThreadPool() as pool:
        acr_results = \
            pool.map(func=_work, iterable=((forest, column, get_states(method, model, column), method, model,
                                            column2parameters[column] if column in column2parameters else None)
                                           for (column, method, model) in zip(columns, prediction_methods, models)))

    result = []
    for acr_res in acr_results:
        if isinstance(acr_res, list):
            result.extend(acr_res)
        else:
            result.append(acr_res)

    return result


def _quote(str_list):
    return ', '.join('"{}"'.format(_) for _ in str_list) if str_list is not None else ''


def pastml_pipeline(tree, data, data_sep='\t', id_index=0,
                    columns=None, prediction_method=MPPA, model=F81, parameters=None,
                    name_column=None, root_date=None, timeline_type=TIMELINE_SAMPLED,
                    tip_size_threshold=REASONABLE_NUMBER_OF_TIPS,
                    out_data=None, html_compressed=None, html=None, work_dir=None,
                    verbose=False, forced_joint=False, upload_to_itol=False, itol_id=None, itol_project=None,
                    itol_tree_name=None):
    """
    Applies PastML to the given tree(s) with the specified states and visualises the result (as html maps).

    :param tree: path to the input tree(s) in newick format (must be rooted).
    :type tree: str

    :param data: path to the annotation file in tab/csv format with the first row containing the column names.
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
        and tree branch scaling factor (parameter name pastml.ml.SCALING_FACTOR).
    :type parameters: str or list(str) or dict

    :param name_column: (optional) name of the annotation table column to be used for node names
        in the compressed map visualisation
        (must be one of those specified in ``columns``, if ``columns`` are specified).
        If the annotation table contains only one column, it will be used by default.
    :type name_column: str
    :param root_date: (optional) date(s) of the root(s) (for dated tree(s) only),
        if specified, used to visualise a timeline based on dates (otherwise it is based on distances to root).
    :type root_date: str or pandas.datetime or float
    :param tip_size_threshold: (optional, by default is 15) recursively remove the tips
        of size less than threshold-th largest tip from the compressed map (set to 1e10 to keep all).
        The larger it is the less tips will be trimmed.
    :type tip_size_threshold: int
    :param timeline_type: (optional, by default is pastml.visualisation.cytoscape_manager.TIMELINE_SAMPLED)
        type of timeline visualisation: at each date/distance to root selected on the slider, either
        (pastml.visualisation.cytoscape_manager.TIMELINE_SAMPLED) all the lineages sampled after it are hidden; "
        or (pastml.visualisation.cytoscape_manager.TIMELINE_NODES) all the nodes with a
        more recent date/larger distance to root are hidden;
        or (pastml.visualisation.cytoscape_manager.TIMELINE_LTT) all the nodes whose branch started
        after this date/distance to root are hidden, and the external branches are cut
        to the specified date/distance to root if needed;
    :type timeline_type: str

    :param out_data: path to the output annotation file with the reconstructed ancestral character states.
    :type out_data: str
    :param html_compressed: path to the output compressed visualisation file (html).
    :type html_compressed: str
    :param html: (optional) path to the output tree visualisation file (html).
    :type html: str
    :param work_dir: (optional) path to the folder where pastml parameter, named tree
        and marginal probability (for marginal ML methods (pastml.ml.MPPA, pastml.ml.MAP) only) files are to be stored.
        Default is <path_to_input_file>/<input_file_name>_pastml. If the folder does not exist, it will be created.
    :type work_dir: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :param upload_to_itol: (optional, default is False) whether the annotated tree should be uploaded to iTOL
        (https://itol.embl.de/)
    :type upload_to_itol: bool
    :param itol_id: (optional) iTOL user batch upload ID that enables uploading to your iTOL account
        (see https://itol.embl.de/help.cgi#batch). If not specified, the tree will not be associated to any account.
    :type itol_id: str
    :param itol_project: (optional) iTOL project the annotated tree should be uploaded to
        (must exist, and itol_id must be specified). If not specified, the tree will not be associated to any project.
    :type itol_project: str
    :param itol_tree_name: (optional) name for the tree uploaded to iTOL.
    :type itol_tree_name: str

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    age_label = 'Dist. to root' if root_date is None else 'Date'

    roots, df, name_column, root_dates = \
        _validate_input(columns, data, data_sep, root_date if html_compressed or html or upload_to_itol else None,
                        id_index, name_column if html_compressed else None, tree,
                        copy_only=COPY == prediction_method or (isinstance(prediction_method, list)
                                                                and all(COPY == _ for _ in prediction_method)))
    annotate_dates(roots, root_dates=root_dates)

    if parameters:
        if isinstance(parameters, str):
            parameters = [parameters]
        if isinstance(parameters, list):
            parameters = dict(zip(df.columns, parameters))
        elif isinstance(parameters, dict):
            parameters = {col_name2cat(col): params for (col, params) in parameters.items()}
        else:
            raise ValueError('Parameters should be either a list or a dict, got {}.'.format(type(parameters)))
    else:
        parameters = {}

    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)

    acr_results = acr(roots, df, prediction_method=prediction_method, model=model, column2parameters=parameters,
                      force_joint=forced_joint)
    column2states = {acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results}

    if not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file())

    state_df = _serialize_predicted_states(sorted(column2states.keys()), out_data, roots)

    # a meta-method would have added a suffix to the name feature
    if html_compressed and name_column and name_column not in column2states:
        ml_name_column = get_personalized_feature_name(name_column, get_default_ml_method())
        name_column = ml_name_column if ml_name_column in column2states \
            else get_personalized_feature_name(name_column, get_default_mp_method())


    itol_result = None
    pool = ThreadPool()
    new_tree = os.path.join(work_dir, get_named_tree_file(tree))
    nwks = [root.write(format_root_node=True, format=3) for root in roots]
    with open(new_tree, 'w+') as f:
        f.write('\n'.join(nwks))
    async_result = pool.map_async(func=_serialize_acr, iterable=((acr_res, work_dir) for acr_res in acr_results))
    if upload_to_itol:
        itol_result = pool.apply_async(func=generate_itol_annotations,
                                       args=(column2states, work_dir, acr_results, state_df, age_label,
                                             new_tree, itol_id, itol_project, itol_tree_name))

    if html or html_compressed:
        logger.debug('\n=============VISUALISATION=====================')
        visualize(roots, column2states=column2states, html=html, html_compressed=html_compressed,
                  name_column=name_column, tip_size_threshold=tip_size_threshold, date_label=age_label,
                  timeline_type=timeline_type)

    async_result.wait()
    if itol_result:
        itol_result.wait()
    pool.close()


def parse_date(d):
    try:
        return float(d)
    except ValueError:
        try:
            return datetime2numeric(pd.to_datetime(d, infer_datetime_format=True))
        except ValueError:
            raise ValueError('Could not infer the date format for root date "{}", please check it.'
                             .format(d))


def _validate_input(columns, data, data_sep, root_dates, id_index, name_column, tree_nwk,
                    copy_only):
    logger = logging.getLogger('pastml')
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    roots = read_forest(tree_nwk)
    num_neg = 0
    for root in roots:
        for _ in root.traverse():
            if _.dist < 0:
                num_neg += 1
                _.dist = 0
    if num_neg:
        logger.warning('Input tree{} contained {} negative branches: we put them to zero.'
                       .format('s' if len(roots) > 0 else '', num_neg))
    logger.debug('Read the tree{} {}.'.format('s' if len(roots) > 0 else '', tree_nwk))

    df = pd.read_csv(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
    df.index = df.index.map(str)
    logger.debug('Read the annotation file {}.'.format(data))

    # As the date column is only used for visualisation if there is no visualisation we are not gonna validate it
    if root_dates is not None:
        root_dates = [parse_date(d) for d in (root_dates if isinstance(root_dates, list) else [root_dates])]
        if 1 < len(root_dates) < len(roots):
            raise ValueError('{} trees are given, but only {} root dates.'.format(len(roots), len(root_dates)))
        elif 1 == len(root_dates):
            root_dates *= len(roots)

    if columns:
        if isinstance(columns, str):
            columns = [columns]
        unknown_columns = set(columns) - set(df.columns)
        if unknown_columns:
            raise ValueError('{} of the specified columns ({}) {} not found among the annotation columns: {}.'
                             .format('One' if len(unknown_columns) == 1 else 'Some',
                                     _quote(unknown_columns),
                                     'is' if len(unknown_columns) == 1 else 'are',
                                     _quote(df.columns)))
        df = df[columns]

    df.columns = [col_name2cat(column) for column in df.columns]

    node_names = set.union(*[{n.name for n in root.traverse() if n.name} for root in roots])
    df_index_names = set(df.index)
    filtered_df = df.loc[node_names & df_index_names, :]
    if not filtered_df.shape[0]:
        tip_name_representatives = []
        for _ in roots[0].iter_leaves():
            if len(tip_name_representatives) < 3:
                tip_name_representatives.append(_.name)
            else:
                break
        raise ValueError('Your tree tip names (e.g. {}) do not correspond to annotation id column values (e.g. {}). '
                         'Check your annotation file.'
                         .format(', '.join(tip_name_representatives),
                                 ', '.join(list(df_index_names)[: min(len(df_index_names), 3)])))
    logger.debug('Checked that tip names correspond to annotation file index.')

    if name_column:
        name_column = col_name2cat(name_column)
        if name_column not in df.columns:
            raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                             .format(name_column, _quote(df.columns)))
    elif len(df.columns) == 1:
        name_column = df.columns[0]

    percentage_unknown = filtered_df.isnull().sum(axis=0) / filtered_df.shape[0]
    max_unknown_percentage = percentage_unknown.max()
    if max_unknown_percentage >= (.9 if not copy_only else 1):
        raise ValueError('{:.1f}% of tip annotations for column "{}" are unknown, '
                         'not enough data to infer ancestral states. '
                         'Check your annotation file and if its id column corresponds to the tree tip names.'
                         .format(max_unknown_percentage * 100, percentage_unknown.idxmax()))
    percentage_unique = filtered_df.nunique() / filtered_df.count()
    max_unique_percentage = percentage_unique.max()
    if filtered_df.count()[0] > 100 and max_unique_percentage > .5:
        raise ValueError('The column "{}" seem to contain non-categorical data: {:.1f}% of values are unique. '
                         'PASTML cannot infer ancestral states for a tree with too many tip states.'
                         .format(percentage_unique.idxmax(), 100 * max_unique_percentage))
    logger.debug('Finished input validation.')
    return roots, df, name_column, root_dates


def _serialize_predicted_states(columns, out_data, roots):
    ids, data = [], []
    # Not using DataFrames to speed up document writing
    with open(out_data, 'w+') as f:
        f.write('node\t{}\n'.format('\t'.join(columns)))
        for root in roots:
            for node in root.traverse():
                vs = [node.dist, getattr(node, DATE)]
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


def _set_up_pastml_logger(verbose):
    logger = logging.getLogger('pastml')
    logger.setLevel(level=logging.DEBUG if verbose else logging.ERROR)
    logger.propagate = False
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(name)s:%(levelname)s:%(asctime)s %(message)s', datefmt="%H:%M:%S")
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


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
    annotation_group.add_argument('-d', '--data', required=True, type=str,
                                  help="annotation file in tab/csv format with the first row "
                                       "containing the column names.")
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
                           choices=[JC, F81, EFT, HKY, JTT],
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
                                'and tree branch scaling factor (parameter name {}).'.format(SCALING_FACTOR))

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

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_data', required=False, type=str,
                           help="path to the output annotation file with the reconstructed ancestral character states.")
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where pastml parameter, named tree "
                                "and marginal probability (for marginal ML methods ({}) only) files are to be stored. "
                                "Default is <path_to_input_file>/<input_file_name>_pastml. "
                                "If the folder does not exist, it will be created."
                           .format(', '.join(MARGINAL_ML_METHODS)))
    out_group.add_argument('-p', '--html_compressed', required=False, default=None, type=str,
                           help="path to the output compressed map visualisation file (html).")
    out_group.add_argument('-l', '--html', required=False, default=None, type=str,
                           help="path to the output full tree visualisation file (html).")
    out_group.add_argument('-v', '--verbose', action='store_true',
                           help="print information on the progress of the analysis (to console)")

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    itol_group = parser.add_argument_group('iTOL-related arguments')
    itol_group.add_argument('--upload_to_itol', action='store_true',
                            help="upload the ACR annotated tree to iTOL (https://itol.embl.de/)")
    itol_group.add_argument('--itol_id', required=False, default=None, type=str,
                            help="iTOL user batch upload ID that enables uploading to your iTOL account "
                                 "(see https://itol.embl.de/help.cgi#batch). "
                                 "If not specified, the tree will not be associated to any account.")
    itol_group.add_argument('--itol_project', required=False, default=None, type=str,
                            help="iTOL project the annotated tree should be associated with "
                                 "(must exist, and --itol_id must be specified). "
                                 "If not specified, the tree will not be associated with any project.")
    itol_group.add_argument('--itol_tree_name', required=False, default=None, type=str,
                            help="name for the tree uploaded to iTOL.")

    params = parser.parse_args()

    pastml_pipeline(**vars(params))


if '__main__' == __name__:
    main()
