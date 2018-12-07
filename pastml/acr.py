import logging
import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from pastml.models.f81_like import F81, JC, EFT
from pastml.ml import SCALING_FACTOR, MODEL, FREQUENCIES, MARGINAL_PROBABILITIES, is_ml, is_marginal, MPPA, ml_acr, \
    ML_METHODS, MAP, JOINT, ALL, ML, META_ML_METHODS, MARGINAL_ML_METHODS, get_default_ml_method
from pastml.models.hky import KAPPA, HKY_STATES, HKY
from pastml.models.jtt import JTT_STATES, JTT
from pastml import col_name2cat, value2list, STATES, METHOD, CHARACTER, get_personalized_feature_name
from pastml.annotation import preannotate_tree, get_tree_stats
from pastml.visualisation.cytoscape_manager import visualize
from pastml.file import get_combined_ancestral_state_file, get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file
from pastml.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS, MP, \
    get_default_mp_method
from pastml.tree import read_tree, name_tree, date_tips, collapse_zero_branches, DATE, annotate_depth, DEPTH
from pastml.visualisation.tree_compressor import REASONABLE_NUMBER_OF_TIPS

COPY = 'COPY'


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
            param_dict = pd.read_table(params, header=0, index_col=0)
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
    frequencies_specified = set(states) & set(params.keys())
    if frequencies_specified:
        if len(frequencies_specified) < len(states):
            logger.error('Frequency parameters are specified ({}), but not for all of the states ({}), '
                         'ignoring them.'.format(', '.join(sorted(frequencies_specified)), ', '.join(states)))
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
        for name in sorted(acr_result.keys()):
            if name not in [FREQUENCIES, STATES, MARGINAL_PROBABILITIES]:
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


def reconstruct_ancestral_states(tree, character, states, prediction_method=MPPA, model=F81,
                                 params=None, avg_br_len=None, num_nodes=None, num_tips=None,
                                 force_joint=True):
    """
    Reconstructs ancestral states for the given character on the given tree.

    :param character: character whose ancestral states are to be reconstructed.
    :type character: str
    :param tree: tree whose ancestral state are to be reconstructed,
        annotated with the feature specified as `character` containing node states when known.
    :type tree: ete3.Tree
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
        num_nodes = sum(1 for _ in tree.traverse())
    if not num_tips:
        num_tips = len(tree)
    if is_ml(prediction_method):
        if avg_br_len is None:
            avg_br_len = np.mean(n.dist for n in tree.traverse() if n.dist)
        freqs, sf, kappa = None, None, None
        if params is not None:
            freqs, sf, kappa = _parse_pastml_parameters(params, states)
        return ml_acr(tree=tree, character=character, prediction_method=prediction_method, model=model, states=states,
                      avg_br_len=avg_br_len, num_nodes=num_nodes, num_tips=num_tips, freqs=freqs, sf=sf, kappa=kappa,
                      force_joint=force_joint)
    if is_parsimonious(prediction_method):
        return parsimonious_acr(tree, character, prediction_method, states, num_nodes, num_tips)

    raise ValueError('Method {} is unknown, should be one of ML ({}), one of MP ({}) or {}'
                     .format(prediction_method, ', '.join(ML_METHODS), ', '.join(MP_METHODS), COPY))


def acr(tree, df, prediction_method=MPPA, model=F81, column2parameters=None,
        force_joint=True):
    """
    Reconstructs ancestral states for the given tree and
    all the characters specified as columns of the given annotation dataframe.

    :param df: dataframe indexed with tree node names
        and containing characters for which ACR should be performed as columns.
    :type df: pandas.DataFrame
    :param tree: tree whose ancestral state are to be reconstructed.
    :type tree: ete3.Tree
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

    :return: list of ACR result dictionaries, one per character.
    :rtype: list(dict)
    """
    columns = preannotate_tree(df, tree)
    name_tree(tree)
    collapse_zero_branches(tree, features_to_be_merged=df.columns)

    avg_br_len, num_nodes, num_tips = get_tree_stats(tree)

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
            pool.map(func=_work, iterable=((tree, column, get_states(method, model, column), method, model,
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
                    name_column=None, date_column=None, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS,
                    out_data=None, html_compressed=None, html=None, work_dir=None,
                    verbose=False, no_forced_joint=False):
    """
    Applies PASTML to the given tree with the specified states and visualizes the result (as html maps).

    :param tree: path to the input tree in newick format (must be rooted).
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
    :param no_forced_joint: (optional, default is False) do not add JOINT state to the MPPA state selection
        when it is not selected by Brier score.
    :type no_forced_joint: bool
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
    :param date_column: (optional) name of the annotation table column that contains tip dates,
        if specified it is used to add a time slider to the visualisation.
    :type date_column: str
    :param tip_size_threshold: (optional, by default is 15) remove the tips of size less than threshold-th largest tip
        from the compressed map (set to 1e10 to keep all). The larger it is the less tips will be trimmed.
    :type tip_size_threshold: int

    :param out_data: path to the output annotation file with the reconstructed ancestral character states.
    :type out_data: str
    :param html_compressed: path to the output compressed visualisation file (html).
    :type html_compressed: str
    :param html: (optional) path to the output tree visualisation file (html).
    :type html: str
    :param work_dir: (optional) path to the folder where pastml parameter, named tree
        and marginal probability (for marginal ML methods (pastml.ml.MPPA, pastml.ml.MAP) only) files are to be stored.
        If the specified folder does not exist, it will be created.
    :type work_dir: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    if work_dir:
        os.makedirs(work_dir, exist_ok=True)

    root, df, years, name_column = _validate_input(columns, data, data_sep, date_column, html, html_compressed,
                                                   id_index, name_column, tree,
                                                   copy_only=COPY == prediction_method
                                                             or (isinstance(prediction_method, list)
                                                                 and all(COPY == _ for _ in prediction_method)))
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

    acr_results = acr(root, df, prediction_method=prediction_method, model=model, column2parameters=parameters,
                      force_joint=not no_forced_joint)
    column2states = {acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results}

    columns = sorted(column2states.keys())
    if work_dir and not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file(columns=columns))
    if out_data:
        _serialize_predicted_states(columns, out_data, root)

    # a meta-method would have added a suffix to the name feature
    if html_compressed and name_column and name_column not in column2states:
        ml_name_column = get_personalized_feature_name(name_column, get_default_ml_method())
        name_column = ml_name_column if ml_name_column in column2states \
            else get_personalized_feature_name(name_column, get_default_mp_method())

    async_result = None
    pool = None
    if work_dir:
        pool = ThreadPool()
        new_tree = os.path.join(work_dir, get_named_tree_file(tree))
        root.write(outfile=new_tree, format_root_node=True, format=3)
        async_result = pool.map_async(func=_serialize_acr, iterable=((acr_res, work_dir) for acr_res in acr_results))

    if html or html_compressed:
        logger.debug('\n=============VISUALISATION=====================')
        visualize(root, column2states=column2states,
                  html=html, html_compressed=html_compressed, years=years,
                  name_column=name_column, tip_size_threshold=tip_size_threshold)

    if async_result:
        async_result.wait()
        pool.close()
    return root


def _validate_input(columns, data, data_sep, date_column, html, html_compressed, id_index, name_column, tree_nwk,
                    copy_only):
    logger = logging.getLogger('pastml')
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    root = read_tree(tree_nwk)
    logger.debug('Read the tree {}{}.'.format(root.name, tree_nwk))

    df = pd.read_table(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
    logger.debug('Read the annotation file {}.'.format(data))

    # As the date column is only used for visualisation if there is no visualisation we are not gonna validate it
    years = []
    if html_compressed or html:
        min_date, max_date = None, None
        if date_column:
            if date_column not in df.columns:
                raise ValueError('The date column "{}" not found among the annotation columns: {}.'
                                 .format(date_column, _quote(df.columns)))
            try:
                df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)
            except ValueError:
                try:
                    df[date_column] = pd.to_datetime(df[date_column], format='%Y.0')
                except ValueError:
                    raise ValueError('Could not infer the date format for column "{}", please check it.'
                                     .format(date_column))
            min_date, max_date = date_tips(root, df[date_column])
            logger.debug("Extracted tip dates: they vary between {} and {}.".format(min_date, max_date))
        annotate_depth(root)
        if not date_column:
            for tip in root:
                date = getattr(tip, DEPTH)
                tip.add_feature(DATE, date)
                min_date = min(min_date, date) if min_date is not None else date
                max_date = max(max_date, date) if max_date is not None else date

        step = (max_date - min_date) / 5
        years = [min_date]
        while years[-1] < max_date:
            years.append(min(max_date, years[-1] + step))
        years = sorted(set(min(max_date, max(min_date, round(_, 4))) for _ in years))

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

    df.index = df.index.map(str)
    node_names = {n.name for n in root.traverse() if n.name}
    df_index_names = set(df.index)
    df = df.loc[node_names & df_index_names, :]
    if not df.shape[0]:
        tip_name_representatives = []
        for _ in root.iter_leaves():
            if len(tip_name_representatives) < 3:
                tip_name_representatives.append(_.name)
            else:
                break
        raise ValueError('Your tree tip names (e.g. {}) do not correspond to annotation id column values (e.g. {}). '
                         'Check your annotation file.'
                         .format(', '.join(tip_name_representatives),
                                 ', '.join(df_index_names[: min(len(df_index_names), 3)])))
    logger.debug('Checked that tip names correspond to annotation file index.')

    if columns:
        df = df[columns]
    df.columns = [col_name2cat(column) for column in df.columns]
    if html_compressed and name_column:
        name_column = col_name2cat(name_column)
        if name_column not in df.columns:
            raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                             .format(name_column, _quote(df.columns)))
    elif len(df.columns) == 1:
        name_column = df.columns[0]

    percentage_unknown = df.isnull().sum(axis=0) / df.shape[0]
    max_unknown_percentage = percentage_unknown.max()
    if max_unknown_percentage >= (.9 if not copy_only else 1):
        raise ValueError('{:.1f}% of tip annotations for column "{}" are unknown, '
                         'not enough data to infer ancestral states. '
                         'Check your annotation file and if its id column corresponds to the tree tip names.'
                         .format(max_unknown_percentage * 100, percentage_unknown.idxmax()))
    percentage_unique = df.nunique() / df.count()
    max_unique_percentage = percentage_unique.max()
    if df.count()[0] > 100 and max_unique_percentage > .5:
        raise ValueError('The column "{}" seem to contain non-categorical data: {:.1f}% of values are unique. '
                         'PASTML cannot infer ancestral states for a tree with too many tip states.'
                         .format(percentage_unique.idxmax(), 100 * max_unique_percentage))
    logger.debug('Finished input validation.')
    return root, df, years, name_column


def _serialize_predicted_states(columns, out_data, root):
    # Not using DataFrames to speed up document writing
    with open(out_data, 'w+') as f:
        f.write('node\t{}\n'.format('\t'.join(columns)))
        for node in root.traverse():
            column2values = {}
            for column in columns:
                value = getattr(node, column, set())
                if value:
                    column2values[column] = sorted(value, reverse=True)
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
    tree_group.add_argument('-t', '--tree', help="input tree in newick format (must be rooted).",
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
    acr_group.add_argument('--no_forced_joint', action='store_true',
                           help='do not add {joint} state to the {mppa} state selection '
                                'when it is not selected by Brier score.'.format(joint=JOINT, mppa=MPPA))
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
    vis_group.add_argument('--date_column', required=False, default=None,
                           help="name of the annotation table column that contains tip dates, "
                                "if specified it is used to add a time slider to the visualisation.",
                           type=str)
    vis_group.add_argument('--tip_size_threshold', type=int, default=REASONABLE_NUMBER_OF_TIPS,
                           help="remove the tips of size less than threshold-th largest tip"
                                "from the compressed map (set to 1e10 to keep all tips). "
                                "The larger it is the less tips will be trimmed.")

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_data', required=False, type=str,
                           help="path to the output annotation file with the reconstructed ancestral character states.")
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where pastml parameter, named tree "
                                "and marginal probability (for marginal ML methods ({}) only) files are to be stored. "
                                "If the specified folder does not exist, it will be created."
                           .format(', '.join(MARGINAL_ML_METHODS)))
    out_group.add_argument('-p', '--html_compressed', required=False, default=None, type=str,
                           help="path to the output compressed map visualisation file (html).")
    out_group.add_argument('-l', '--html', required=False, default=None, type=str,
                           help="path to the output full tree visualisation file (html).")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print information on the progress of the analysis")
    parser.add_argument('--version', action='version', version='%(prog)s 1.9')

    params = parser.parse_args()

    pastml_pipeline(**vars(params))


if '__main__' == __name__:
    main()
