import logging
import os
from multiprocessing.pool import ThreadPool

import numpy as np
import pandas as pd

from pastml import col_name2cat, value2list, STATES, METHOD, CHARACTER
from pastml.annotation import preannotate_tree, get_tree_stats
from pastml.cytoscape_manager import visualize
from pastml.file import get_combined_ancestral_state_file, get_named_tree_file, get_pastml_parameter_file, \
    get_pastml_marginal_prob_file
from pastml.ml import is_ml, ml_acr, MPPA, MAP, JOINT, F81, is_marginal, JC, EFT, FREQUENCIES, MARGINAL_PROBABILITIES, \
    SCALING_FACTOR, MODEL
from pastml.parsimony import is_parsimonious, parsimonious_acr, ACCTRAN, DELTRAN, DOWNPASS
from pastml.tree import read_tree, name_tree, date_tips, REASONABLE_NUMBER_OF_TIPS

COPY = 'COPY'


def _parse_pastml_parameters(params, states):
    logger = logging.getLogger('pastml')
    frequencies, sf = None, None
    if not isinstance(params, str) and not isinstance(params, dict):
        raise ValueError('Parameters must be specified either as a dict or as a path to a csv file, not as {}!'
                         .format(type(params)))
    if isinstance(params, str):
        if not os.path.exists(params):
            raise ValueError('You have specified some parameters ({}) but such a file does not exist!'
                             .format(params))
        try:
            param_dict = pd.read_csv(params, header=0, index_col=0)
            if 'value' not in param_dict.columns:
                logger.error('Could not find the "value" column in the parameter file {}. '
                             'It should be a csv file with two columns, the first one containing parameter names, '
                             'and the second, named "value", containing parameter values. '
                             'Ignoring these parameters.'.format(params))
                return frequencies, sf
            param_dict = param_dict.to_dict()['value']
            params = param_dict
        except:
            logger.error('The specified parameter file {} is malformatted, '
                         'should be a csv file with two columns, the first one containing parameter names, '
                         'and the second, named "value", containing parameter values. '
                         'Ignoring these parameters.'.format(params))
            return frequencies, sf
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
    return frequencies, sf


def _serialize_acr(args):
    acr_result, work_dir = args
    out_param_file = \
        os.path.join(work_dir,
                     get_pastml_parameter_file(method=acr_result[METHOD],
                                               model=acr_result[MODEL] if MODEL in acr_result else None,
                                               column=acr_result[CHARACTER]))

    # Not using DataFrames to speed up document writing
    with open(out_param_file, 'w+') as f:
        f.write('parameter,value\n')
        for name, value in acr_result.items():
            if name not in [FREQUENCIES, STATES, MARGINAL_PROBABILITIES]:
                f.write('{},{}\n'.format(name, value))
        if is_ml(acr_result[METHOD]):
            for state, freq in zip(acr_result[STATES], acr_result[FREQUENCIES]):
                f.write('{},{}\n'.format(state, freq))
    logging.getLogger('pastml').debug('Serialized ACR results for {} to {}.'
                                      .format(acr_result[CHARACTER], out_param_file))

    if is_marginal(acr_result[METHOD]):
        out_mp_file = \
            os.path.join(work_dir,
                         get_pastml_marginal_prob_file(method=acr_result[METHOD], model=acr_result[MODEL],
                                                       column=acr_result[CHARACTER]))
        acr_result[MARGINAL_PROBABILITIES].to_csv(out_mp_file, sep='\t', index_label='node')
        logging.getLogger('pastml').debug('Serialized marginal probabilities for {} to {}.'
                                          .format(acr_result[CHARACTER], out_mp_file))


def reconstruct_ancestral_states(tree, feature, states, avg_br_len=None, num_nodes=None, num_tips=None,
                                 prediction_method=MPPA, model=F81, params=None):
    """
    Reconstructs ancestral states for the given character on the given tree.

    :param feature: character whose ancestral states are to be reconstructed.
    :type feature: str
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
                                      .format(feature, prediction_method,
                                              '\n\tModel:\t{}'.format(model)
                                              if model and is_ml(prediction_method) else ''))
    if COPY == prediction_method:
        return {CHARACTER: feature, STATES: states, METHOD: prediction_method}
    if not num_nodes:
        num_nodes = sum(1 for _ in tree.traverse())
    if not num_tips:
        num_tips = len(tree)
    if is_ml(prediction_method):
        if avg_br_len is None:
            avg_br_len = np.mean(n.dist for n in tree.traverse() if n.dist)
        freqs, sf = None, None
        if params is not None:
            freqs, sf = _parse_pastml_parameters(params, states)
        return ml_acr(tree, feature, prediction_method, model, states, avg_br_len, num_nodes, num_tips, freqs, sf)
    if is_parsimonious(prediction_method):
        return parsimonious_acr(tree, feature, prediction_method, states, num_nodes, num_tips)

    raise ValueError('Method {} is unknown, should be one of ML ({}, {}, {}), one of MP ({}, {}, {}) or {}'
                     .format(prediction_method, MPPA, MAP, JOINT, ACCTRAN, DELTRAN, DOWNPASS, COPY))


def acr(tree, df, prediction_method=MPPA, model=F81, column2parameters=None):
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
        where param can be a state (then the value should specify its frequency between 0 and 1),
        or "scaling factor" (then the value should be the scaling factor for three branches,
        e.g. set to 1 to keep the original branches). Could also be in a form {column: path_to_param_file}.
    :type column2parameters: dict

    :return: list of ACR result dictionaries, one per character.
    :rtype: list(dict)
    """
    columns = preannotate_tree(df, tree)
    name_tree(tree)

    avg_br_len, num_nodes, num_tips = get_tree_stats(tree)

    logging.getLogger('pastml').debug('\n=============ACR===============================')

    column2parameters = column2parameters if column2parameters else {}

    def _work(args):
        return reconstruct_ancestral_states(*args)

    prediction_methods = value2list(len(columns), prediction_method, MPPA)
    models = value2list(len(columns), model, F81)

    with ThreadPool() as pool:
        acr_results = \
            pool.map(func=_work, iterable=((tree, column, np.sort([_ for _ in df[column].unique()
                                                                   if pd.notnull(_) and _ != '']),
                                            avg_br_len, num_nodes, num_tips, method, model,
                                            column2parameters[column] if column in column2parameters else None)
                                           for (column, method, model) in zip(columns, prediction_methods, models)))

    return acr_results


def _quote(str_list):
    return ', '.join('"{}"'.format(_) for _ in str_list) if str_list is not None else ''


def pastml_pipeline(tree, data, out_data=None, html_compressed=None, html=None, data_sep='\t', id_index=0, columns=None,
                    name_column=None, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS,
                    model=F81, prediction_method=MPPA, verbose=False, date_column=None, column2parameters=None,
                    work_dir=None):
    """
    Applies PASTML to the given tree with the specified states and visualizes the result (as html maps).

    :param date_column: (optional) name of the data table column that contains tip dates.
    :type date_column: str
    :param out_data: path to the output annotation file with the states inferred by PASTML.
    :type out_data: str
    :param tree: path to the input tree in newick format.
    :type tree: str
    :param data: path to the annotation file in tab/csv format with the first row containing the column names.
    :type data: str
    :param html_compressed: path to the output compressed visualisation file (html).
    :type html_compressed: str
    :param html: (optional) path to the output tree visualisation file (html).
    :type html: str
    :param data_sep: (optional, by default '\t') column separator for the data table.
        By default is set to tab, i.e. for tab-delimited file. Set it to ',' if your file is csv.
    :type data_sep: char
    :param id_index: (optional, by default is 0) the index of the column in the data table
        that contains the tree tip names, indices start from zero.
    :type id_index: int
    :param columns: (optional) names of the data table columns that contain states
        to be analysed with PASTML, if not specified all columns will be considered.
    :type columns: list
    :param name_column: (optional) name of the data table column to be used for node names in the visualisation
        (must be one of those specified in columns, if columns are specified).
        If the data table contains only one column, it will be used by default.
    :type name_column: str
    :param tip_size_threshold: (optional, by default is 15) remove the tips of size less than threshold-th
        from the compressed map (set to 1e10 to keep all). The larger it is the less tips will be trimmed.
    :type tip_size_threshold: int
    :param model: (optional, default is F81) model(s) to be used by PASTML,
        can be either one model to be used for all the characters,
        or a list of different models (in the same order as the annotation dataframe columns)
    :type model: str or list(str)
    :param prediction_method: (optional, default is MPPA) ancestral state prediction method(s) to be used by PASTML,
        can be either one method to be used for all the characters,
        or a list of different methods (in the same order as the annotation dataframe columns)
    :type prediction_method: str or list(str)
    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool
    :param column2parameters: an optional way to fix some parameters, must be in a form {column: {param: value}},
        where param can be a state (then the value should specify its frequency between 0 and 1),
        or "scaling factor" (then the value should be the scaling factor for three branches,
        e.g. set to 1 to keep the original branches). Could also be in a form {column: path_to_param_file}.
    :type column2parameters: dict
    :param work_dir: path to the folder where PASTML should put its files (e.g. estimated parameters, etc.).
        If the specified folder does not exist, it will be created.
    :type work_dir: str

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    if work_dir:
        os.makedirs(work_dir, exist_ok=True)

    root, df, max_date, min_date, name_column = _validate_input(columns, data, data_sep, date_column,
                                                                html, html_compressed, id_index, name_column, tree)

    column2parameters = {col_name2cat(col): params for (col, params) in column2parameters.items()} \
        if column2parameters else {}

    acr_results = acr(root, df, prediction_method=prediction_method, model=model, column2parameters=column2parameters)

    if work_dir and not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file(columns=df.columns))
    if out_data:
        _serialize_predicted_states(df.columns, out_data, root)

    async_result = None
    pool = None
    if work_dir:
        pool = ThreadPool()
        new_tree = os.path.join(work_dir, get_named_tree_file(tree))
        root.write(outfile=new_tree, format_root_node=True, format=3)
        async_result = pool.map_async(func=_serialize_acr, iterable=((acr_res, work_dir) for acr_res in acr_results))

    if html or html_compressed:
        logger.debug('\n=============VISUALISATION=====================')
        visualize(root, column2states={acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results},
                  html=html, html_compressed=html_compressed, min_date=min_date, max_date=max_date,
                  name_column=name_column, tip_size_threshold=tip_size_threshold)

    if async_result:
        async_result.wait()
        pool.close()
    return root


def _validate_input(columns, data, data_sep, date_column, html, html_compressed, id_index, name_column, tree_nwk):
    logger = logging.getLogger('pastml')
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    root = read_tree(tree_nwk)
    logger.debug('Read the tree {}.'.format(tree_nwk))

    df = pd.read_table(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
    logger.debug('Read the annotation file {}.'.format(data))

    # As the date column is only used for visualisation if there is no visualisation we are not gonna validate it
    min_date, max_date = 0, 0
    if (html_compressed or html) and date_column:
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

    if columns:
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
    if name_column:
        name_column = col_name2cat(name_column)
        if name_column not in df.columns:
            raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                             .format(name_column, _quote(df.columns)))
    elif len(df.columns) == 1:
        name_column = df.columns[0]

    percentage_unknown = df.isnull().sum(axis=0) / df.shape[0]
    max_unknown_percentage = percentage_unknown.max()
    if max_unknown_percentage > .9:
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
    return root, df, max_date, min_date, name_column


def _serialize_predicted_states(columns, out_data, root):
    # Not using DataFrames to speed up document writing
    with open(out_data, 'w+') as f:
        f.write('node\t{}\n'.format('\t'.join(columns)))
        for node in root.traverse():
            f.write('{}'.format(node.name))
            for column in columns:
                value = getattr(node, column, None)
                f.write('\t{}'.format(' or '.join(value) if isinstance(value, list) else value))
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

    parser = argparse.ArgumentParser(description="Visualisation of annotated phylogenetic trees (as html maps).")

    annotation_group = parser.add_argument_group('annotation-related arguments')
    annotation_group.add_argument('-d', '--data', required=True, type=str,
                                  help="the annotation file in tab/csv format with the first row "
                                       "containing the column names.")
    annotation_group.add_argument('-s', '--data_sep', required=False, type=str, default='\t',
                                  help="the column separator for the data table. "
                                       "By default is set to tab, i.e. for tab file. "
                                       "Set it to ',' if your file is csv.")
    annotation_group.add_argument('-i', '--id_index', required=False, type=int, default=0,
                                  help="the index of the column in the data table that contains the tree tip names, "
                                       "indices start from zero (by default is set to 0).")
    annotation_group.add_argument('-c', '--columns', nargs='*',
                                  help="names of the data table columns that contain states "
                                       "to be analysed with PASTML, or to be copied."
                                       "If not specified, all columns will be considered.",
                                  type=str)
    annotation_group.add_argument('--date_column', required=False, default=None,
                                  help="name of the data table column that contains tip dates.",
                                  type=str)

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree', help="the input tree in newick format.", type=str, required=True)

    pastml_group = parser.add_argument_group('ancestral-state inference-related arguments')
    pastml_group.add_argument('-m', '--model', required=False, default=F81, choices=[JC, F81, EFT], type=str,
                              help='the evolutionary model to be used by PASTML, by default {}.'.format(F81))
    pastml_group.add_argument('--prediction_method', required=False, default=MPPA,
                              choices=[MPPA, MAP, JOINT, DOWNPASS, ACCTRAN, DELTRAN, COPY], type=str,
                              help='the ancestral state prediction method to be used by PASTML, '
                                   'by default {}.'.format(MPPA))
    pastml_group.add_argument('--column2parameters', required=False, default=None, type=dict,
                              help='optional way to fix some parameters, must be in a form {column: {param: value}}, '
                                   'where param can be a state (then the value should specify its frequency between 0 and 1),'
                                   'or "scaling factor" (then the value should be the scaling factor for tree branches, '
                                   'e.g. set to 1 to keep the original branches).')

    vis_group = parser.add_argument_group('visualisation-related arguments')
    vis_group.add_argument('-n', '--name_column', type=str, default=None,
                           help="name of the data table column to be used for node names "
                                "in the compressed map visualisation (must be one of those specified in columns)."
                                "If the data table contains only one column it will be used by default.")
    vis_group.add_argument('--tip_size_threshold', type=int, default=REASONABLE_NUMBER_OF_TIPS,
                           help="Remove the tips of size less than the threshold-th from the compressed map "
                                "(set to 1e10 to keep all tips). The larger it is the less tips will be trimmed.")

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_data', required=False, type=str,
                           help="the output annotation file with the states inferred by PASTML.")
    pastml_group.add_argument('--work_dir', required=False, default=None, type=str,
                              help="(optional) str: path to the folder where pastml should put its files "
                                   "(e.g. estimated parameters, etc.). "
                                   "If the specified folder does not exist, it will be created.")
    out_group.add_argument('-p', '--html_compressed', required=False, default=None, type=str,
                           help="the output summary map visualisation file (html).")
    out_group.add_argument('-l', '--html', required=False, default=None, type=str,
                           help="the output tree visualisation file (html).")

    parser.add_argument('-v', '--verbose', action='store_true',
                        help="print information on the progress of the analysis")
    params = parser.parse_args()

    pastml_pipeline(**vars(params))


if '__main__' == __name__:
    main()
