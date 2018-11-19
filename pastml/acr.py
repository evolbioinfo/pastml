import logging
import os
from collections import namedtuple
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
from pastml.tree import read_tree, name_tree, collapse_zero_branches, date_tips, REASONABLE_NUMBER_OF_TIPS

COPY = 'COPY'

ACRCopyResult = namedtuple('ACRCopyResult', field_names=['character', 'states', 'method'])


def parse_parameters(params, states):
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


def reconstruct_ancestral_states(tree, feature, states, avg_br_len, num_nodes,
                                 prediction_method=MPPA, model=None, params=None):
    logging.getLogger('pastml').debug('ACR settings for {}:\n\tMethod:\t{}{}.\n'.format(feature, prediction_method,
                                                                                        '\n\tModel:\t{}'.format(model)
                                                                                        if model and is_ml(
                                                                                            prediction_method) else ''))
    if COPY == prediction_method:
        return {CHARACTER: feature, STATES: states, METHOD: prediction_method}
    if is_ml(prediction_method):
        freqs, sf = None, None
        if params is not None:
            freqs, sf = parse_parameters(params, states)
        return ml_acr(tree, feature, prediction_method, model, states, avg_br_len, num_nodes, freqs, sf)
    if is_parsimonious(prediction_method):
        return parsimonious_acr(tree, feature, prediction_method, states, num_nodes)

    raise ValueError('Method {} is unknown, should be one of ML ({}, {}, {}), one of MP ({}, {}, {}) or {}'
                     .format(prediction_method, MPPA, MAP, JOINT, ACCTRAN, DELTRAN, DOWNPASS, COPY))


def acr(tree, df, prediction_method=MPPA, model=F81, column2parameters=None):
    columns = preannotate_tree(df, tree)
    if column2parameters is not None:
        column2parameters = {col_name2cat(col): params for (col, params) in column2parameters.items()}
    else:
        column2parameters = {}

    avg_br_len, num_nodes = get_tree_stats(tree)

    logging.getLogger('pastml').debug('\n=============RECONSTRUCTING ANCESTRAL STATES=============\n')

    def _work(args):
        return reconstruct_ancestral_states(*args)

    prediction_methods = value2list(len(columns), prediction_method, MPPA)
    models = value2list(len(columns), model, F81)

    with ThreadPool() as pool:
        acr_results = \
            pool.map(func=_work, iterable=((tree, column, np.sort([_ for _ in df[column].unique()
                                                                   if pd.notnull(_) and _ != '']),
                                            avg_br_len, num_nodes, method, model,
                                            column2parameters[column] if column in column2parameters else None)
                                           for (column, method, model) in zip(columns, prediction_methods, models)))

    return acr_results


def quote(str_list):
    return ', '.join('"{}"'.format(_) for _ in str_list) if str_list is not None else ''


def pastml_pipeline(tree, data, out_data=None, html_compressed=None, html=None, data_sep='\t', id_index=0, columns=None,
                    name_column=None, tip_size_threshold=REASONABLE_NUMBER_OF_TIPS,
                    model=F81, prediction_method=MPPA, verbose=False, date_column=None, column2parameters=None,
                    work_dir=None):
    """
    Applies PASTML to the given tree with the specified states and visualizes the result (as html maps).

    :param date_column: str (optional), name of the data table column that contains tip dates.
    :param out_data: str, path to the output annotation file with the states inferred by PASTML.
    :param tree: str, path to the input tree in newick format.
    :param data: str, path to the annotation file in tab/csv format with the first row containing the column names.
    :param html_compressed: str, path where the output summary map visualisation file (html) will be created.
    :param html: str (optional), path where the output tree visualisation file (html) will be created.
    :param data_sep: char (optional, by default '\t'), the column separator for the data table.
    By default is set to tab, i.e. for tab-delimited file. Set it to ',' if your file is csv.
    :param id_index: int (optional, by default is 0) the index of the column in the data table
    that contains the tree tip names, indices start from zero.
    :param columns: list of str (optional), names of the data table columns that contain states
    to be analysed with PASTML, if not specified all columns will be considered.
    :param name_column: str (optional), name of the data table column to be used for node names in the visualisation
    (must be one of those specified in columns, if columns are specified). If the data table contains only one column,
    it will be used by default.
    :param tip_size_threshold: int (optional, by default is 15), remove the tips of size less than threshold-th
    from the compressed map (set to 1e10 to keep all). The larger it is the less tips will be trimmed.
    :param model: str (optional, default is F81), model to be used by PASTML.
    :param prediction_method: str (optional, default is MPPA), ancestral state prediction method to be used by PASTML.
    :param verbose: bool, print information on the progress of the analysis.
    :param column2parameters: dict, an optional way to fix some parameters, must be in a form {column: {param: value}},
    where param can be a state (then the value should specify its frequency between 0 and 1),
    or "scaling factor" (then the value should be the scaling factor for three branches,
    e.g. set to 1 to keep the original branches).
    :param work_dir: str, path to the folder where PASTML should put its files (e.g. estimated parameters, etc.).
    If the specified folder does not exist, it will be created.
    :return: void
    """
    logger = set_up_logger(verbose)

    if work_dir:
        os.makedirs(work_dir, exist_ok=True)

    root = read_tree(tree)
    name_tree(root)
    collapse_zero_branches(root)
    # this is expected by PASTML
    root.name = 'ROOT'
    root.dist = 0

    df = pd.read_table(data, sep=data_sep, index_col=id_index, header=0, dtype=str)
    df.index = df.index.map(str)
    df.columns = [col_name2cat(column) for column in df.columns]

    if columns is None:
        columns = list(df.columns)
    else:
        columns = [col_name2cat(column) for column in columns]

    min_date, max_date = 0, 0
    # As the date column is only used for visualisation if there is no visualisation we are not gonna validate it
    if (html_compressed or html) and date_column:
        date_column = col_name2cat(date_column)
        if date_column not in df.columns:
            raise ValueError('The date column {} not found among the annotation columns: {}.'
                             .format(date_column, quote(df.columns)))
        try:
            df[date_column] = pd.to_datetime(df[date_column], infer_datetime_format=True)
        except ValueError:
            try:
                df[date_column] = pd.to_datetime(df[date_column], format='%Y.0')
            except ValueError:
                raise ValueError("Could not infer the date format for column {}, please check it.".format(date_column))
        min_date, max_date = date_tips(root, df[date_column])
        logger.debug("Dates vary between {} and {}.".format(min_date, max_date))

    unknown_columns = set(columns) - set(df.columns)
    if unknown_columns:
        raise ValueError('{} of the specified columns ({}) {} not found among the annotation columns: {}.'
                         .format('One' if len(unknown_columns) == 1 else 'Some',
                                 quote(unknown_columns),
                                 'is' if len(unknown_columns) == 1 else 'are',
                                 quote(df.columns)))

    if name_column:
        name_column = col_name2cat(name_column)
        if name_column not in columns:
            raise ValueError('The name column ({}) should be one of those specified as columns ({}).'
                             .format(quote([name_column]), quote(columns)))
    elif len(columns) == 1:
        name_column = columns[0]

    df = df[columns]
    node_names = np.array([n.name for n in root.traverse()])
    tip_names = np.array([n.name for n in root])
    df = df[np.in1d(df.index, node_names)]

    if not df.shape[0]:
        raise ValueError('Your tree tip names (e.g. {}) do not correspond to annotation id column values (e.g. {}). '
                         'Check your annotation file.'
                         .format(', '.join(tip_names[: min(len(tip_names), 3)]),
                                 ', '.join(df.index[: min(len(df), 3)])))

    percentage_unknown = df.isnull().sum(axis=0) / df.shape[0]
    max_unknown_percentage = percentage_unknown.max()
    if max_unknown_percentage > .9:
        raise ValueError('{:.1f}% of tip annotations for {} are unknown, not enough data to infer ancestral states. '
                         'Check your annotation file and if its id column corresponds to the tree tip names.'
                         .format(max_unknown_percentage * 100, percentage_unknown.idxmax()))

    percentage_unique = df.nunique() / df.count()
    max_unique_percentage = percentage_unique.max()
    if df.count()[0] > 100 and max_unique_percentage > .5:
        raise ValueError('The column {} seem to contain non-categorical data: {:.1f}% of values are unique.'
                         'PASTML cannot infer ancestral states for a tree with too many tip states.'
                         .format(percentage_unique.idxmax(), 100 * max_unique_percentage))

    acr_results = acr(root, df, prediction_method=prediction_method, model=model, column2parameters=column2parameters)

    if work_dir and not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file(columns=columns))
    if out_data:
        df = pd.DataFrame(columns=columns)
        for node in root.traverse():
            for column in columns:
                value = getattr(node, column, None)
                df.loc[node.name, column] = ' or '.join(value) if isinstance(value, list) else value
        df.to_csv(out_data, sep='\t')

    if work_dir:
        new_tree = os.path.join(work_dir, get_named_tree_file(tree))
        root.write(outfile=new_tree, format_root_node=True)

        for acr_result in acr_results:
            out_param_file = \
                os.path.join(work_dir,
                             get_pastml_parameter_file(method=acr_result[METHOD],
                                                       model=acr_result[MODEL] if MODEL in acr_result else None,
                                                       column=acr_result[CHARACTER]))
            df = pd.DataFrame(columns=['value'])
            for name, value in acr_result.items():
                if name not in [FREQUENCIES, STATES, MARGINAL_PROBABILITIES]:
                    df.loc[name, 'value'] = value
            if is_ml(acr_result[METHOD]):
                for state, freq in zip(acr_result[STATES], acr_result[FREQUENCIES]):
                    df.loc[state, 'value'] = freq
            df.to_csv(out_param_file, index_label='parameter')

            if is_marginal(acr_result[METHOD]):
                out_mp_file = \
                    os.path.join(work_dir,
                                 get_pastml_marginal_prob_file(method=acr_result[METHOD], model=acr_result[MODEL],
                                                               column=acr_result[CHARACTER]))
                acr_result[MARGINAL_PROBABILITIES].to_csv(out_mp_file, sep='\t')

    if html or html_compressed:
        logger.debug('\n=============VISUALISATION=============\n')
        visualize(root, column2states={acr_result[CHARACTER]: acr_result[STATES] for acr_result in acr_results},
                  html=html, html_compressed=html_compressed, min_date=min_date, max_date=max_date,
                  name_column=name_column, tip_size_threshold=tip_size_threshold)

    return root


def set_up_logger(verbose):
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
