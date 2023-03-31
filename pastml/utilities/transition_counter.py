import os

import numpy as np
import pandas as pd

from pastml.acr import _validate_input, PASTML_VERSION, _set_up_pastml_logger, \
    calculate_observed_freqs
from pastml.annotation import ForestStats
from pastml.file import get_pastml_work_dir
from pastml.ml import marginal_counts
from pastml.models import SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.models.CustomRatesModel import CUSTOM_RATES, CustomRatesModel
from pastml.models.EFTModel import EFT, EFTModel
from pastml.models.F81Model import F81, F81Model
from pastml.models.HKYModel import HKY, HKYModel, HKY_STATES
from pastml.models.JCModel import JC, JCModel
from pastml.models.JTTModel import JTT, JTTModel, JTT_STATES
from pastml.visualisation.colour_generator import parse_colours, get_enough_colours
from pastml.visualisation.cytoscape_manager import save_as_transition_html

model2class = {F81: F81Model, JC: JCModel, CUSTOM_RATES: CustomRatesModel, HKY: HKYModel, JTT: JTTModel, EFT: EFTModel}


def count_transitions(tree, data, column, parameters, out_transitions, data_sep='\t', id_index=0, model=F81,
                      threshold=1,
                      n_repetitions=1000, rate_matrix=None, work_dir=None, html=None, verbose=False,
                      offline=False, colours=None):
    """

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
    :param column: name of the annotation table column that contains the character to be analysed.
    :type column: str

    :param model: (optional, default is pastml.models.f81_like.F81) evolutionary model.
    :type model: str
    :param parameters: ACR method parameters.
        Could be specified as
        (1a) a dict {column: {param: value}},
        where column corresponds to the character for which these parameters should be used,
        or (1b) in a form {column: path_to_param_file};
        or (2) as a path to parameter file.
        The parameter file should be tab-delimited, with two columns: the first one containing parameter names,
        and the second, named "value", containing parameter values.
        Parameters can include character state frequencies (parameter name should be the corresponding state,
        and parameter value - the float frequency value, between 0 and 1),
        tree branch scaling factor (parameter name pastml.ml.SCALING_FACTOR),
        and tree branch smoothing factor (parameter name pastml.ml.SMOOTHING_FACTOR).
    :type parameters: str or dict
    :param rate_matrix: (only for pastml.models.rate_matrix.CUSTOM_RATES model) path to the file
        specifying the rate matrix.
        Could be specified as
        (1) a dict {column: path_to_file},
        where column corresponds to the character for which this rate matrix should be used,
        or (2) as a path to rate matrix file.
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

    :param out_transitions: path to the output transition table file.
    :type out_transitions: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :param n_repetitions: Number of times the ancestral scenario is drawn from the marginal probabilities.
        The transition counts are averaged over all these scenarios.


    :param html: (optional) path to the output transition visualisation file (html).
    :type html: str
    :param threshold: threshold under which the transitions are not shown in the visualisation.
    :type threshold: float
    :param colours: optional way to specify the colours used for character state visualisation.
        Could be specified as
        (1) a dict {state: colour},
        or (2) as a path to colour file, tab-delimited, with two columns: the first one containing character states,
        and the second, named "colour", containing colours, in HEX format (e.g. #a6cee3).
    :type colours: str or list(str) or dict


    :param offline: (optional, default is False) By default (offline=False) PastML assumes
        that there is an internet connection available,
        which permits it to fetch CSS and JS scripts needed for visualisation online.
        With offline=True, PastML will store all the needed CSS/JS scripts in the folder specified by work_dir,
        so that internet connection is not needed
        (but you must not move the output html files to any location other that the one specified by html).
    :type offline: bool
    :param work_dir: (optional) path to the folder where pastml will store all the needed CSS/JS script files
        if offline is set to True.
        Default is <path_to_input_tree>/<input_tree_name>_pastml. If the folder does not exist, it will be created.
    :type work_dir: str

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    threshold = max(0, threshold)

    forest, columns, column2states, name_column, age_label, parameters, rates = \
        _validate_input(tree, [column], None, data, data_sep, id_index, None, parameters=[parameters],
                        rates=[rate_matrix])

    column = columns[0]
    states = column2states[column]
    parameters = parameters[column]

    if model in {HKY, JTT}:
        initial_states = states
        states = HKY_STATES if HKY == model else JTT_STATES
        if not set(initial_states) & set(states):
            raise ValueError('The allowed states for model {} are {}, '
                             'but your annotation file specifies {} as states in column {}.'
                             .format(model, ', '.join(states), ', '.join(initial_states), column))

    missing_data, observed_frequencies, state2index = calculate_observed_freqs(column, forest, states)

    model = model2class[model](parameter_file=parameters,
                               rate_matrix_file=rates[column] if column in rates else None,
                               reoptimise=False, states=states, forest_stats=ForestStats(forest),
                               observed_frequencies=observed_frequencies)

    state_set = set(states)
    counts = np.zeros(len(states))
    state2i = dict(zip(states, range(len(states))))
    for root in forest:
        for n in root.traverse():
            if hasattr(n, column):
                n_states = state_set & getattr(n, column)
                n.add_feature(column, n_states)
                for s in n_states:
                    counts[state2i[s]] += 1 / len(n_states)

    logger.debug('\n=============COUNTING TRANSITIONS for {} ({} repetitions)==============================='
                 .format(column, n_repetitions))
    logger.debug(model)
    result = marginal_counts(forest, column, model, n_repetitions=n_repetitions)
    df = pd.DataFrame(data=result, columns=states, index=states)
    df.to_csv(out_transitions, sep='\t', index_label='from')
    logger.info('Transition counts are saved as {}.'.format(out_transitions))

    if html:
        if offline:
            if not work_dir:
                work_dir = get_pastml_work_dir(html)
            os.makedirs(work_dir, exist_ok=True)

        if colours:
            colours = parse_colours(colours, states)
        else:
            colours = get_enough_colours(len(states))

        save_as_transition_html(character=column, states=states, counts=counts, transitions=result, out_html=html,
                                state2colour=dict(zip(states, colours)), work_dir=work_dir, local_css_js=offline,
                                threshold=threshold)


def main():
    """
    Entry point, calling :py:func:`pastml.utilities.transition_counter.count_transitions` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Transition counting for PastML ACR", prog='transition_counter')

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
    acr_group.add_argument('-c', '--column', required=True,
                           help="the name of the annotation table column that contain the character "
                                "to be analysed.", type=str)
    acr_group.add_argument('-m', '--model', default=F81,
                           choices=[JC, F81, EFT, HKY, JTT, CUSTOM_RATES],
                           type=str, required=False,
                           help='evolutionary model for ML methods (ignored by MP methods). '
                                'When multiple ancestral characters are specified (see -c, --columns), '
                                'the same model can be used for all of them (if only one model is specified), '
                                'or different models can be used (specified in the same order as -c, --columns). '
                                'If multiple models are given, but not for all the characters, '
                                'for the rest of them the default model ({}) is chosen.'.format(F81))
    acr_group.add_argument('--parameters', type=str, required=True,
                           help='the ML-method parameters, '
                                'specified as a path to a tab-delimited file with two columns: '
                                'the first one containing parameter names, '
                                'and the second, named "value", containing parameter values. '
                                'Parameters can include character state frequencies '
                                '(parameter name should be the corresponding state, '
                                'and parameter value - the float frequency value, between 0 and 1),'
                                'tree branch scaling factor (parameter name {}),'.format(SCALING_FACTOR) +
                                'and tree branch smoothing factor (parameter name {}),'.format(SMOOTHING_FACTOR))
    acr_group.add_argument('--rate_matrix', type=str, required=False, default=None,
                           help='(only for {} model) path to the file containing the rate matrix. '
                                'The rate matrix file should specify character states in its first line, '
                                'preceded by #  and separated by spaces. '
                                'The following lines should contain a symmetric squared rate matrix with positive rates'
                                '(and zeros on the diagonal), separated by spaces, '
                                'in the same order at the character states specified in the first line.'
                                'For example, for four states, A, C, G, T '
                                'and the rates A<->C 1, A<->G 4, A<->T 1, C<->G 1, C<->T 4, G<->T 1,'
                                'the rate matrix file would look like:'
                                '# A C G T'
                                '0 1 4 1'
                                '1 0 1 4'
                                '4 1 0 1'
                                '1 4 1 0'.format(CUSTOM_RATES))
    acr_group.add_argument('-n', '--n_repetitions', type=int, required=False, default=1000,
                           help='(default 1000) '
                                'Number of times the ancestral scenario is drawn from the marginal probabilities.'
                                'The transition counts are averaged over all these scenarios.')

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--out_transitions', required=True, type=str,
                           help="path to the output transition count file.")
    out_group.add_argument('-v', '--verbose', action='store_true',
                           help="print information on the progress of the analysis (to console)")
    out_group.add_argument('--html', required=False, default=None, type=str,
                           help="path to the output transition visualisation file (html).")

    vis_group = parser.add_argument_group('visualisation-related arguments')
    vis_group.add_argument('--offline', action='store_true',
                           help="By default (without --offline option) PastML assumes "
                                "that there is an internet connection available, "
                                "which permits it to fetch CSS and JS scripts needed for visualisation online."
                                "With --offline option turned on, PastML will store all the needed CSS/JS scripts "
                                "in the folder specified by --work_dir, so that internet connection is not needed "
                                "(but you must not move the output html files to any location "
                                "other that the one specified by --html).")
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where pastml will store all the needed CSS/JS scripts "
                                "for the --offline mode. "
                                "Default is <path_to_input_tree>/<input_tree_name>_pastml. "
                                "If the folder does not exist, it will be created.")
    vis_group.add_argument('--colours', type=str, required=False, default=None,
                           help='optional way to specify the colours used for character state visualisation. '
                                'A tab-delimited file, with two columns: '
                                'the first one containing character states, '
                                'and the second, named "colour", containing colours, in HEX format (e.g. #a6cee3).')
    vis_group.add_argument('--threshold', type=float, required=False, default=1,
                           help='Do not visualise the interactions below this threshold value.')

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    params = parser.parse_args()

    count_transitions(**vars(params))


if '__main__' == __name__:
    main()
