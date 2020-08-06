import os

import pandas as pd
import numpy as np

from pastml.acr import _parse_pastml_parameters, _validate_input, PASTML_VERSION, _set_up_pastml_logger
from pastml.file import get_pastml_work_dir
from pastml.annotation import get_forest_stats
from pastml.ml import SCALING_FACTOR, SMOOTHING_FACTOR, \
    marginal_counts
from pastml.models.f81_like import F81, JC, EFT
from pastml.models.hky import HKY_STATES, HKY
from pastml.models.jtt import JTT_STATES, JTT
from pastml.visualisation.colour_generator import parse_colours, get_enough_colours
from pastml.visualisation.cytoscape_manager import save_as_transition_html


def count_transitions(tree, data, column, parameters, out_transitions, data_sep='\t', id_index=0, model=F81, threads=0,
                      n_repetitions=1000, work_dir=None, html=None, verbose=False, offline=False, colours=None):
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

    :param out_transitions: path to the output transition table file.
    :type out_transitions: str

    :param verbose: (optional, default is False) print information on the progress of the analysis.
    :type verbose: bool

    :param threads: (optional, default is 0, which stands for automatic) number of threads to be used for parallesation.
        By default detected automatically based on the system.
    :type threads: int

    :param n_repetitions: Number of times the ancestral scenario is drawn from the marginal probabilities.
        The transition counts are averaged over all these scenarios.

    :return: void
    """
    logger = _set_up_pastml_logger(verbose)

    forest, columns, column2states, name_column, age_label, parameters = \
        _validate_input(tree, [column], None, data, data_sep, id_index, None, parameters=[parameters])

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

    avg_br_len, num_nodes, num_tips, tree_len = get_forest_stats(forest)
    freqs, sf, kappa, tau = _parse_pastml_parameters(parameters, states, num_tips=num_tips, reoptimise=False)
    if not tau:
        tau = 0

    logger.debug('\n=============COUNTING TRANSITIONS for {} ({} repetitions)==============================='
                 .format(column, n_repetitions))
    logger.debug('Using the following parameters:{}{}{}{}.'
                 .format(''.join('\n\tfrequency of {}:\t{:.6f}'
                                 .format(state, freq) for state, freq in zip(states, freqs)),
                         '\n\tkappa:\t{:.6f}'.format(kappa) if HKY == model else '',
                         '\n\tscaling factor:\t{:.6f}, i.e. {:.6f} changes per avg branch'
                         .format(sf, sf * avg_br_len),
                         '\n\tsmoothing factor:\t{:.6f}'.format(tau))
                 )
    result = marginal_counts(forest, column, model, states, num_nodes, tree_len, freqs, sf, kappa, tau,
                             n_repetitions=n_repetitions)
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
                                state2colour=dict(zip(states, colours)), work_dir=work_dir, local_css_js=offline)


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
                           choices=[JC, F81, EFT, HKY, JTT],
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
                                "other that the one specified by --html/--html_compressed).")
    vis_group.add_argument('--colours', type=str, required=False, default=None,
                           help='optional way to specify the colours used for character state visualisation. '
                                'Should be in the same order '
                                'as the ancestral characters (see -c, --columns) '
                                'for which the reconstruction is to be preformed. '
                                'Could be given only for the first few characters. '
                                'Each file should be tab-delimited, with two columns: '
                                'the first one containing character states, '
                                'and the second, named "colour", containing colours, in HEX format (e.g. #a6cee3).')

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    parser.add_argument('--threads', required=False, default=0, type=int,
                        help="Number of threads the program can use for parallesation. "
                             "By default detected automatically based on the system. ")

    params = parser.parse_args()

    count_transitions(**vars(params))


if '__main__' == __name__:
    main()
