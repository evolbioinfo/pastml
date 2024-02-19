import os
import warnings
from collections import defaultdict
from multiprocessing.pool import ThreadPool

from pastml import value2list, col_name2cat, PASTML_VERSION, quote
from pastml.acr import CHARACTER, METHOD
from pastml.acr.acr import acr, serialize_predicted_states, serialize_acr
from pastml.acr.maxlikelihood import ML_METHODS, MARGINAL_ML_METHODS
from pastml.acr.maxlikelihood.models.CustomRatesModel import CUSTOM_RATES
from pastml.acr.maxlikelihood.models.EFTModel import EFT
from pastml.acr.maxlikelihood.models.HKYModel import HKY
from pastml.acr.maxlikelihood.models.JTTModel import JTT
from pastml.annotation import annotate_forest
from pastml.file import get_combined_ancestral_state_file, get_named_tree_file, get_pastml_work_dir
from pastml.logger import set_up_pastml_logger
from pastml.acr.maxlikelihood.ml import MPPA, MAP, JOINT
from pastml.acr.maxlikelihood.models.SimpleModel import SCALING_FACTOR, SMOOTHING_FACTOR
from pastml.acr.maxlikelihood.models.F81Model import F81
from pastml.acr.maxlikelihood.models.JCModel import JC
from pastml.acr.parsimony import ACCTRAN, DELTRAN, DOWNPASS, MP_METHODS
from pastml.politomy import resolve_polytomies_based_on_acr, COPY
from pastml.tree import read_forest, save_tree, copy_forest
from pastml.visualisation.cytoscape_manager import TIMELINE_SAMPLED, TIMELINE_NODES, TIMELINE_LTT
from pastml.visualisation.tree_compressor import REASONABLE_NUMBER_OF_TIPS, VERTICAL, HORIZONTAL, TRIM
from pastml.visualisation.visualize import visualize_itol_html_pajek

warnings.filterwarnings("ignore", append=True)


def pastml_pipeline(tree, data=None, data_sep='\t', id_index=0,
                    columns=None, prediction_method=MPPA, model=F81,
                    parameters=None, rate_matrix=None,
                    name_column=None, root_date=None, timeline_type=TIMELINE_SAMPLED,
                    tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, colours=None,
                    out_data=None, html_compressed=None, html=None, html_mixed=None, work_dir=None,
                    verbose=False, forced_joint=False, upload_to_itol=False, itol_id=None, itol_project=None,
                    itol_tree_name=None, offline=False, threads=0, reoptimise=False, focus=None,
                    resolve_polytomies=False, smoothing=False, frequency_smoothing=False,
                    pajek=None, pajek_timing=VERTICAL):
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
    logger = set_up_pastml_logger(verbose)
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    if isinstance(columns, str):
        columns = [columns]
    roots = read_forest(tree, columns=columns if data is None else None, root_dates=root_date)
    columns, column2states = annotate_forest(roots, columns=columns, data=data, data_sep=data_sep, id_index=id_index,
                                             unknown_treshold=.9, state_threshold=.75)
    if name_column:
        name_column = col_name2cat(name_column)
        if name_column not in columns:
            raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                             .format(name_column, quote(columns)))
    elif len(columns) == 1:
        name_column = columns[0]

    if resolve_polytomies:
        copied_roots = copy_forest(roots)

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

    column2parameters = parameters if parameters else defaultdict(lambda: None)
    column2rates = rate_matrix if rate_matrix else defaultdict(lambda: None)

    prediction_methods = value2list(len(columns), prediction_method, MPPA)
    column2method = dict(zip(columns, prediction_methods))
    models = value2list(len(columns), model, F81)
    column2model = dict(zip(columns, models))

    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)

    def _work(character):
        prediction_method = column2method[character]
        if COPY == prediction_method:
            return {CHARACTER: character, METHOD: prediction_method}
        acr_res = acr(roots, character, column2states[character],
                      prediction_method=column2method[character], model=column2model[character],
                      parameters=column2parameters[character], rate_file=column2rates[character],
                      force_joint=forced_joint,
                      reoptimise=reoptimise, tau=smoothing, frequency_smoothing=frequency_smoothing)
        serialize_acr((acr_res, work_dir))
        return acr_res

    if threads > 1:
        with ThreadPool(processes=threads - 1) as pool:
            pool.map(func=_work, iterable=columns)
    else:
        for character in columns:
            _work(character)

    if resolve_polytomies:
        roots = copied_roots
        resolve_polytomies_based_on_acr(roots, columns, column2states=column2states,
                                        prediction_method=prediction_methods, model=models,
                                        column2parameters=column2parameters, column2rates=column2rates,
                                        force_joint=forced_joint)
    logger.debug('\n=============SAVING RESULTS=============')
    if not out_data:
        out_data = os.path.join(work_dir, get_combined_ancestral_state_file())
    serialize_predicted_states(columns, out_data, roots)
    new_nwk = get_named_tree_file(tree)
    save_tree(roots, columns=columns, nwk=os.path.join(work_dir, new_nwk))

    visualize_itol_html_pajek(colours, column2states, focus, html, html_compressed, html_mixed, itol_id,
                              itol_project, itol_tree_name, name_column, new_nwk, offline, pajek, pajek_timing,
                              roots, timeline_type, tip_size_threshold, upload_to_itol, work_dir)


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

    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=PASTML_VERSION))

    parser.add_argument('--threads', required=False, default=0, type=int,
                        help="Number of threads PastML can use for parallesation. "
                             "By default detected automatically based on the system. "
                             "Note that PastML will at most use as many threads "
                             "as the number of characters (-c option) being analysed plus one.")

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
