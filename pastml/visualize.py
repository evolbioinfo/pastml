import logging
import os
import warnings
from collections import defaultdict

from pastml import col_name2cat, get_personalized_feature_name, PASTML_VERSION, quote
from pastml.annotation import annotate_forest, annotate_dates
from pastml.file import get_named_tree_file, get_pastml_work_dir
from pastml.logger import set_up_pastml_logger
from pastml.ml import MARGINAL_ML_METHODS, get_default_ml_method
from pastml.parsimony import get_default_mp_method
from pastml.tree import read_forest, save_tree
from pastml.visualisation.cytoscape_manager import visualize, TIMELINE_SAMPLED, TIMELINE_NODES, TIMELINE_LTT
from pastml.visualisation.itol_manager import generate_itol_annotations
from pastml.visualisation.tree_compressor import REASONABLE_NUMBER_OF_TIPS, VERTICAL, HORIZONTAL, TRIM

warnings.filterwarnings("ignore", append=True)


def vis_pipeline(tree, data=None, data_sep='\t', id_index=0,
                 columns=None, name_column=None, root_date=None, timeline_type=TIMELINE_SAMPLED,
                 tip_size_threshold=REASONABLE_NUMBER_OF_TIPS, colours=None,
                 html_compressed=None, html=None, html_mixed=None, work_dir=None,
                 verbose=False,
                 upload_to_itol=False, itol_id=None, itol_project=None,
                 itol_tree_name=None, offline=False, focus=None,
                 pajek=None, pajek_timing=VERTICAL):
    """
    Visualized ACRs for the given tree(s) as html maps.

    :param tree: path to the input tree(s) in newick format (must be rooted).
    :type tree: str

    :param data: (optional) path to the annotation file(s) in tab/csv format with the first row containing the column names.
        If not given, the annotations should be contained in the tree file itself.
    :type data: list(str)
    :param data_sep: (optional, by default '\t') column separator for the annotation table(s).
        By default is set to tab, i.e. for tab-delimited file. Set it to ',' if your file(s) is(are) csv.
    :type data_sep: char
    :param id_index: (optional, by default is 0) index of the column in the annotation table(s)
        that contains the tree tip names, indices start from zero.
    :type id_index: int

    :param columns: (optional) name(s) of the annotation table column(s) that contain character(s)
        to be analysed. If not specified all annotation table columns will be considered.
    :type columns: str or list(str)
    :param name_column: (optional) name of the annotation table column to be used for node names
        in the compressed map visualisation
        (must be one of those specified in ``columns``, if ``columns`` are specified).
        If only one annotation table is given, and it contains only one column (in addition to the index column),
        it will be used by default.
    :type name_column: str
    :param root_date: (optional) date(s) of the root(s) (for dated tree(s) only),
        if specified, used to visualize a timeline based on dates (otherwise it is based on distances to root).
    :type root_date: str or pandas.datetime or float or list
    :param tip_size_threshold: (optional, by default is 15) recursively remove the tips
        of size less than the threshold-th largest tip from the compressed map (set to 1e10 to keep all).
        The larger it is the fewer tips will be trimmed.
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
    :param work_dir: (optional) path to the folder where pastml files (such as state-to-colour mapping) are to be stored.
        Default is <path_to_input_tree>/<input_tree_name>_pastml. If the folder does not exist, it will be created.
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
    logger = set_up_pastml_logger(verbose, default_level=logging.INFO)
    logger.debug('\n=============INPUT DATA VALIDATION=============')
    roots = read_forest(tree, columns=columns if data is None else None, root_dates=root_date)
    columns, column2states = annotate_forest(roots, columns=columns, data=data, data_sep=data_sep, id_index=id_index,
                                             unknown_treshold=1, state_threshold=1)
    if name_column:
        name_column = col_name2cat(name_column)
        if name_column not in column2states:
            raise ValueError('The name column ("{}") should be one of those specified as columns ({}).'
                             .format(name_column, quote(columns)))
    elif len(column2states) == 1:
        name_column = columns[0]
    logger.debug('Finished input validation.')

    if not work_dir:
        work_dir = get_pastml_work_dir(tree)
    os.makedirs(work_dir, exist_ok=True)
    new_nwk = os.path.join(work_dir, get_named_tree_file(tree))
    save_tree(roots, columns=column2states.keys(), nwk=new_nwk)

    visualize_itol_html_pajek(colours, column2states, focus, html, html_compressed, html_mixed, itol_id,
                              itol_project, itol_tree_name, logger, name_column, new_nwk, offline, pajek, pajek_timing,
                              roots, timeline_type, tip_size_threshold, upload_to_itol, work_dir)


def visualize_itol_html_pajek(colours, column2states, focus, html, html_compressed, html_mixed, itol_id,
                              itol_project, itol_tree_name, name_column, new_nwk, offline, pajek, pajek_timing,
                              roots, timeline_type, tip_size_threshold, upload_to_itol, work_dir):
    logger = logging.getLogger('pastml')
    if upload_to_itol or html or html_compressed:
        if colours:
            if isinstance(colours, str):
                colours = [colours]
            if isinstance(colours, list):
                colours = dict(zip(column2states.keys(), colours))
            elif isinstance(colours, dict):
                colours = {col_name2cat(col): cls for (col, cls) in colours.items()}
            else:
                raise ValueError('Colours should be either a list or a dict, got {}.'.format(type(colours)))
        else:
            colours = {}
    if upload_to_itol:
        generate_itol_annotations(roots, column2states, work_dir, new_nwk, itol_id, itol_project, itol_tree_name, colours)
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
                  name_column=name_column, tip_size_threshold=tip_size_threshold,
                  timeline_type=timeline_type, work_dir=work_dir, local_css_js=offline, column2colours=colours,
                  focus=focus, pajek=pajek, pajek_timing=pajek_timing)


def main():
    """
    Entry point, calling :py:func:`pastml.visualize.vis_pipeline` with command-line arguments.

    :return: void
    """
    import argparse

    parser = argparse.ArgumentParser(description="Ancestral scenario visualisation "
                                                 "for rooted phylogenetic trees.", prog='pastml_viz')

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree', help="input tree(s) in newick format (must be rooted).",
                            type=str, required=True)

    annotation_group = parser.add_argument_group('annotation-file-related arguments')
    annotation_group.add_argument('-d', '--data', nargs='*',  type=str,
                                  help="annotation file(s) in tab/csv format with the first row "
                                       "containing the column names. "
                                       "If not given, the annotations should be contained in the tree file itself.")
    annotation_group.add_argument('-s', '--data_sep', required=False, type=str, default='\t',
                                  help="column separator for the annotation table(s). "
                                       "By default is set to tab, i.e. for a tab-delimited file. "
                                       "Set it to ',' if your file(s) is(are) csv.")
    annotation_group.add_argument('-i', '--id_index', required=False, type=int, default=0,
                                  help="index of the annotation table column containing tree tip names, "
                                       "indices start from zero (by default is set to 0).")

    annotation_group.add_argument('-c', '--columns', nargs='*',
                                  help="names of the annotation table columns that contain characters "
                                       "to be visualised. "
                                       "If not specified, all columns are considered.", type=str)

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

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('--work_dir', required=False, default=None, type=str,
                           help="path to the folder where pastml files, such as state-to-colour mapping"
                                "are to be stored. "
                                "Default is <path_to_input_tree>/<input_tree_name>_pastml. "
                                "If the folder does not exist, it will be created."
                           .format(', '.join(MARGINAL_ML_METHODS)))
    out_group.add_argument('--html_compressed', required=False, default=None, type=str,
                           help="path to the output compressed map visualisation file (html).")
    out_group.add_argument('--pajek', required=False, default=None, type=str,
                           help="path to the output vertically compressed visualisation file (Pajek NET Format). "
                                "Produced only if --html_compressed is specified.")
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

    vis_pipeline(**vars(params))


if '__main__' == __name__:
    main()
