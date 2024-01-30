import logging
import os
from pathlib import Path

import pandas as pd

from pastml.visualisation import get_formatted_date
from pastml.visualisation.colour_generator import get_enough_colours, parse_colours
from itolapi import Itol

from pastml.tree import DATE
from pastml.visualisation.cytoscape_manager import DATE_LABEL

STYLE_FILE_HEADER_TEMPLATE = """DATASET_STYLE

SEPARATOR TAB
DATASET_LABEL	{column} (branch colours)
COLOR	#ffffff

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column}

DATA
#NODE_ID TYPE   NODE  COLOR SIZE_FACTOR LABEL_OR_STYLE
"""

COLORSTRIP_FILE_HEADER_TEMPLATE = """DATASET_COLORSTRIP

SEPARATOR TAB
BORDER_WIDTH	0.5
MARGIN	5
STRIP_WIDTH	25
COLOR	#ffffff
DATASET_LABEL	{column} (colour strip)

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column}

DATA
#NODE_ID COLOR LABEL_OR_STYLE
"""

POPUP_FILE_HEADER = """POPUP_INFO

SEPARATOR TAB

DATA
#NODE_ID POPUP_TITLE POPUP_CONTENT
"""

POPUP_CONTENT_TEMPLATE = "<b>{key}: </b>" \
                         "<div style='overflow:auto;max-width:50vw;'>" \
                         "<span style='white-space:nowrap;'>{value}</span></div>"

DEFAULT_ITOL_PROJECT = 'Sample project'


def generate_itol_annotations(forest, column2states, work_dir, date_col,
                              tree_path, itol_id=None, itol_project=None, itol_tree_name=None,
                              column2colours=None):
    annotation_files = []
    popup_file = os.path.join(work_dir, 'iTOL_popup_info.txt')

    for column, states in column2states.items():
        num_unique_values = len(states)
        if column2colours and column in column2colours:
            colours = parse_colours(column2colours[column], states)
        else:
            colours = get_enough_colours(num_unique_values)
        value2colour = dict(zip(states, colours))
        style_file = os.path.join(work_dir, 'iTOL_style-{}.txt'.format(column))
        with open(style_file, 'w+') as sf:
            sf.write(STYLE_FILE_HEADER_TEMPLATE
                     .format(column=column, colours='\t'.join(colours), states='\t'.join(states),
                             shapes='\t'.join(['1'] * len(states))))
            for tree in forest:
                for node in tree.traverse():
                    if len(getattr(node, column, set())) == 1:
                        state = next(iter(getattr(node, column)))
                        sf.write('{name}\tbranch\tnode\t{color}\t2\tnormal\n'
                                 .format(name=node.name, color=value2colour[state]))
        annotation_files.append(style_file)
        logging.getLogger('pastml').debug('Generated iTol style file for {}: {}.'.format(column, style_file))

        colorstrip_file = os.path.join(work_dir, 'iTOL_colorstrip-{}.txt'.format(column))
        with open(colorstrip_file, 'w+') as csf:
            csf.write(COLORSTRIP_FILE_HEADER_TEMPLATE.format(column=column, colours='\t'.join(colours),
                                                             states='\t'.join(states),
                                                             shapes='\t'.join(['1'] * len(states))))
            for tree in forest:
                for tip in tree:
                    if len(getattr(tip, column, set())) == 1:
                        state = next(iter(getattr(tip, column)))
                        csf.write('{name}\t{color}\t{name}\n'
                                  .format(name=tip.name, color=value2colour[state]))
        annotation_files.append(colorstrip_file)
        logging.getLogger('pastml').debug('Generated iTol colorstrip file for {}: {}.'.format(column, colorstrip_file))

    with open(popup_file, 'w+') as pf:
        pf.write(POPUP_FILE_HEADER)
        for tree in forest:
            for node in tree.traverse():
                info = ['<b>{key}: </b> {value}'.format(key=k, value=v)
                        for (k, v) in (('node id', node.name), ('node dist', node.dist),
                                       (date_col, get_formatted_date(node, date_col == DATE_LABEL)))]
                for c in sorted(column2states.keys()):
                    v = getattr(node, c, set())
                    if pd.isna(v) or v is None:
                        v = []
                    info.append(POPUP_CONTENT_TEMPLATE.format(key=c, value=' or '.join(sorted(v))))
                pf.write('{name}\tACR results\t{info}\n'.format(name=node.name, info='<br>'.join(info)))
    annotation_files.append(popup_file)
    logging.getLogger('pastml').debug('Generated iTol pop-up file: {}.'.format(popup_file))

    if itol_id:
        if not itol_project:
            logging.getLogger('pastml').info('Trying "{}" as iTOL project, the tree will be uploaded to. '
                                              'To upload to a different project, use itol_project argument.'
                                              .format(DEFAULT_ITOL_PROJECT))
        if not itol_tree_name:
            itol_tree_name = os.path.splitext(os.path.basename(tree_path))[0]
        tree_id, web_page = upload_to_itol(tree_path, annotation_files, tree_name=itol_tree_name,
                                           tree_description=None, project_name=itol_project, upload_id=itol_id)
        if web_page:
            with open(os.path.join(work_dir, 'iTOL_url.txt'), 'w+') as f:
                f.write(web_page)
        if tree_id:
            with open(os.path.join(work_dir, 'iTOL_tree_id.txt'), 'w+') as f:
                f.write(tree_id)
    else:
        logging.getLogger('pastml').info('To upload your tree(s) to iTOL, please specify the itol_id argument.')


def upload_to_itol(tree_path, dataset_paths, tree_name=None, tree_description=None, project_name=None, upload_id=None):
    try:
        itol_uploader = Itol()
        itol_uploader.add_file(Path(tree_path))
        for annotation_file in dataset_paths:
            itol_uploader.add_file(Path(annotation_file))
        if tree_name:
            itol_uploader.params['treeName'] = tree_name
        if tree_description:
            itol_uploader.params['treeDescription'] = tree_description
        if upload_id:
            itol_uploader.params['uploadID'] = upload_id
            if project_name:
                itol_uploader.params['projectName'] = project_name
        if itol_uploader.upload():
            logging.getLogger('pastml').info(
                'Successfully uploaded your tree ({}) to iTOL: {}.'.format(itol_uploader.comm.tree_id,
                                                                           itol_uploader.get_webpage()))
            return itol_uploader.comm.tree_id, itol_uploader.get_webpage()
        else:
            status = itol_uploader.comm.upload_output
    except Exception as e:
        status = e
    logging.getLogger('pastml').error(
        'Failed to upload your tree to iTOL because of "{}". Please check your internet connection and iTOL settings{}.'
            .format(status,
                    (', e.g. your iTOL batch upload id ({}){}'
                     .format(upload_id,
                             (' and whether the project "{}" exists'.format(project_name) if project_name else '')))
                    if upload_id else ''))
    return None, None
