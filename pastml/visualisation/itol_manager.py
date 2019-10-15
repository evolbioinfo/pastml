import logging
import os

import pandas as pd

from pastml.ml import is_marginal, MARGINAL_PROBABILITIES, MODEL
from pastml import METHOD, CHARACTER, STATES
from pastml.visualisation.colour_generator import get_enough_colours
from itolapi import Itol

from pastml.tree import DATE

STYLE_FILE_HEADER_TEMPLATE = """DATASET_STYLE

SEPARATOR TAB
DATASET_LABEL	{column} ({method})
COLOR	#ffffff

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column} ({method})

DATA
#NODE_ID TYPE   NODE  COLOR SIZE_FACTOR LABEL_OR_STYLE
"""

COLORSTRIP_FILE_HEADER_TEMPLATE = """DATASET_COLORSTRIP

SEPARATOR TAB
BORDER_WIDTH	0.5
MARGIN	5
STRIP_WIDTH	25
COLOR	#ffffff
DATASET_LABEL	{column}

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


def generate_itol_annotations(column2states, work_dir, acrs, state_df, date_col,
                              tree_path, itol_id=None, itol_project=None, itol_tree_name=None):
    annotation_files = []
    popup_file = os.path.join(work_dir, 'iTOL_popup_info.txt')
    with open(popup_file, 'w+') as pf:
        pf.write(POPUP_FILE_HEADER)

    state_df['itol_type'] = 'branch'
    state_df['itol_node'] = 'node'
    state_df['itol_style'] = 'normal'
    state_df['itol_size'] = 2

    for acr_result in acrs:
        column = acr_result[CHARACTER]
        states = acr_result[STATES]
        colours = get_enough_colours(len(states))
        value2colour = dict(zip(states, colours))
        style_file = os.path.join(work_dir, 'iTOL_style-{}.txt'.format(column))
        with open(style_file, 'w+') as sf:
            sf.write(STYLE_FILE_HEADER_TEMPLATE
                     .format(column=column, colours='\t'.join(colours), states='\t'.join(states),
                             shapes='\t'.join(['1'] * len(states)),
                             method='{}{}'.format(acr_result[METHOD],
                                                  ('+{}'.format(acr_result[MODEL]) if MODEL in acr_result else ''))))
        col_df = state_df[state_df[column].apply(len) == 1]
        col_df['itol_label'] = col_df[column].apply(lambda _: next(iter(_)))
        col_df['itol_colour'] = col_df['itol_label'].apply(lambda _: value2colour[_])
        col_df[['itol_type', 'itol_node', 'itol_colour', 'itol_size', 'itol_style']].to_csv(style_file, sep='\t',
                                                                                            header=False, mode='a')
        annotation_files.append(style_file)
        logging.getLogger('pastml').debug('Generated iTol style file for {}: {}.'.format(column, style_file))

        colorstrip_file = os.path.join(work_dir, 'iTOL_colorstrip-{}.txt'.format(column))
        with open(colorstrip_file, 'w+') as csf:
            csf.write(COLORSTRIP_FILE_HEADER_TEMPLATE.format(column=column, colours='\t'.join(colours),
                                                             states='\t'.join(states),
                                                             shapes='\t'.join(['1'] * len(states))))
        col_df[['itol_colour', 'itol_label']].to_csv(colorstrip_file, sep='\t', header=False, mode='a')
        annotation_files.append(colorstrip_file)
        logging.getLogger('pastml').debug('Generated iTol colorstrip file for {}: {}.'.format(column, colorstrip_file))

    state_df = state_df[list(column2states.keys()) + ['dist', DATE]]
    for c in column2states.keys():
        state_df[c] = state_df[c].apply(lambda _: ' or '.join(sorted(_)))
    state_df.columns = ['ACR {} predicted state'.format(c) for c in column2states.keys()] + ['Node dist', date_col]
    state_df['Node id'] = state_df.index

    for acr_result in acrs:
        if is_marginal(acr_result[METHOD]):
            df = acr_result[MARGINAL_PROBABILITIES]
            state_df.loc[df.index.map(str),
                         'ACR {character} marginal probabilities'.format(character=acr_result[CHARACTER])] = \
                df.apply(lambda vs: ', '.join(('{}: {:g}'.format(c, mp) for (c, mp) in zip(df.columns, vs))),
                         axis=1)
    cols = sorted(state_df.columns, reverse=True)
    state_df['popup_info'] = \
        state_df[cols].apply(lambda vs: '<br>'.join(((POPUP_CONTENT_TEMPLATE
                                                      if c.startswith('ACR ') else '<b>{key}: </b>: {value}')
                                                    .format(key=c, value=v) for (c, v) in zip(cols, vs)
                                                     if not pd.isna(v))),
                             axis=1)
    state_df['label'] = 'ACR results'
    state_df[['label', 'popup_info']].to_csv(popup_file, sep='\t', mode='a', header=False)
    annotation_files.append(popup_file)
    logging.getLogger('pastml').debug('Generated iTol pop-up file: {}.'.format(popup_file))

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


def upload_to_itol(tree_path, dataset_paths, tree_name=None, tree_description=None, project_name=None, upload_id=None):
    itol_uploader = Itol()
    itol_uploader.add_file(tree_path)
    for annotation_file in dataset_paths:
        itol_uploader.add_file(annotation_file)
    if tree_name:
        itol_uploader.params['treeName'] = tree_name
    if tree_description:
        itol_uploader.params['treeDescription'] = tree_description
    if upload_id:
        itol_uploader.params['uploadID'] = upload_id
        if project_name:
            itol_uploader.params['projectName'] = project_name
    status = itol_uploader.upload()
    if not status:
        logging.getLogger('pastml').error(
            'Failed to upload your tree to iTOL, please check your internet connection and itol settings{}.'
                .format((', e.g. your iTOL batch upload id ({}){}'
                         .format(upload_id,
                                 (' and whether the project {} exists'.format(project_name) if project_name else '')))
                        if upload_id else ''))
        return None, None
    logging.getLogger('pastml').debug(
        'Successfully uploaded your tree ({}) to iTOL: {}.'.format(itol_uploader.comm.tree_id,
                                                                   itol_uploader.get_webpage()))
    return itol_uploader.comm.tree_id, itol_uploader.get_webpage()
