import logging
import os

import pandas as pd

from pastml.ml import is_marginal, MARGINAL_PROBABILITIES
from pastml import METHOD, CHARACTER
from pastml.visualisation.colour_generator import get_enough_colours

STYLE_FILE_HEADER_TEMPLATE = """DATASET_STYLE

SEPARATOR TAB
DATASET_LABEL	{column}
COLOR	#ffffff

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column}

DATA
#NODE_ID TYPE   NODE  COLOR LABEL_OR_STYLE SIZE_FACTOR
"""

POPUP_FILE_HEADER = """POPUP_INFO

SEPARATOR TAB

DATA
#NODE_ID POPUP_TITLE POPUP_CONTENT
"""

POPUP_CONTENT_TEMPLATE = "<b>{key}: </b>" \
                         "<div style='overflow:auto;max-width:50vw;'>" \
                         "<span style='white-space:nowrap;'>{value}</span></div>"


def generate_itol_annotations(column2states, work_dir, acrs, state_df, date_col, tip2date):
    popup_file = os.path.join(work_dir, 'iTOL_popup_info.txt')
    with open(popup_file, 'w+') as pf:
        pf.write(POPUP_FILE_HEADER)

    state_df['itol_type'] = 'branch'
    state_df['itol_node'] = 'node'
    state_df['itol_style'] = 'normal'
    state_df['itol_size'] = 2

    for column, states in column2states.items():
        colours = get_enough_colours(len(states))
        value2colour = dict(zip(states, colours))
        style_file = os.path.join(work_dir, 'iTOL_style-{}.txt'.format(column))
        with open(style_file, 'w+') as sf:
            sf.write(STYLE_FILE_HEADER_TEMPLATE.format(column=column, colours='\t'.join(colours),
                                                       states='\t'.join(states),
                                                       shapes='\t'.join(['1'] * len(states))))
        col_df = state_df[state_df[column].apply(len) == 1]
        col_df['itol_colour'] = col_df[column].apply(lambda _: value2colour[next(iter(_))])
        col_df[['itol_type', 'itol_node', 'itol_colour', 'itol_style', 'itol_size']].to_csv(style_file, sep='\t',
                                                                                            header=False, mode='a')
        logging.getLogger('pastml').debug('Generated iTol style file for {}: {}.'.format(column, style_file))
    state_df = state_df[list(column2states.keys()) + ['dist']]
    for c in column2states.keys():
        state_df[c] = state_df[c].apply(lambda _: ' or '.join(sorted(_)))
    state_df.columns = ['ACR {} predicted state'.format(c) for c in column2states.keys()] + ['Node dist']
    state_df['Node id'] = state_df.index
    state_df.loc[list(tip2date.keys()), date_col] = list(tip2date.values())

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
    logging.getLogger('pastml').debug('Generated iTol pop-up file: {}.'.format(popup_file))


