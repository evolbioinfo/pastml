import os

from pastml.visualisation.colour_generator import get_enough_colours

total_template = """DATASET_STYLE

SEPARATOR TAB
DATASET_LABEL	{column}
COLOR	#ffffff

LEGEND_COLORS	{colours}
LEGEND_LABELS	{states}
LEGEND_SHAPES	{shapes}
LEGEND_TITLE	{column}

DATA
"""
node_template = "{id}	branch	node	{color}	2	normal"


def generate_itol_annotations(root, column2states, work_dir):
    for column, states in column2states.items():
        colours = get_enough_colours(len(states))
        value2colour = dict(zip(states, colours))
        with open(os.path.join(work_dir, 'iTOL_style-{}.txt'.format(column)), 'w+') as f:
            f.write(total_template.format(column=column, colours='\t'.join(colours), states='\t'.join(states),
                                          shapes='\t'.join(['1'] * len(states))))
            for n in root.traverse('preorder'):
                if not n.is_root():
                    states = getattr(n, column, set())
                    state = next(iter(states)) if len(states) == 1 else None
                    if state:
                        f.write('{id}\tbranch\tnode\t{colour}\t2\tnormal\n'.format(id=n.name, colour=value2colour[state]))
