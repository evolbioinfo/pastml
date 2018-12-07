import os
from html import escape

import numpy as np
import pandas as pd
from jinja2 import Environment, PackageLoader

from pastml.visualisation.colour_generator import get_enough_colours
from pastml.tree import read_tree

ISO_EXISTS = False
try:
    from hdx.location.country import Country

    ISO_EXISTS = True
except ImportError:
    pass


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Visualisation of locations.")

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
    annotation_group.add_argument('-c', '--country', required=True,
                                  help="name of the data table column that contains countries.",
                                  type=str)
    annotation_group.add_argument('-l', '--location', required=True,
                                  help="name of the data table column that contains locations (to be mapped to colours).",
                                  type=str)

    tree_group = parser.add_argument_group('tree-related arguments')
    tree_group.add_argument('-t', '--tree', help="the input tree in newick format.", type=str, required=True)

    out_group = parser.add_argument_group('output-related arguments')
    out_group.add_argument('-o', '--html', required=False, default=None, type=str,
                           help="the output map visualisation file (html).")
    params = parser.parse_args()

    generate_map(**vars(params))


def generate_map(data, country, location, tree, html, data_sep='\t', id_index=0):
    df = pd.read_table(data, sep=data_sep, header=0, index_col=id_index)
    if country not in df.columns:
        raise ValueError('The country column {} not found among the annotation columns: {}.'
                         .format(country, df.columns))
    if location not in df.columns:
        raise ValueError('The location column {} not found among the annotation columns: {}.'
                         .format(location, df.columns))
    df = df[np.in1d(df.index.astype(np.str), [_.name for _ in read_tree(tree)])]
    df.sort_values(by=[location], inplace=True, na_position='last')
    ddf = df.drop_duplicates(subset=[country], inplace=False, keep='first')
    country2location = {c: l for c, l in zip(ddf[country], ddf[location]) if not pd.isnull(c) and not pd.isnull(l)}
    if ISO_EXISTS:
        country2iso = {_: Country.get_iso2_from_iso3(iso) for (_, iso) in
                       ((_, Country.get_iso3_country_code_fuzzy(_)[0]) for _ in country2location.keys()) if iso}
    else:
        country2iso = {_: escape(_) for _ in country2location.keys()}
    iso2num = {iso: len(df[df[country] == c]) for c, iso in country2iso.items()}
    iso2loc = {iso: country2location[c] for c, iso in country2iso.items()}
    iso2loc_num = {iso: len(df[df[location] == loc]) for iso, loc in iso2loc.items()}
    iso2tooltip = {iso: escape('{}: {} samples (out of {} in {})'
                               .format(c, iso2num[iso], iso2loc_num[iso], iso2loc[iso]))
                   for (c, iso) in country2iso.items()}
    locations = sorted([_ for _ in df[location].unique() if not pd.isnull(_)])
    colours = get_enough_colours(len(locations))
    iso2colour = {iso: colours[locations.index(loc)] for iso, loc in iso2loc.items()}

    env = Environment(loader=PackageLoader('pastml'))
    template = env.get_template('geo_map.html')
    page = template.render(iso2colour=iso2colour, colours=colours, iso2tooltip=iso2tooltip)
    os.makedirs(os.path.abspath(os.path.dirname(html)), exist_ok=True)
    with open(html, 'w+') as fp:
        fp.write(page)


if '__main__' == __name__:
    main()
