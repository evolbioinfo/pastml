from visualisation.generate_geomap import generate_map

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--tree', required=True, type=str)
    parser.add_argument('--metadata', required=True, type=str)
    parser.add_argument('--geo_html', required=True, type=str)
    parser.add_argument('--location_col', required=True, type=str)
    parser.add_argument('--country_col', required=True, type=str)
    params = parser.parse_args()

    generate_map(data=params.metadata, country=params.country_col, location=params.location_col,
                 tree=params.tree, html=params.geo_html)
