from Bio import SeqIO
from Bio.Alphabet import generic_dna
import pandas as pd
from hdx.location.country import Country

if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_fa', required=True, type=str)
    parser.add_argument('--output_fa', required=True, type=str)
    parser.add_argument('--drm_tab', required=True, type=str)
    parser.add_argument('--input_data', required=True, type=str)
    parser.add_argument('--output_data', required=True, type=str)
    params = parser.parse_args()

    # Read and fix DRM metadata
    df = pd.read_table(params.drm_tab, index_col=0, header=0)
    df['Sierra subtype'] = df['subtypeText'].map(lambda s: s[0])
    df.drop(['subtypeText'], axis=1, inplace=True)
    df = df.join(pd.read_table(params.input_data, index_col=0, header=0), how='outer')

    df = df[df['Subtype'] == df['Sierra subtype']]
    df.drop(['Sierra subtype'], axis=1, inplace=True)

    SeqIO.write((_ for _ in SeqIO.parse(params.input_fa, "fasta", alphabet=generic_dna) if _.id in df.index),
                params.output_fa, "fasta")

    # Prettify mutation states
    for mutation in (_ for _ in df.columns if _.startswith('RT:') or _.startswith('PR:')):
        df[mutation] = df[mutation].fillna(False).astype(bool).map(
            {True: 'resistant', False: 'sensitive', None: 'sensitive'})

    # Add Location info
    countries = df['Country Code'].unique()
    country2iso3 = {_: Country.get_iso3_country_code_fuzzy(_)[0] for _ in countries if not pd.isnull(_)}
    iso32info = {_: Country.get_country_info_from_iso3(_) for _ in country2iso3.values()}


    def get_location(_):
        if _ not in iso32info:
            return None
        info = iso32info[_]
        region = info['Region Name']
        if region == 'Africa':
            if info['Country or Area'] == 'South Africa':
                return 'South Africa'
            return info['Intermediate Region Name'] if info['Intermediate Region Name'] else info['Sub-region Name']
        if region == 'Americas':
            return info['Sub-region Name']
        return region


    df['Country ISO3'] = df['Country Code'].apply(lambda _: country2iso3[_] if _ in country2iso3 else None)
    df['Continent'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Region Name'] if _ in iso32info else None)
    df['Country'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Country or Area'] if _ in iso32info else None)
    df['Sub-region'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Sub-region Name'] if _ in iso32info else None)
    df['Region'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Region Name'] if _ in iso32info else None)
    df['Location'] = df['Country ISO3'].apply(get_location)
    df['Int-region'] = df['Country ISO3'].apply(lambda _: (iso32info[_]['Intermediate Region Name']
                                                           if iso32info[_]['Intermediate Region Name']
                                                           else iso32info[_]['Sub-region Name']) if _ in iso32info else None)
    df['IsDeveloped'] = df['Country ISO3'].apply(
        lambda _: iso32info[_]['Developed / Developing Countries']
        if _ in iso32info and 'Developed / Developing Countries' in iso32info[_] else None)

    df.to_csv(params.output_data, sep='\t', header=True, index=True)
