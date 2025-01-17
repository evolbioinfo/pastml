import pandas as pd
from hdx.location.country import Country

country2loc = {'China': 'Asia', 'South Korea': 'Asia',
               'Myanmar': 'Asia', 'Philippines': 'Asia', 'Thailand': 'Asia',
               'India': 'Indian subcontinent', 'Nepal': 'Indian subcontinent', 'Pakistan': 'Indian subcontinent',
               #
               'Israel': 'Europe', 'Yemen': 'Horn of Africa', 'Georgia': 'Europe',
               #
               'Austria': 'Europe', 'Belgium': 'Europe', 'Cyprus': 'Europe', 'Denmark': 'Europe',
               'Finland': 'Europe', 'France': 'Europe', 'Germany': 'Europe', 'Greece': 'Europe',
               'Italy': 'Europe', 'Luxembourg': 'Europe', 'Netherlands': 'Europe',
               'Norway': 'Europe', 'Portugal': 'Europe', 'Spain': 'Europe', 'Sweden': 'Europe',
               'Switzerland': 'Europe', 'United Kingdom': 'Europe',
               'Czech Rep.': 'Europe', 'Slovakia': 'Europe', 'Poland': 'Europe', 'Romania': 'Europe',
               #
               'Russian Federation': 'Europe', 'Ukraine': 'Europe',
               #
               'Australia': 'Australia',
               #
               'United States': 'North America',
               #
               'Cuba': 'Central America', 'Honduras': 'Central America',
               #
               'Argentina': 'South America', 'Brazil': 'South America', 'Uruguay': 'South America',
               'Venezuela': 'South America',
               #
               'Central Africa': 'Central Africa', 'Dem. Rep. of Congo': 'Central Africa',
               'Zambia': 'Central Africa',
               #
               'Burundi': 'East Africa', 'Kenya': 'East Africa', 'Rwanda': 'East Africa', 'Tanzania': 'East Africa',
               'Uganda': 'East Africa',
               #
               'Djibouti': 'Horn of Africa', 'Eritrea': 'Horn of Africa', 'Ethiopia': 'Horn of Africa',
               'Somalia': 'Horn of Africa', 'Sudan': 'Horn of Africa',
               #
               'South Africa': 'South Africa', 'Swaziland': 'South Africa',
               #
               'Botswana': 'Southern Africa ex SA', 'Malawi': 'Southern Africa ex SA',
               'Mozambique': 'Southern Africa ex SA', 'Zimbabwe': 'Southern Africa ex SA',
               #
               'Cameroon': 'West Africa', 'Equatorial Guinea': 'West Africa', 'Gabon': 'West Africa',
               'Mali': 'West Africa', 'Nigeria': 'West Africa', 'Senegal': 'West Africa'}

iso32loc = {Country.get_iso3_country_code_fuzzy(c)[0]: loc for (c, loc) in country2loc.items()
            if Country.get_iso3_country_code_fuzzy(c)[0]}


if '__main__' == __name__:
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--drm_tab', required=True, type=str)
    parser.add_argument('--input_data', required=True, type=str)
    parser.add_argument('--output_data', required=True, type=str)
    params = parser.parse_args()

    df = pd.read_csv(params.drm_tab, index_col=0, header=0, sep='\t')
    df = df.join(pd.read_csv(params.input_data, index_col=0, header=0, sep='\t'), how='outer')

    # Add Location info
    countries = df['Country Code'].unique()
    country2iso3 = {_: Country.get_iso3_country_code_fuzzy(_)[0] for _ in countries if not pd.isnull(_)}
    iso32info = {_: Country.get_country_info_from_iso3(_) for _ in country2iso3.values()}

    df['Country ISO3'] = df['Country Code'].apply(lambda _: country2iso3[_] if _ in country2iso3 else None)
    df['Continent'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Region Name'] if _ in iso32info else None)
    df['Country'] = df['Country ISO3'].apply(lambda _: iso32info[_]['Country or Area'] if _ in iso32info else None)

    def get_location(_, sa=False):
        if pd.isnull(_):
            return None
        loc = iso32loc[_]
        if 'Indian' in loc:
            return 'Asia'
        if not sa:
            return loc
        if 'Europe' in loc or 'Asia' in loc:
            return loc
        return Country.get_country_info_from_iso3(_)['Country or Area']

    df['Loc_SA'] = df['Country ISO3'].map(lambda _: get_location(_, True))
    df['Loc'] = df['Country ISO3'].map(get_location)
    df.to_csv(params.output_data, sep='\t')
