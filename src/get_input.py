from dotenv.main import load_dotenv
import dbnomics as db
import pandas as pd
from static_data import pop_countries_rename_dict, rename_dict

load_dotenv()


class DataMacro:
    """Class to scrape macroeconomic data"""

    def __init__(self, base_year):
        self.base_year = base_year
        self.countries_to_drop_list = None

    @property
    def gdp_constant_prices(self):
        # Get the input data
        data = db.fetch_series(provider_code='IMF',
                               dataset_code='WEO:latest',
                               series_code='.NGDP_R.national_currency',
                               max_nb_series=300)

        gdp_global = data[['WEO Country', 'weo-country', 'period', 'value']]
        gdp_global = gdp_global.rename(columns={'WEO Country': 'country',
                                                'weo-country': 'iso3c',
                                                'period': 'date'}).sort_values(by=['iso3c', 'date'])

        gdp_global.set_index(['country', 'iso3c'], inplace=True)

        # Drop countries
        gdp_global = pd.pivot_table(gdp_global, index=gdp_global.index.names, values='value', columns=['date'])
        countries_to_drop_list = gdp_global[gdp_global.iloc[:, -4:].isna().all(axis=1)].index
        gdp_global.drop(index=countries_to_drop_list, inplace=True)
        self.countries_to_drop_list = countries_to_drop_list.get_level_values("country").to_list()

        return gdp_global

    @property
    def population(self):
        pop = db.fetch_series(provider_code='UNDATA',
                              dataset_code='DF_UNDATA_WPP',
                              series_code='SP_POP_TOTL.A._T._T._T..M',
                              max_nb_series=300)
        pop = pop.query("period >= '1990' and period <= '2050'")
        pop = pop[['period', 'value', 'Reference Area']].rename({'period': 'date', 'Reference Area': 'country'}, axis=1)
        pop.country = pop.country.replace(pop_countries_rename_dict)
        pop.country = pop.country.str.replace(' ', '_')
        pop = pop.query(f"country not in  {self.countries_to_drop_list}")
        pop.set_index(['country'], inplace=True)
        pop = pd.pivot_table(pop, index=pop.index, values='value', columns=['date']).T

        return pop

    @property
    def imp_ppp_conv_rate_base(self):
        ppp_fx = db.fetch_series(provider_code='IMF',
                                 dataset_code='WEO:latest',
                                 series_code='.PPPEX.national_currency_per_current_international_dollar',
                                 max_nb_series=300)
        ppp_fx_base = ppp_fx.query(f"period == {self.base_year}")
        ppp_fx_base = ppp_fx_base[['WEO Country', 'weo-country', 'period', 'value']]
        ppp_fx_base = ppp_fx_base.rename(columns={'WEO Country': 'country',
                                                  'weo-country': 'iso3c',
                                                  'period': 'date'}).sort_values(by=['iso3c', 'date'])

        ppp_fx_base.set_index(['country', 'iso3c'], inplace=True)
        ppp_fx_base = pd.pivot_table(ppp_fx_base, index=ppp_fx_base.index.names, values='value', columns=['date']).T

        # Data cleaning
        ppp_fx_base.columns = ppp_fx_base.columns.set_levels(ppp_fx_base.columns.levels[0].str.replace(' ', '_'),
                                                             level=0)
        ppp_fx_base.columns = ppp_fx_base.columns.set_levels(
            ppp_fx_base.columns.levels[0].to_series().replace(rename_dict),
            level=0)
        ppp_fx_base.columns = ppp_fx_base.columns.set_levels(
            ppp_fx_base.columns.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)

        return ppp_fx_base

    @property
    def gdp_current_prices_base(self):
        gdp_curr_p = db.fetch_series(provider_code='IMF',
                                     dataset_code='WEO:latest',
                                     series_code='.NGDP.national_currency',
                                     max_nb_series=300)
        gdp_curr_p = gdp_curr_p[['WEO Country', 'weo-country', 'period', 'value']].query(f"period == {self.base_year}") \
            .rename(columns={'WEO Country': 'country', 'weo-country': 'iso3c', 'period': 'date'}).sort_values(
            by=['iso3c', 'date'])
        gdp_curr_p = gdp_curr_p.query(f"country not in  {self.countries_to_drop_list}")
        gdp_curr_p.set_index(['country', 'iso3c'], inplace=True)

        gdp_curr_p = pd.pivot_table(gdp_curr_p, index=gdp_curr_p.index.names, values='value', columns=['date']).T

        # Data cleaning
        gdp_curr_p.columns = gdp_curr_p.columns.set_levels(gdp_curr_p.columns.levels[0].str.replace(' ', '_'), level=0)
        gdp_curr_p.columns = gdp_curr_p.columns.set_levels(
            gdp_curr_p.columns.levels[0].to_series().replace(rename_dict),
            level=0)
        gdp_curr_p.columns = gdp_curr_p.columns.set_levels(
            gdp_curr_p.columns.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)

        return gdp_curr_p
