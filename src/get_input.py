from dotenv.main import load_dotenv
import numpy as np
import dbnomics as db
import pandas as pd


load_dotenv()


class DataMacro:
    """Class to scrape macroeconomic data"""
    def __init__(self, base_year):
        self.base_year = base_year

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

        f'List of countries dropped: {countries_to_drop_list}'

        return gdp_global

    @property
    def population(self):
        pop = db.fetch_series(provider_code='UNDATA',
                              dataset_code='DF_UNDATA_WPP',
                              series_code='SP_POP_TOTL.A._T._T._T..M',
                              max_nb_series=200)
        pop = pop.query("period >= '1980' and period <= '2050'")
        pop = pop[['period', 'value', 'Reference Area']].rename({'period': 'date', 'Reference Area': 'UN_country'}, axis=1)

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

        return ppp_fx_base

    @property
    def gdp_current_prices_base(self):
        gdp_curr_p = db.fetch_series(provider_code='IMF',
                                     dataset_code='WEO:latest',
                                     series_code='.NGDP.national_currency',
                                     max_nb_series=300)
        gdp_curr_p = gdp_curr_p[['WEO Country', 'weo-country', 'period', 'value']].query(f"period == {self.base_year}")\
            .rename(columns={'WEO Country': 'country', 'weo-country': 'iso3c', 'period': 'date'}).sort_values(
            by=['iso3c', 'date'])

        gdp_curr_p.set_index(['country', 'iso3c'], inplace=True)

        gdp_curr_p = pd.pivot_table(gdp_curr_p, index=gdp_curr_p.index.names, values='value', columns=['date']).T

        return gdp_curr_p








