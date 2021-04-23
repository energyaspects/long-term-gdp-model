import dbnomics as db
import pandas as pd
from src.static_data import pop_countries_rename_dict, rename_dict
from helper_functions_ea import check_env
from helper_functions_ea.logger_tool.main import Logger

check_env()

# Static Functions


def country_name_columns_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Format country names for consistency across datasets."""

    # Data cleaning
    df.columns = df.columns.set_levels(df.columns.levels[0].str.replace(' ', '_'), level=0)
    df.columns = df.columns.set_levels(df.columns.levels[0].to_series().replace(rename_dict), level=0)
    df.columns = df.columns.set_levels(df.columns.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)
    return df


def imf_base_data_transformation(df: pd.DataFrame, base_year: int, countries_to_drop_list: list) -> pd.DataFrame:
    """
    Return a transposed dataframe: select and rename columns, remove countries with missing data & filter on base year.
    """

    df = (df[['WEO Country', 'weo-country', 'period', 'value']]
          .rename(columns={'WEO Country': 'country', 'weo-country': 'iso3c', 'period': 'date'})
          .query(f"date == {base_year} & country not in {countries_to_drop_list} ")
          .sort_values(by=['iso3c', 'date']))

    df = pd.pivot_table(df, index=['country', 'iso3c'], values='value', columns=['date']).T
    return df

# Input class


class DataMacro:
    """Class to scrape macroeconomic data from DBnomics."""

    def __init__(self, base_year, **kwargs):
        self.base_year = base_year
        self.countries_to_drop_list = None
        self.cty_missing_values = kwargs.get('cty_missing_values_list')
        self.input_logger = Logger(name="Input Data Logger").logger

    @property
    def gdp_constant_prices(self):
        """Set the constant GDP property."""
        
        # Get the input data
        data = db.fetch_series(provider_code='IMF',
                               dataset_code='WEO:latest',
                               series_code='.NGDP_R.national_currency',
                               max_nb_series=300)

        gdp_global = (data[['WEO Country', 'weo-country', 'period', 'value']]
                      .rename(columns={'WEO Country': 'country',
                                       'weo-country': 'iso3c',
                                       'period': 'date'})
                      .sort_values(by=['iso3c', 'date'])
                      .set_index(['country', 'iso3c']))

        # Drop countries
        gdp_global = pd.pivot_table(gdp_global, index=gdp_global.index.names, values='value', columns=['date'])
        countries_to_drop_list = gdp_global[gdp_global.iloc[:, -4:].isna().all(axis=1)].index
        gdp_global.drop(index=countries_to_drop_list, inplace=True)

        # Get list of countries with missing recent values (i.e. after 2022)
        # Add countries with known missing historical data (clearly specified in methodology and country grouping list)
        self.countries_to_drop_list = (countries_to_drop_list.get_level_values("country").to_list() +
                                       ["Eritrea", "Nauru", "Palau", "San_Marino", "South_Sudan", "Timor-Leste"])

        self.input_logger.info("GDP constant IMF data fetch successful.")
        self.input_logger.warning(f"The following countries were dropped: {self.countries_to_drop_list}")

        return gdp_global

    @property
    def population(self):
        """Set the IMF constant GDP property."""
        
        pop = db.fetch_series(provider_code='UNDATA',
                              dataset_code='DF_UNDATA_WPP',
                              series_code='SP_POP_TOTL.A._T._T._T..M',
                              max_nb_series=300)
        # Data cleaning
        pop = (pop[['period', 'value', 'Reference Area']]
               .query("period >= '1990' and period <= '2050'")
               .rename({'period': 'date', 'Reference Area': 'country'}, axis=1))

        # Format country names to match EA naming convention (different names than IMF)
        pop.country = pop.country.replace(pop_countries_rename_dict)
        pop.country = pop.country.str.replace(' ', '_')

        # Drop countries with missing data & transpose df
        pop = pop.query(f"country not in  {self.countries_to_drop_list}")
        pop.set_index(['country'], inplace=True)
        pop = pd.pivot_table(pop, index=pop.index, values='value', columns=['date']).T

        self.input_logger.info("Population UNPOP data fetch successful.")

        return pop

    @property
    def imp_ppp_conv_rate_base(self):
        """Set the IMF PPP property."""
        
        ppp_fx = db.fetch_series(provider_code='IMF',
                                 dataset_code='WEO:latest',
                                 series_code='.PPPEX.national_currency_per_current_international_dollar',
                                 max_nb_series=300)
        # Data cleaning
        ppp_fx_base = imf_base_data_transformation(ppp_fx, self.base_year, self.countries_to_drop_list)
        ppp_fx_base = country_name_columns_formatting(ppp_fx_base)

        self.input_logger.info("PPP IMF data fetch successful.")

        return ppp_fx_base

    @property
    def gdp_current_prices_base(self):
        """Set the IMF current GDP property."""
        
        gdp_curr_p = db.fetch_series(provider_code='IMF',
                                     dataset_code='WEO:latest',
                                     series_code='.NGDP.national_currency',
                                     max_nb_series=300)
        # Data cleaning
        gdp_curr_p = imf_base_data_transformation(gdp_curr_p, self.base_year, self.countries_to_drop_list)
        gdp_curr_p = country_name_columns_formatting(gdp_curr_p)

        self.input_logger.info("GDP current IMF data fetch successful.")

        return gdp_curr_p

    @property
    def gdp_per_capita_cepii(self):
        """Set the CEPII constant PPP GDP/C property."""
        
        series = [f'CEPII/CHELEM-GDP/{iso3c}.GDP-PPP-CAP' for iso3c in self.cty_missing_values]
        data = db.fetch_series(series, max_nb_series=300)

        cepii_per_cap = (data[['period', 'value', 'country']]
                         .query('period > 1989')
                         .rename({'country': 'iso3c'}, axis=1))

        cepii_per_cap = pd.pivot_table(cepii_per_cap, index='period', columns=['iso3c'], values='value').div(1000)

        self.input_logger.info(f"GDP/C CEPII for {self.cty_missing_values} data fetch successful.")
        
        return cepii_per_cap
