import numpy as np
import pandas as pd
import os
from src.get_input import DataMacro
from src.gdp_model import GDPModel
from src.static_data import rename_dict, extended_iso_dict
from helper_functions_ea.data_mappings.country_iso import get_iso
from helper_functions_ea.shooju_helper.main import ShoojuTools
from helper_functions_ea.logger_tool.main import Logger
from datetime import date
import shooju
from dateutil.relativedelta import relativedelta

# Static Functions


def map_country_to_region(df: pd.DataFrame, region_df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: region & sub_region."""

    df.columns = (pd.MultiIndex.from_tuples([(c[0],
                                              c[1],
                                              dict(zip(region_df.ISO3C, region_df.Region))[c[1]],
                                              dict(zip(region_df.ISO3C, region_df.Sub_regions_others))[c[1]])
                                             for c in df.columns]))
    df.columns.set_names(['country', 'iso3c', 'region', 'other_region'], inplace=True)
    return df


def add_other_regions_col_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: country name & iso3c for new 'other_regions' series."""

    df.columns = (pd.MultiIndex.from_tuples(
        [(dict(zip(df.columns, ['Other_Africa', 'Other_Asia', 'Other_Europe', 'Other_FSU', 'Other_Latam', 'Other_Middle']))[c],
          c, '', '') for c in df.columns]))
    return df


def add_regions_col_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with new column levels: country name & iso3c for new 'region' series."""

    df.columns = (
        pd.MultiIndex.from_tuples([(c, dict(zip(df.columns, ['AFR', 'AP', 'EU', 'FSU', 'LA', 'ME', 'NA']))[c],
                                    '', '') for c in df.columns]))
    return df


def country_name_index_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with formatted country names."""

    # Data cleaning
    df.index = df.index.set_levels(df.index.levels[0].str.replace(' ', '_'), level=0)
    df.index = df.index.set_levels(df.index.levels[0].to_series().replace(rename_dict), level=0)
    df.index = df.index.set_levels(df.index.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)
    return df

#
# Class for scenario modelling
#


class BaselineScenario:
    """Class to forecast GDP for a baseline scenario."""

    def __init__(self, args_dict, sj_path):
        self.args_dict = args_dict
        self.scenario = 'Baseline'
        self.base_year = args_dict['base_year']
        self.gdp_global = None  # Historical real GDP values in national currency
        self.gdp_log = None  # Historical real GDP log values in national currency
        self.population = None  # UN population medium scenario estimates (Historic & Forecast)
        self.ppp_conv_rate = None  # Purchasing power parity values ($ 2017)
        self.gdp_const_p = None  # Real GDP forecasts in national currency
        self.gdp_current_p = None  # 2017 Nominal GDP values in national currency
        self.gdp_per_capita = None  # GDP per capita ($2017 PPP) (Historic & Forecast)
        self.gdp_growth = None  # Real GDP growth rate historic & forecast
        self.gdp_real_ppp = None  # Real PPP GDP historic & forecast ($2017 PPP)
        self.forecasts_df = pd.DataFrame()
        self.metadata_dictionary = {}
        self.sj = ShoojuTools()
        self.is_train = args_dict['train_model']
        self.sj_path = sj_path
        self.train_end_year = date(args_dict['train_end_year'], 1, 1) if args_dict['train_end_year'] else None
        self.imf_fcst_end_year = None
        self.cols_filter = None
        self.adj_country_list = []
        self.run_type = 'baseline'
        self.country_region_mapping = pd.read_excel(r'utils\region_aggregates.xlsx', sheet_name='new_list')
        self.scenario_logger = Logger(name="Scenario Logger").logger

    def load_inputs(self):
        """Set all macro-economic properties."""

        # Get input data
        self.scenario_logger.info("Loading inputs")
        data_obj = DataMacro(self.base_year)
        self.gdp_global = data_obj.gdp_constant_prices

        self.population = data_obj.population
        self.ppp_conv_rate = data_obj.imp_ppp_conv_rate_base
        self.gdp_current_p = data_obj.gdp_current_prices_base

        # Additional properties
        self.gdp_log = self.gdp_global.apply(np.log)
        self.cols_filter = self.population.columns.intersection(self.ppp_conv_rate.columns.levels[0])  # get common countries between UNPOP and IMF
        self.imf_fcst_end_year = self.gdp_global.columns.max()  # most recent IMF estimate

    def get_forecasts(self):
        """Set forcasts_df property with GDP growth forecasts."""

        groups = self.gdp_log.groupby('iso3c')
        group_keys = list(groups.groups.keys())

        self.scenario_logger.info("Getting forecasts")

        for key in group_keys:
            target = groups.get_group(key).T
            target.dropna(inplace=True)
            target = target[:self.train_end_year]

            mdl_obj = GDPModel(target, key, self.args_dict, self.imf_fcst_end_year)

            if self.is_train:
                mdl_obj.train_predict_pipeline()

            else:
                mdl_obj.predict_pipeline()

            self.forecasts_df = self.forecasts_df.append(mdl_obj.predictions_log.T.set_index(target.columns))

    def combine_dfs(self):
        """Set the constant GDP property and re-value population property (by thousands)."""

        # Join IMF real GDP values with predicted growth rates
        growth_rate_df = self.forecasts_df.div(100).add(1)

        full_gdp_df = self.gdp_global.iloc[:, :].join(growth_rate_df.iloc[:, 1:])
        full_gdp_df.columns = pd.to_datetime(full_gdp_df.columns)

        # Data cleaning
        full_gdp_df = country_name_index_formatting(full_gdp_df)

        # Filter countries and derive real GDP forecasts
        self.gdp_const_p = full_gdp_df.T.loc['1990':, self.cols_filter]
        self.gdp_const_p[f'{self.imf_fcst_end_year.year}':] = self.gdp_const_p[
                                                              f'{self.imf_fcst_end_year.year}':].cumprod()

        # Transform population data
        self.population = self.population.div(1000)

    def get_other_economic_variables(self):
        """Set additional macro-economic properties."""

        self.scenario_logger.info("Populating other macro-economic variables")

        # Get growth rate df
        self.gdp_growth = self.gdp_const_p.pct_change() * 100
        self.gdp_growth = map_country_to_region(self.gdp_growth, self.country_region_mapping)

        # Get GDP base nominal/real
        ratio = self.gdp_current_p[self.cols_filter].div(self.gdp_const_p.loc[f'{self.base_year}'])

        # Apply PPP conversion rates and calculate GDP per capita
        self.population = self.population[self.cols_filter]
        self.population.columns = (pd.MultiIndex.from_tuples(
            [(c, dict(zip(self.country_region_mapping.Name, self.country_region_mapping.ISO3C))[c])
             for c in self.population.columns]))

        self.gdp_per_capita = (self.gdp_const_p
                               .div(self.population)
                               .div(self.ppp_conv_rate[self.cols_filter].values)
                               .mul(ratio.values))

        # Fill missing historical values
        # Filter countries with NANs
        cty_missing_values_list = (self.gdp_per_capita[self.gdp_per_capita.columns[self.gdp_per_capita.isna().any(0)]]
                                   .columns.get_level_values('iso3c').to_list())

        # Get new data - CEPII gdp per capita values
        api_obj = DataMacro(self.base_year, cty_missing_values_list=cty_missing_values_list)
        cepii_gdp_per_c = api_obj.gdp_per_capita_cepii

        # Add county names using EA naming conventions (key=iso3c)
        cepii_gdp_per_c.columns = (pd.MultiIndex.from_tuples(
            [(dict(zip(self.country_region_mapping.ISO3C, self.country_region_mapping.Name))[c], c)
             for c in cepii_gdp_per_c.columns]))

        # Fill nan with CEPII values
        self.gdp_per_capita.update(cepii_gdp_per_c, overwrite=False)

    def get_regional_aggregations(self):
        """Update Population, GDP/C, GDP constant PPP properties with regional aggregates."""

        self.scenario_logger.info("Performing regional aggregations")

        # Add regions and sub_regions to column levels
        self.population = map_country_to_region(self.population, self.country_region_mapping)
        self.gdp_per_capita = map_country_to_region(self.gdp_per_capita, self.country_region_mapping)

        # Get Real GDP levels
        self.gdp_real_ppp = self.gdp_per_capita.mul(self.population)

        # Calculate GDP/C for other regions & regions
        o_gdp = (self.gdp_real_ppp.loc(axis=1)[:, :, :, ['oafrc', 'oasia', 'oeuro', 'ofsu', 'olatam', 'omiddle']]
                 .groupby(level=3, axis=1).sum())
        r_gdp = self.gdp_real_ppp.groupby(level=2, axis=1).sum()

        o_pop = (self.population.loc(axis=1)[:, :, :, ['oafrc', 'oasia', 'oeuro', 'ofsu', 'olatam', 'omiddle']]
                 .groupby(level=3, axis=1).sum())
        r_pop = self.population.groupby(level=2, axis=1).sum()

        o_gdp_per_c = o_gdp.div(o_pop)
        r_gdp_per_c = r_gdp.div(r_pop)

        # Add columns levels
        o_gdp_per_c = add_other_regions_col_levels(o_gdp_per_c)
        r_gdp_per_c = add_regions_col_levels(r_gdp_per_c)

        o_pop = add_other_regions_col_levels(o_pop)
        r_pop = add_regions_col_levels(r_pop)

        o_gdp = add_other_regions_col_levels(o_gdp)
        r_gdp = add_regions_col_levels(r_gdp)

        # Concat existing dfs with other regions and regions

        self.population = pd.concat([self.population, o_pop, r_pop], axis=1)
        self.gdp_real_ppp = pd.concat([self.gdp_real_ppp, o_gdp, r_gdp], axis=1)
        self.gdp_per_capita = pd.concat([self.gdp_per_capita, o_gdp_per_c, r_gdp_per_c], axis=1)

    def upload_to_sj(self, df, df_dict, **kwargs):
        """Create metadata and upload macro-economic data to SJ."""

        self.scenario_logger.info(f"Uploading {df_dict['var']} to SJ")

        # Get list of modelled series (i.e. model output not aggregate) & published series
        country_modelled = self.country_region_mapping.ISO3C.to_list()
        series_published = self.country_region_mapping.Published.to_list() + list(self.country_region_mapping.Region_iso3c.unique())

        # Add iso2c and construct sids
        df.columns = pd.MultiIndex.from_tuples([(c[0], c[1], c[2], c[3],
                                                 f'{get_iso(c[0], extended_dict=extended_iso_dict)}',
                                                 f'{c[1]}_{df_dict["var"]}_{self.scenario}') for c in
                                                df.columns])

        df.columns.set_names(['country', 'iso3c', 'region', 'other_region', 'iso2c', 'sid'], inplace=True)

        # create metadata dictionary with fields
        description = kwargs.get('adj_description')

        for n in range(0, len(df.columns)):
            fields_additions = {
                f'{df.columns[n][5]}':
                    {'country': df.columns[n][0],
                     'country_iso': df.columns[n][4],
                     'iso3c': df.columns[n][1],
                     'region': df.columns[n][2],
                     'sub_region': df.columns[n][3],
                     'economic_property': f'global_macro_{df_dict["var"]}',
                     'unit': df_dict['unit'],
                     'scenario': f'{self.scenario}',
                     'source': 'IMF, Macroeconomic model',
                     'lifecycle_stage': 'forecast',
                     'modelled': True if df.columns[n][1] in country_modelled else False,
                     'published': True if df.columns[n][1] in series_published else False,
                     'frequency': 'yearly'}
            }
            if df.columns[n][0] in self.adj_country_list:
                extra_fields = {
                    'type': "Adj",
                    'description': description
                }
                fields_additions[df.columns[n][5]].update(extra_fields)

            elif self.run_type == 'baseline' or self.run_type == 'scenario':
                extra_fields = {
                    'type': "Raw",
                    'description': f'Macroeconomic Data for {self.scenario} scenario'
                }
                fields_additions[df.columns[n][5]].update(extra_fields)

            else:
                extra_fields = {}
                fields_additions[df.columns[n][5]].update(extra_fields)

            self.metadata_dictionary.update(fields_additions)

        # Only keep column with sids and convert index to date to drop timestamp
        df = df.droplevel(['country', 'iso3c', 'region', 'other_region', 'iso2c'], axis=1)
        df.index = df.index.date

        # Upload to sj
        self.sj.df_upload_wide(
            df=df,
            sid_pefix=self.sj_path,
            metadata_dictionary=self.metadata_dictionary,
            name=df_dict['job_name']
        )

    def upload_preprocessing(self, **kwargs):
        """Create property specific metadata fields."""

        gdp_growth_fields_dict = {'var': 'GDP_growth', 'unit': 'difflog', 'job_name': 'GDP_growth_forecast_upload'}
        population_fields_dict = {'var': 'POP', 'unit': 'Total population (thousands)',
                                  'job_name': 'Pop_forecast_upload'}
        gdp_per_capita_fields_dict = {'var': 'GDP_Capita', 'unit': 'GDP/capita (2017k$ PPP)',
                                      'job_name': 'GDP_per_capita_forecast_upload'}
        gdp_real_ppp_fields_dict = {'var': 'GDP', 'unit': 'GDP (2017B$ PPP)', 'job_name': 'GDP_forecast_upload'}

        # Get dfs list
        self.upload_to_sj(self.gdp_growth, gdp_growth_fields_dict, **kwargs)
        self.upload_to_sj(self.population, population_fields_dict, **kwargs)
        self.upload_to_sj(self.gdp_real_ppp, gdp_real_ppp_fields_dict, **kwargs)
        self.upload_to_sj(self.gdp_per_capita, gdp_per_capita_fields_dict, **kwargs)

    def load_inputs_from_sj(self):
        """Load macro-economic data from SJ."""

        # Get input data from external sources
        self.load_inputs()

        # Load baseline gdp growth from SJ
        conn = shooju.Connection(server=os.environ["SHOOJU_SERVER"],
                                 user=os.environ["SHOOJU_USER"],
                                 api_key=os.environ["SHOOJU_KEY"])
        sj_growth_df = conn.get_df(
            f'sid={self.sj_path}\* economic_property="global_macro_GDP_growth" scenario="Baseline"',
            fields=['iso3c', 'Country'],
            df=date(2019, 1, 1),
            max_points=-1)

        # Data cleaning
        sj_growth_df.drop(columns='series_id', inplace=True)
        sj_growth_df.sort_values(by=['country', 'date'], inplace=True)
        sj_growth_df.date = sj_growth_df.date.dt.date
        sj_growth_df = pd.pivot_table(sj_growth_df, index=['country', 'iso3c'], columns='date', values='points')
        sj_growth_df = sj_growth_df.div(100).add(1)

        # Derive real gdp values and set gdp_growth
        self.population = self.population.div(1000)
        self.gdp_global = country_name_index_formatting(self.gdp_global)

        self.gdp_const_p = sj_growth_df.join(self.gdp_global.loc[:, '1990-01-01':'2018-01-01']).T
        self.gdp_const_p.index = pd.to_datetime(self.gdp_const_p.index)
        self.gdp_const_p.sort_index(inplace=True)
        self.gdp_const_p['2018-01-01':] = self.gdp_const_p['2018-01-01':].cumprod()

        self.gdp_growth = self.gdp_const_p.pct_change() * 100

    def adjustment1(self, adj_dict):
        """
        Adjust forecast on a specific year and smooth forecasts for subsequent years for a given list of countries.
        """

        x = adj_dict['adj_value']  # % change
        T = date(adj_dict['adj_year'], 1, 1)  # effective year
        c_name = adj_dict['adj_country']  # selected country
        cut_year = T + relativedelta(years=-1)
        self.run_type = 'adjustment'

        idx = pd.IndexSlice
        adj_df = self.gdp_growth.loc[T:, idx[c_name, :]].apply(lambda p: (1 + 0.01 * x) * p / 100 + 1)

        if len(adj_df.columns.levels) > 2:
            self.gdp_const_p.update(adj_df.droplevel(['region', 'iso2c', 'sid'], axis=1))

        else:
            self.gdp_const_p.update(adj_df)

        self.gdp_const_p.loc[cut_year:, (c_name, slice(None))] = self.gdp_const_p.loc[cut_year:, (c_name, slice(None))].cumprod()
        self.adj_country_list = c_name
        description = f'{x}% growth rate adjustment in {T}'

        # Overwrite baseline values with adjustments
        self.get_other_economic_variables()
        self.get_regional_aggregations()
        self.upload_preprocessing(adj_description=description)

    def adjustment2(self, adj_dict):
        """Apply same adjustment on forecasts across all years from the effective year for a given list of countries."""

        x = adj_dict['adj_value']  # % change
        T = adj_dict['adj_year']  # effective year
        c_name = adj_dict['adj_country']  # selected country
        cut_year = T + relativedelta(years=-1)
        self.run_type = 'adjustment'

        idx = pd.IndexSlice
        adj_slice = self.gdp_growth.loc[T:, idx[c_name, :]]
        t = np.array(range(0, adj_slice.shape[0])).reshape(adj_slice.shape[0], 1)
        adj_df = (1 + 0.01 * x) ** t * adj_slice / 100 + 1

        if len(adj_df.columns.levels) > 2:
            self.gdp_const_p.update(adj_df.droplevel(['region', 'iso2c', 'sid'], axis=1))

        else:
            self.gdp_const_p.update(adj_df)

        self.gdp_const_p.loc[cut_year:, (c_name, slice(None))] = self.gdp_const_p.loc[cut_year:,
                                                                 (c_name, slice(None))].cumprod()
        self.adj_country_list = c_name
        description = f'{x}% growth rate adjustment from {T}'

        # Overwrite baseline values with adjustments
        self.get_other_economic_variables()
        self.get_regional_aggregations()
        self.upload_preprocessing(adj_description=description)

    def run_scenario_pipeline(self):
        self.load_inputs()
        self.get_forecasts()
        self.combine_dfs()
        self.get_other_economic_variables()
        self.get_regional_aggregations()
        self.upload_preprocessing()


class Scenario(BaselineScenario):
    """Class to adjust GDP forecast for a given scenario."""

    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.scenario = args_dict['scenario_name']
        self.run_type = 'scenario'

    def run_scenario_pipeline(self):
        super().load_inputs_from_sj()


class ScenarioFactory:
    @staticmethod
    def run_scenario(arg_dict, sj_path):
        if arg_dict['scenario'] == 'baseline':
            return BaselineScenario(arg_dict, sj_path)
        elif arg_dict['scenario'] == 'scenario':
            return Scenario(arg_dict)
        else:
            f"Warning: the scenario parsed doesn't exist"
