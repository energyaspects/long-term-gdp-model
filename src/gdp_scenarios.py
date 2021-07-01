import numpy as np
import pandas as pd
import os
from get_input import DataMacro
from gdp_model import GDPModel
from static_data import extended_iso_dict, prop_unit_dict
from helper_functions_ea.data_mappings.country_iso import get_iso
from helper_functions_ea.shooju_helper.main import ShoojuTools
from helper_functions_ea.logger_tool.main import Logger
from datetime import date
from static_functions import (map_country_to_region, get_region_aggregates_multi, country_name_index_formatting,
                              get_sub_region_aggregates, get_region_aggregates_standard)

#
# Class for scenario modelling
#


class BaselineScenario:
    """Class to forecast GDP for a baseline scenario."""

    proportions = None
    gdp_global = None  # Historical real GDP values in national currency
    gdp_log = None  # Historical real GDP log values in national currency
    population = None  # UN population medium scenario estimates (Historic & Forecast)
    ppp_conv_rate = None  # Purchasing power parity values ($ 2017)
    gdp_const_p = None  # Real GDP forecasts in national currency
    gdp_current_p = None  # 2017 Nominal GDP values in national currency
    gdp_per_capita = None  # GDP per capita ($2017 PPP) (Historic & Forecast)
    gdp_growth = None  # Real GDP growth rate historic & forecast
    gdp_real_ppp = None  # Real PPP GDP historic & forecast ($2017 PPP)
    imf_fcst_end_year = None
    cols_filter = None
    adj_country_list = []
    forecasts_df = pd.DataFrame()

    def __init__(self, args_dict, sj_path):
        self.args_dict = args_dict
        self.scenario = 'Baseline'
        self.base_year = args_dict['base_year']
        self.sj = ShoojuTools()
        self.is_train = args_dict['train_model']
        self.update_proportions = args_dict['update_proportions']
        self.sj_path = sj_path
        self.train_end_year = date(args_dict['train_end_year'], 1, 1) if args_dict['train_end_year'] else None
        self.run_type = 'baseline'
        self.country_region_mapping = pd.read_excel(os.environ["GROUPING_SHEET_PATH"], sheet_name='new_list',
                                                    keep_default_na=False)
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
        self.cols_filter = self.population.columns.intersection(
            self.ppp_conv_rate.columns.levels[0])  # get common countries between UNPOP and IMF
        self.imf_fcst_end_year = self.gdp_global.columns.max()  # most recent IMF estimate

    def load_proportions(self):
        """load proportion data from SJ."""

        proportions = (
            self.sj.sj.get_df(fr'sid={self.sj_path}\* economic_property=global_macro_ratio', fields=['*'], max_points=-1)
            .drop(columns='series_id')
            .sort_values(by=['region', 'date'])
            )
        self.proportions = pd.pivot_table(proportions, index=['date'], columns='region', values='points')

    def load_macro_inputs_from_sj(self):
        """Load macro-economic data from SJ."""

        # Get input data from external sources
        self.load_inputs()

        # Load baseline gdp growth from SJ
        sj_growth_df = (self.sj.sj.get_df(
            fr'sid={self.sj_path}\* economic_property=global_macro_GDP_growth scenario={self.scenario} region=(Africa,Europe,Asia-Pacific,Latin_America,FSU,Middle_East,North_America)',
            fields=['iso3c', 'Country', 'type'],
            df=date(self.imf_fcst_end_year.year + 1, 1, 1),
            max_points=-1)
                        .drop(columns='series_id')
                        .sort_values(by=['country', 'date'])
                        )

        # Data cleaning
        self.adj_country_list = list(sj_growth_df.query('type == "Adj"').country.unique())
        sj_growth_df.date = sj_growth_df.date.dt.date
        sj_growth_df = pd.pivot_table(sj_growth_df, index=['country', 'iso3c'], columns='date', values='points')
        sj_growth_df = sj_growth_df.div(100).add(1)

        # Derive real gdp values and set gdp_growth
        self.population = self.population.div(1000)
        self.gdp_global = country_name_index_formatting(self.gdp_global)

        self.gdp_const_p = sj_growth_df.join(self.gdp_global.loc[:, '1990-01-01':]).T
        self.gdp_const_p.index = pd.to_datetime(self.gdp_const_p.index)
        self.gdp_const_p.sort_index(inplace=True)
        self.gdp_const_p[f'{self.imf_fcst_end_year.year}':] = self.gdp_const_p[
                                                              f'{self.imf_fcst_end_year.year}':].cumprod()

        self.gdp_growth = self.gdp_const_p.pct_change() * 100

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
        # Use IMF historical GDP/c values for Russia (CEPII too high)-other countries IMF historical data not available
        idx = pd.IndexSlice
        imf_hist_per_cap_russia = [21.433, 20.296]
        self.gdp_per_capita.loc['1990':'1991', idx[:, 'RUS']] = imf_hist_per_cap_russia

        # Filter other countries with NANs

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
        o_gdp, c_gdp = get_sub_region_aggregates(self.gdp_real_ppp)
        o_pop, c_pop = get_sub_region_aggregates(self.population)

        c_gdp_per_c = c_gdp.div(c_pop)

        if self.update_proportions:
            o_gdp_per_c = o_gdp.div(o_pop)
            self.proportions = o_gdp_per_c.div(c_gdp_per_c.values)
            self.get_prop_metadata_dict(df=self.proportions)

        else:
            try:
                self.load_proportions()
                assert c_gdp_per_c.shape == self.proportions.shape, 'Incorrect shape - values are not aligned'
                o_gdp_per_c = c_gdp_per_c.mul(self.proportions.values)
                o_gdp_per_c.columns = o_pop.columns
                o_gdp = o_pop.mul(o_gdp_per_c)

            except:
                self.scenario_logger.warning(
                    "Proportions SIDs not available at this location - set 'update_proportions' == True and re-run code.")

        r_gdp = get_region_aggregates_multi(df=self.gdp_real_ppp, other_df=pd.concat([c_gdp, o_gdp], axis=1))
        r_pop = get_region_aggregates_standard(self.population)
        r_gdp_per_c = r_gdp.div(r_pop)

        # Concat existing dfs with other regions and regions
        self.population = pd.concat([self.population, o_pop, r_pop], axis=1)
        self.gdp_real_ppp = pd.concat([self.gdp_real_ppp, o_gdp, r_gdp], axis=1)
        self.gdp_per_capita = pd.concat([self.gdp_per_capita, o_gdp_per_c, r_gdp_per_c], axis=1)
        self.gdp_growth = self.gdp_real_ppp.pct_change() * 100

    def upload_preprocessing(self, **kwargs):
        """Create property specific metadata fields."""

        gdp_growth_fields_dict = {'var': 'GDP_growth', 'unit': 'difflog', 'job_name': 'GDP_growth_forecast_upload'}
        population_fields_dict = {'var': 'POP', 'unit': 'Total population (thousands)',
                                  'job_name': 'Pop_forecast_upload'}
        gdp_per_capita_fields_dict = {'var': 'GDP_Capita', 'unit': 'GDP/capita (2017k$ PPP)',
                                      'job_name': 'GDP_per_capita_forecast_upload'}
        gdp_real_ppp_fields_dict = {'var': 'GDP', 'unit': 'GDP (2017B$ PPP)', 'job_name': 'GDP_forecast_upload'}

        # Get dfs list
        self.get_macro_metadata_dict(self.gdp_growth, gdp_growth_fields_dict, **kwargs)
        self.get_macro_metadata_dict(self.population, population_fields_dict, **kwargs)
        self.get_macro_metadata_dict(self.gdp_real_ppp, gdp_real_ppp_fields_dict, **kwargs)
        self.get_macro_metadata_dict(self.gdp_per_capita, gdp_per_capita_fields_dict, **kwargs)

    def get_macro_metadata_dict(self, df, df_dict, **kwargs):
        """Create metadata and upload macro-economic data to SJ."""

        self.scenario_logger.info(f"Uploading {df_dict['var']} to SJ")

        # Get list of modelled series (i.e. model output not aggregate) & published series
        series_published = self.country_region_mapping.Published.to_list() + list(
            self.country_region_mapping.Region_iso3c.unique() + 'W')

        # Add iso2c and construct sids
        df.columns = pd.MultiIndex.from_tuples([(c[0], c[1], c[2], c[3],
                                                 f'{get_iso(c[0], extended_dict=extended_iso_dict)}',
                                                 f'{c[1]}_{df_dict["var"]}_{self.scenario}') for c in
                                                df.columns])

        df.columns.set_names(['country', 'iso3c', 'region', 'other_region', 'iso2c', 'sid'], inplace=True)

        # create metadata dictionary with fields
        description = kwargs.get('adj_description')

        macro_metadata_dictionary = {}

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

            macro_metadata_dictionary.update(fields_additions)

        self.upload_to_sj(df=df,
                          drop_level=['country', 'iso3c', 'region', 'other_region', 'iso2c'],
                          metadata_dict=macro_metadata_dictionary,
                          job_name=df_dict['job_name'])

    def get_prop_metadata_dict(self, df):
        """Create metadata and upload GDP/Capita regional ratios to SJ."""

        self.scenario_logger.info("Uploading ratios to SJ")

        df.columns = pd.MultiIndex.from_tuples([(c[0], c[1], f'global_macro_ratio_{c[1]}') for c in df.columns])

        prop_metadata_dictionary = {}

        for n in range(0, len(df.columns)):
            fields_additions = {
                f'{df.columns[n][2]}':
                    {'region': df.columns[n][0],
                     'economic_property': 'global_macro_ratio',
                     'description': 'Other region GDP/capita proportion',
                     'unit': prop_unit_dict[df.columns[n][0]]}
            }
            prop_metadata_dictionary.update(fields_additions)

        self.upload_to_sj(df=df,
                          drop_level=[0, 1],
                          metadata_dict=prop_metadata_dictionary,
                          job_name='Uploading GDP/capita ratios')

    def upload_to_sj(self, df, drop_level, metadata_dict, job_name):
        # Only keep column with sids and convert index to date to drop timestamp
        df = df.droplevel(drop_level, axis=1)
        df.index = df.index.date

        # Upload to sj
        self.sj.df_upload_wide(
            df=df,
            sid_pefix=self.sj_path,
            metadata_dictionary=metadata_dict,
            name=job_name
        )

    def upload_adj_variables(self, description):
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

    def __init__(self, args_dict, sj_path):
        super().__init__(args_dict, sj_path)
        self.scenario = args_dict['scenario_name']
        self.run_type = 'scenario'

    def run_scenario_pipeline(self):
        super().load_macro_inputs_from_sj()


class ScenarioFactory:
    @staticmethod
    def run_scenario(arg_dict, sj_path):
        if arg_dict['scenario'] == 'baseline':
            return BaselineScenario(arg_dict, sj_path)
        elif arg_dict['scenario'] == 'scenario':
            return Scenario(arg_dict, sj_path)
        else:
            raise Exception("Warning: the scenario parsed doesn't exist")


if __name__ == '__main__':
    print(extended_iso_dict)
