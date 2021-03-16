import numpy as np
import pandas as pd
import os
from get_input import DataMacro
from gdp_model import GDPModel
from static_data import rename_dict, extended_region_dict, extended_iso_dict
from helper_functions_ea.data_mappings.country_region import get_region
from helper_functions_ea.data_mappings.country_iso import get_iso
from helper_functions_ea.shooju_helper.main import ShoojuTools
from datetime import date
import shooju


def country_has_region_test(df) -> None:
    country = df.columns.get_level_values(0).to_list()
    for c in country:
        try:
            get_region(index=c, index_mapping_method="country_name", energy_product="macro",
                       extended_dict=extended_region_dict)
        except IndexError as e:
            f'No region available for {c}, add region to the extended dictionary from static_data.py'
            raise e


class BaselineScenario:
    """Class to forecast GDP for a baseline scenario"""

    def __init__(self, args_dict):
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
        self.sj_path = args_dict['sj_path']
        self.train_end_year = date(args_dict['train_end_year'], 1, 1)
        self.cols_filter = None
        self.adj_country_list = []
        self.run_type = 'baseline'

    def load_inputs(self):

        data_obj = DataMacro(self.base_year)
        self.gdp_global = data_obj.gdp_constant_prices
        self.gdp_log = self.gdp_global.apply(np.log)
        self.population = data_obj.population
        self.ppp_conv_rate = data_obj.imp_ppp_conv_rate_base
        self.gdp_current_p = data_obj.gdp_current_prices_base
        self.cols_filter = self.population.columns.intersection(self.ppp_conv_rate.columns.levels[0])

    def get_forecasts(self):

        groups = self.gdp_log.groupby('iso3c')
        group_keys = list(groups.groups.keys())

        for key in group_keys:
            target = groups.get_group(key).T
            target.dropna(inplace=True)
            target = target[:self.train_end_year]

            mdl_obj = GDPModel(target, key, self.args_dict)

            if self.is_train:
                mdl_obj.train_predict_pipeline()

            else:
                mdl_obj.predict_pipeline()

            # self.forecasts_df = self.forecasts_df.append(mdl_obj.predictions_log.T.set_index(target.columns))
            self.forecasts_df = self.forecasts_df.append(mdl_obj.predictions_log.T.set_index(target.columns))

    def combine_dfs(self):

        # Join IMF real GDP values with predicted growth rates
        growth_rate_df = self.forecasts_df.div(100).add(1)
        full_gdp_df = self.gdp_global.iloc[:, 1:].join(growth_rate_df.iloc[:, 1:])
        full_gdp_df.columns = pd.to_datetime(full_gdp_df.columns)

        # Data cleaning
        full_gdp_df.index = full_gdp_df.index.set_levels(full_gdp_df.index.levels[0].str.replace(' ', '_'), level=0)
        full_gdp_df.index = full_gdp_df.index.set_levels(full_gdp_df.index.levels[0].to_series().replace(rename_dict),
                                                         level=0)
        full_gdp_df.index = full_gdp_df.index.set_levels(
            full_gdp_df.index.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)

        # Filter countries and derive real GDP forecasts
        self.gdp_const_p = full_gdp_df.T.loc['1990':, self.cols_filter]
        self.gdp_const_p['2025':] = self.gdp_const_p['2025':].cumprod()

        # Transform population data
        self.population = self.population.div(1000)

    def get_other_economic_variables(self):

        # get growth rate df
        self.gdp_growth = self.gdp_const_p.pct_change() * 100

        # get GDP base nominal/real
        ratio = self.gdp_current_p[self.cols_filter].div(self.gdp_const_p.loc[f'{self.base_year}'])

        # Apply PPP conversion rates and calculate GDP per capita
        self.population = self.population[self.cols_filter]
        self.population.columns = pd.MultiIndex.from_tuples([(c[0], c[1]) for c in self.gdp_const_p.columns])

        self.gdp_per_capita = (self.gdp_const_p
                               .div(self.population)
                               .div(self.ppp_conv_rate[self.cols_filter].values)
                               .mul(ratio.values))

        self.gdp_real_ppp = self.gdp_per_capita.mul(self.population)

    def upload_to_sj(self, df, df_dict, **kwargs):

        # test if all countries are mapped to a region
        country_has_region_test(df)

        # add region and construct sids
        df.columns = pd.MultiIndex.from_tuples([(c[0], c[1],
                                                 f'{get_region(index=c[0], index_mapping_method="country_name", energy_product="macro", extended_dict=extended_region_dict)}',
                                                 f'{get_iso(c[0], extended_dict=extended_iso_dict)}',
                                                 f'{c[1]}_{df_dict["var"]}_{self.scenario}') for c in
                                                df.columns])

        df.columns.set_names(['country', 'iso3c', 'region', 'iso2c', 'sid'], inplace=True)

        # create metadata dictionary with fields
        description = kwargs.get('adj_description')

        for n in range(0, len(df.columns)):
            fields_additions = {
                f'{df.columns[n][4]}':
                    {'country': df.columns[n][0],
                     'country_iso': df.columns[n][3],
                     'iso3c': df.columns[n][1],
                     'region': df.columns[n][2],
                     'economic_property': f'global_macro_{df_dict["var"]}',
                     'unit': df_dict['unit'],
                     'scenario': f'{self.scenario}',
                     'source': 'IMF, Macroeconomic model',
                     'lifecycle_stage': 'forecast',
                     'frequency': 'yearly'}
            }
            if df.columns[n][0] in self.adj_country_list:
                extra_fields = {
                    'type': "Adj",
                    'description': description
                }
                fields_additions[df.columns[n][4]].update(extra_fields)

            if self.run_type == 'baseline':
                extra_fields = {
                    'type': "Raw",
                    'description': f'Macroeconomic Data for {self.scenario} scenario'
                }
                fields_additions[df.columns[n][4]].update(extra_fields)

            self.metadata_dictionary.update(fields_additions)

        # only keep column with sids and convert index to date to drop timestamp
        df = df.droplevel(['country', 'iso3c', 'region', 'iso2c'], axis=1)
        df.index = df.index.date

        # upload to sj
        self.sj.df_upload_wide(
            df=df,
            sid_pefix=self.sj_path,
            metadata_dictionary=self.metadata_dictionary,
            name=df_dict['job_name']
        )

    def upload_preprocessing(self, **kwargs):

        gdp_growth_fields_dict = {'var': 'GDP_growth', 'unit': 'difflog', 'job_name': 'GDP_growth_forecast_upload'}
        population_fields_dict = {'var': 'POP', 'unit': 'Total population (thousands)', 'job_name': 'Pop_forecast_upload'}
        gdp_per_capita_fields_dict = {'var': 'GDP_Capita', 'unit': 'GDP/capita (2017k$ PPP)', 'job_name': 'GDP_per_capita_forecast_upload'}
        gdp_real_ppp_fields_dict = {'var': 'GDP', 'unit': 'GDP (2017B$ PPP)', 'job_name': 'GDP_forecast_upload'}

        # get dfs list
        self.upload_to_sj(self.gdp_growth, gdp_growth_fields_dict, **kwargs)
        self.upload_to_sj(self.population, population_fields_dict, **kwargs)
        self.upload_to_sj(self.gdp_real_ppp, gdp_real_ppp_fields_dict, **kwargs)
        self.upload_to_sj(self.gdp_per_capita, gdp_per_capita_fields_dict, **kwargs)

    def adjustment1(self, adj_list):
        """ One country one year """

        x = adj_list[-1]  # % change
        T = adj_list[-2]  # effective year
        c_name = adj_list[-3]  # selected country
        cut_year = f'{int(T)-1}'
        self.run_type = 'adjustment'

        adj_df = self.gdp_growth.xs(c_name, level=0, axis=1, drop_level=False).loc[T:].apply(
            lambda p: (1 + 0.01 * x) * p / 100 + 1)

        try:
            self.gdp_const_p.update(adj_df.droplevel(['region', 'iso2c', 'sid'], axis=1))
        except Exception as e:
            pass

        try:
            self.gdp_const_p.update(adj_df)
        except Exception as e:
            raise e

        self.gdp_const_p.loc[cut_year:, (c_name, slice(None))] = self.gdp_const_p.loc[cut_year:, (c_name, slice(None))].cumprod()
        self.adj_country_list = [c_name]
        description = f'{x}% growth rate adjustment in {T}'

        # Overwrite baseline values with adjustments
        self.get_other_economic_variables()
        self.upload_preprocessing(adj_description=description)

    def adjustment2(self, adj_list):
        """ All country single year """
        x = adj_list[-1]  # % change
        T = adj_list[-2]  # effective year
        cut_year = f'{int(T) - 1}'
        self.run_type = 'adjustment'
        adj_df = self.gdp_growth.loc[T:].apply(lambda p: (1 + 0.01 * x) * p / 100 + 1)

        try:
            self.gdp_const_p.update(adj_df.droplevel(['region', 'iso2c', 'sid'], axis=1))
        except Exception:
            pass

        try:
            self.gdp_const_p.update(adj_df)
        except Exception as e:
            raise e

        self.gdp_const_p.loc[cut_year:, :] = self.gdp_const_p.loc[cut_year:, :].cumprod()
        self.adj_country_list = self.gdp_const_p.columns.get_level_values('country').to_list()
        description = f'{x}% real GDP growth rate adjustment in {T}'

        # Overwrite baseline values with adjustments
        self.get_other_economic_variables()
        self.upload_preprocessing(adj_description=description)

    def adjustment3(self, adj_list):
        """ One country all years """
        x = adj_list[-1]  # % change
        T = adj_list[-2]  # effective year
        c_name = adj_list[-3]  # selected country
        cut_year = f'{int(T) - 1}'
        self.run_type = 'adjustment'
        #current_year_str = str(datetime.datetime.now().year + 1) + '-01-01'

        adj_slice = self.gdp_growth.xs(c_name, level=0, axis=1, drop_level=False).loc[T:]
        t = np.array(range(0, adj_slice.size)).reshape(adj_slice.size, 1)
        adj_df = (1 + 0.01 * x) ** t * adj_slice / 100 + 1

        try:
            self.gdp_const_p.update(adj_df.droplevel(['region', 'iso2c', 'sid'], axis=1))
        except Exception as e:
            pass

        try:
            self.gdp_const_p.update(adj_df)
        except Exception as e:
            raise e

        self.gdp_const_p.loc[cut_year:, (c_name, slice(None))] = self.gdp_const_p.loc[cut_year:,
                                                                 (c_name, slice(None))].cumprod()
        self.adj_country_list = [c_name]
        description = f'{x}% growth rate adjustment from {T}'

        # Overwrite baseline values with adjustments
        self.get_other_economic_variables()
        self.upload_preprocessing(adj_description=description)

    def run_scenario_pipeline(self):
        self.load_inputs()
        self.get_forecasts()
        self.combine_dfs()
        self.get_other_economic_variables()
        self.upload_preprocessing()


class Scenario(BaselineScenario):
    """Class to adjust GDP forecast for a given scenario"""

    def __init__(self, args_dict, scenario_name):
        super().__init__(args_dict)
        # self.scenario = args_dict['scenario_name']
        self.scenario = scenario_name

    def load_inputs(self):
        super().load_inputs()

        # load baseline gdp growth from SJ
        conn = shooju.Connection(server=os.environ["SHOOJU_SERVER"],
                                 user=os.environ["SHOOJU_USER"],
                                 api_key=os.environ["SHOOJU_KEY"])

        sj_growth_df = conn.get_df('sid=users\emilie.allen\GDP\* economic_property="global_macro_GDP_growth" scenario="Baseline"',
                         fields=['iso3c', 'Country'],
                         max_points=-1)

        # data cleaning
        sj_growth_df.drop(columns='series_id', inplace=True)
        sj_growth_df.sort_values(by=['country', 'date'], inplace=True)
        sj_growth_df.date = sj_growth_df.date.dt.date
        sj_growth_df = pd.pivot_table(sj_growth_df, index=['country', 'iso3c'], columns='date', values='points')
        sj_growth_df = sj_growth_df.div(100).add(1)

        # derive real gdp values and set gdp_growth
        self.population = self.population.div(1000)

        self.gdp_const_p = sj_growth_df.join(self.gdp_global[['1990-01-01']]).T
        self.gdp_const_p.index = pd.to_datetime(self.gdp_const_p.index)
        self.gdp_const_p.sort_index(inplace=True)
        self.gdp_const_p = self.gdp_const_p.cumprod()

        self.gdp_growth = self.gdp_const_p.pct_change() * 100

    def run_scenario_pipeline(self):
        self.load_inputs()


class ScenarioFactory:
    @staticmethod
    def run_scenario(arg_dict):
        if arg_dict['scenario'] == 'baseline':
            return BaselineScenario(arg_dict)
        elif arg_dict['scenario'] == 'scenario':
            return Scenario(arg_dict)
        else:
            f"Warning: the scenario parsed doesn't exist"
