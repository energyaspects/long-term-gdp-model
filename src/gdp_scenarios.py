import numpy as np
import pandas as pd
import datetime as datetime
from get_input import DataMacro
from gdp_model import GDPModel
from static_data import rename_dict, extended_region_dict
from helper_functions_ea.data_mappings.country_region import get_region
from helper_functions_ea.shooju_helper.main import ShoojuTools
from datetime import date


def country_has_region_test(df) -> None:
    country = df.columns.get_level_values('country').to_list()
    for c in country:
        try:
            get_region(index=c, index_mapping_method="country_name", energy_product="macro", extended_dict=extended_region_dict)
        except IndexError as e:
            f'No region available for {c}, add region to the extended dictionary from static_data.py'
            raise e


class BaselineScenario:
    """Class to forecast GDP for a baseline scenario"""

    def __init__(self, args_dict):
        self.args_dict = args_dict
        self.scenario = 'Baseline'
        self.base_year = args_dict['base_year']
        self.gdp_global = None
        self.gdp_log = None
        self.gdp_growth = None
        self.population = None
        self.ppp_conv_rate = None
        self.gdp_const_p = None
        self.gdp_current_p = None
        self.gdp_per_capita = None
        self.forecasts_df = pd.DataFrame()
        self.sj_final_df = None
        self.metadata_dictionary = {}
        self.sj = ShoojuTools()
        self.is_train = args_dict['train_model']
        self.sj_path = args_dict['sj_path']
        self.train_end_year = date(args_dict['train_end_year'], 1, 1)

    def load_inputs(self):
        data_obj = DataMacro(self.base_year)
        self.gdp_global = data_obj.gdp_constant_prices
        self.gdp_log = self.gdp_global.apply(np.log)
        self.gdp_growth = self.gdp_log.diff(axis='columns') * 100
        self.population = data_obj.population
        self.ppp_conv_rate = data_obj.imp_ppp_conv_rate_base
        self.gdp_current_p = data_obj.gdp_current_prices_base

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
        full_gdp_df = self.forecasts_df.iloc[:, 1:].join(self.gdp_growth.iloc[:, 1:])
        full_gdp_df.sort_index(axis=1, inplace=True)

        # Data cleaning
        full_gdp_df.index = full_gdp_df.index.set_levels(full_gdp_df.index.levels[0].str.replace(' ', '_'), level=0)
        full_gdp_df.index = full_gdp_df.index.set_levels(full_gdp_df.index.levels[0].to_series().replace(rename_dict),
                                                         level=0)
        full_gdp_df.index = full_gdp_df.index.set_levels(
            full_gdp_df.index.levels[0].str.replace(r'St.', 'Saint', regex=True), level=0)

        self.sj_final_df = full_gdp_df.T

    def get_other_economic_variables(self):

        # get full GDP values in constant prices
        gdp_const_p = self.sj_final_df.apply(np.exp).loc['1990':]
        gdp_const_p.update(self.gdp_global)
        gdp_const_p.loc['2025':] = gdp_const_p.loc['2025':].cumprod()
        self.gdp_const_p = gdp_const_p

        # get GDP base nominal/real
        ratio = self.gdp_current_p.div(self.gdp_const_p.loc[f'{self.base_year}'])

        # Apply PPP conversion rates and calculate GDP per capita
        # 'Real GDP/POP/Implied PPP rate in 2017*(Nominal GDP in 2017/Real GDP in 2017)

        self.gdp_per_capita = self.gdp_current_p / self.population / self.ppp_conv_rate * ratio

    def upload_to_sj(self):
        # test if all countries are mapped to a region
        country_has_region_test(self.sj_final_df)

        # add region and construct sids
        self.sj_final_df.columns = pd.MultiIndex.from_tuples([(c[0], c[1],
                                                               f'{get_region(index=c[0], index_mapping_method="country_name", energy_product="macro", extended_dict=extended_region_dict)}',
                                                               f'{c[1]}_GDP_{self.scenario}') for c in
                                                              self.sj_final_df.columns])

        # create metadata dictionary with fields
        for n in range(0, len(self.sj_final_df.columns)):
            fields_additions = {
                f'{self.sj_final_df.columns[n][3]}':
                    {'country': self.sj_final_df.columns[n][0],
                     'iso3c': self.sj_final_df.columns[n][1],
                     'region': self.sj_final_df.columns[n][2],
                     'economic_property': 'global_macro_GDP',
                     'unit': 'difflog',
                     'scenario': f'{self.scenario}',
                     'source': 'IMF (NGDP_R), Macroeconomic model',
                     'lifecycle_stage': 'forecast',
                     'frequency': 'yearly'}
            }

            self.metadata_dictionary.update(fields_additions)

        # only keep column with sids and convert index to date to drop timestamp
        self.sj_final_df = self.sj_final_df.droplevel([0, 1, 2], axis=1)
        self.sj_final_df.index = self.sj_final_df.index.date

        # upload to sj
        self.sj.df_upload_wide(
            df=self.sj_final_df,
            sid_pefix=self.sj_path,
            metadata_dictionary=self.metadata_dictionary,
            name='GDP_forecasts_upload'
        )

    def run_scenario_pipeline(self):
        self.load_inputs()
        self.get_forecasts()
        self.combine_dfs()
        self.upload_to_sj()


class Scenario1(BaselineScenario):
    """Class to adjust GDP forecast for a given scenario"""

    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.scenario = 'shift_single_country_single_year'
        self.x = args_dict['scenario_params'][0]  # % change
        self.T = args_dict['scenario_params'][1]  # effective year
        self.c_iso3c = args_dict['scenario_params'][-1]  # selected country

    def apply_adjustment(self):
        adj_df = self.sj_final_df.xs(self.c_iso3c, level=1, axis=1, drop_level=False).loc[self.T:].apply(lambda p: (1 + 0.01 * self.x) * p)
        self.sj_final_df.update(adj_df)

    def run_scenario_pipeline(self):
        super().load_inputs()
        super().get_forecasts()
        super().combine_dfs()
        self.apply_adjustment()
        super().upload_to_sj()


class Scenario2(Scenario1):
    """Class to adjust GDP forecast for a given scenario"""

    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.scenario = 'shift_all_country_single_year'

    def apply_adjustment(self):
        adj_df = self.sj_final_df.loc[self.T:].apply(lambda p: (1 + 0.01 * self.x) * p)
        self.sj_final_df.update(adj_df)


class Scenario3(Scenario1):
    """Class to adjust GDP forecast for a given scenario"""

    def __init__(self, args_dict):
        super().__init__(args_dict)
        self.scenario = 'shift_single_country_all_years'

    def apply_adjustment(self):
        current_year_str = str(datetime.datetime.now().year + 1) + '-01-01'
        adj_slice = self.sj_final_df.xs(self.c_iso3c, level=1, axis=1, drop_level=False).loc[current_year_str:]
        t = np.array(range(0, adj_slice.size)).reshape(adj_slice.size, 1)
        adj_df = (1 + 0.01 * self.x) ** t * adj_slice
        self.sj_final_df.update(adj_df)


class ScenarioFactory:
    @staticmethod
    def run_scenario(arg_dict):
        if arg_dict['scenario'] == 'baseline':
            return BaselineScenario(arg_dict)
        elif arg_dict['scenario'] == 'scenario1':
            return Scenario1(arg_dict)
        elif arg_dict['scenario'] == 'scenario2':
            return Scenario2(arg_dict)
        elif arg_dict['scenario'] == 'scenario3':
            return Scenario3(arg_dict)
        else:
            f"Warning: the scenario parsed doesn't exist"

