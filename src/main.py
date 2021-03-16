from gdp_scenarios import ScenarioFactory
import argparse
import os
from dotenv.main import load_dotenv
import datetime as datetime
from pathlib import Path

load_dotenv()


def _parse_args():
    """Parse script parameters"""
    parser = argparse.ArgumentParser(prog='gdp_model')

    parser.add_argument("--base_year",
                        type=int,
                        default=2017,
                        help="Base year for PPP?")
    parser.add_argument("--train_start_year",
                        type=int,
                        default=1980,
                        help="Filter training data from?")
    parser.add_argument("--train_end_year",
                        type=int,
                        default=2025,
                        help="Filter training data until?")
    parser.add_argument("--fcst_end_year",
                        type=int,
                        default=2050,
                        help="Provide forecasting horizon end year")
    parser.add_argument("--train_model",
                        help="Pass if you want to train the model before predicting",
                        default=False,
                        action="store_true"
                        )
    parser.add_argument("--sj_path",
                        type=str,
                        default=f"users\{os.environ['SHOOJU_USER']}\GDP",
                        help="Provide Shooju path to store outputs")
    parser.add_argument("--model_param_path",
                        type=str,
                        default=Path(''),
                        help="Provide path to store models")
    parser.add_argument("--scenario",
                        default="baseline",
                        choices=["baseline", "scenario"],
                        help="Select the scenario - any scenario other than baseline will require 'scenario_params."
                             "Choices description:"
                             "Baseline: uses model output"
                             "Scenario1: % change for one country and one year"
                             "Scenario2: % change for all countries and one year"
                             "Scenario3: % change for one countries and all years")
    parser.add_argument("--adjustment",
                        default=False,
                        action="store_true")
    parser.add_argument("--scenario_params",
                        nargs=3,
                        default=[0.01, 'ALL', 'CHN'],
                        type=list,
                        help="Provide percentage change, effective year and country iso3c."
                             "Params description:"
                             "percentage change: adjustment percentage applied to the GDP forecast"
                             "effective year: year to apply % change - use 'ALL' for scenario3"
                             "country iso3c: iso3c of country used in scenario - use 'ALL' for scenario2")

    parser.add_argument("--port", default=52162)
    parser.add_argument('--scenario_name', default='COVID_Recovery')

    return parser


def main(*adj_list):
    print(datetime.datetime.now())

    # Load args
    parser = _parse_args().parse_args()
    #
    # # test validity of args parsed
    #
    # if parser.scenario == 'scenario2':
    #     assert parser.scenario_params[-1] == 'ALL', 'incorrect iso3c value parsed for selected scenario - change ##'
    #
    # if parser.scenario == 'scenario3':
    #     assert parser.scenario_params[-2] == 'ALL', 'incorrect year value parsed for selected scenario - change ##'

    args_dict = vars(parser)

    if parser.adjustment or parser.scenario == 'scenario':
        fcst_obj = ScenarioFactory.run_scenario(args_dict)
        fcst_obj.run_scenario_pipeline()

        for i in range(0, len(adj_list)):
            if adj_list[i][0] == 'adj_1':
                fcst_obj.ajustment1(adj_list[i])
            elif adj_list[i][0] == 'adj_2':
                fcst_obj.ajustment2(adj_list[i])
            elif adj_list[i][0] == 'adj_3':
                fcst_obj.ajustment3(adj_list[i])

    else:
        # Run GDP model for a given scenario
        fcst_obj = ScenarioFactory.run_scenario(args_dict)
        fcst_obj.run_scenario_pipeline()

    print(datetime.datetime.now())

#
# if __name__ == '__main__':
#     main()

