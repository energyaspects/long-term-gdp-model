from gdp_scenarios import ScenarioFactory
import argparse
import datetime as datetime
from pathlib import Path
import os
from helper_functions_ea import check_env
from helper_functions_ea.logger_tool.main import Logger

check_env()


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
                        help="Filter training data until?")
    parser.add_argument("--fcst_end_year",
                        type=int,
                        default=2050,
                        help="Provide forecasting horizon end year")
    parser.add_argument("--train_model",
                        help="Pass if you want to train the model before predicting",
                        default=False,
                        action="store_true")
    parser.add_argument("--update_proportions",
                        help="Pass if you want to update the current other_region/region ratios",
                        default=True,
                        action="store_true")
    parser.add_argument("--PROD",
                        help="Provide Shooju path to store outputs",
                        default=False,
                        action='store_true')
    parser.add_argument("--model_param_path",
                        type=str,
                        default=Path(''),
                        help="Provide path to store models")
    parser.add_argument("--scenario",
                        default="baseline",
                        choices=["baseline", "scenario"],
                        help="Select the scenario - any scenario other than baseline will require 'scenario_name' and 'adjustment."
                             "Choices description:"
                             "Baseline: uses model output"
                             "Scenario: uses Baseline values and apply adjustments "
                             "under a defined scenario i.e. 'Covid recovery' or 'High Growth")
    parser.add_argument("--update_other_variables",
                        help="If adjustments are done in excel and uploaded to SJ- update other variables with adjusted baseline.",
                        default=True,
                        action="store_true")
    parser.add_argument("--port", default=52162)
    parser.add_argument("--scenario_name",
                        default="",
                        type=str,
                        help="Provide a meaningful scenario name i.e. 'High_growth'")

    return parser


def main():
    main_logger = Logger(name="Main Function Logger").logger
    main_logger.info(f"Start time: {datetime.datetime.now()}")

    # Load args
    parser = _parse_args().parse_args()

    if parser.PROD:
        sj_path = r'teams\long_term\long-term-modelling\macro_forecasts'
    else:
        sj_path = fr"users\{os.environ['SHOOJU_USER']}\GDP"

    args_dict = vars(parser)

    fcst_obj = ScenarioFactory.run_scenario(args_dict, sj_path)

    if not parser.update_other_variables and parser.scenario == 'baseline':
        main_logger.info("Running GDP model.")
        fcst_obj.run_scenario_pipeline()

    elif parser.update_other_variables:
        main_logger.info(f"Updating other variables with adjusted GDP growth rates from SJ")
        fcst_obj.load_macro_inputs_from_sj()
        fcst_obj.upload_adj_variables(description='Long-term Review Adjustments')

    else:
        main_logger.warning(f"Incorrect combination of parameters parsed - unable to run the code")

    main_logger.info(f"End time: {datetime.datetime.now()}")


if __name__ == '__main__':
    main()
