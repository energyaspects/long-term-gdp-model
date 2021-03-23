from pathlib import Path
import os
from dotenv.main import load_dotenv
import datetime

load_dotenv()

model_arguments_dict = {"base_year": 2017,
                        "train_start_year": 1980,
                        "train_end_year": 2025,
                        "fcst_end_year": 2050,
                        "train_model": False,
                        "sj_path": f"users\{os.environ['SHOOJU_USER']}\GDP",
                        "model_param_path": Path(''),
                        "scenario": "baseline",
                        "adjustment": False,
                        }

model_adj_arguments_dict = {"base_year": 2017,
                            "train_start_year": 1980,
                            "train_end_year": 2025,
                            "fcst_end_year": 2050,
                            "train_model": False,
                            "sj_path": f"users\{os.environ['SHOOJU_USER']}\GDP",
                            "model_param_path": Path(''),
                            "scenario": "baseline",
                            "adjustment": False,
                            "scenario_name": ''
                            }