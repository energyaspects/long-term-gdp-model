import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter
import pmdarima as pm
import joblib
from pathlib import Path
from datetime import date
from dateutil.relativedelta import relativedelta


class GDPModel:

    def __init__(self, target, key, args_dict):
        self.target = target
        self.predictions = None
        self.predictions_log = None
        self.model = None
        self.train_end_year = date(args_dict['train_end_year'], 1, 1)
        self.fcst_end_year = date(args_dict['fcst_end_year'], 1, 1)
        self.delta = relativedelta(self.fcst_end_year, self.train_end_year).years + 1
        self.index = pd.date_range(start=self.train_end_year, periods=self.delta, freq='YS')
        self.output_directory = args_dict['model_param_path']
        self.key = key

    def model_train(self):
        # apply HP filter and get trend
        cycle, trend = hpfilter(self.target, lamb=100)

        # train set
        train = trend[:-1]

        # fit an auto_arima model
        auto_arima_model = pm.auto_arima(train,
                                         start_p=1,
                                         start_q=1,
                                         start_P=1,
                                         start_Q=1,
                                         max_d=2,
                                         suppress_warnings=True,
                                         error_action='ignore',
                                         seasonal=False,
                                         stepwise=False,
                                         with_intercept=True
                                         )

        self.model = auto_arima_model
        self.save_model()

    def get_predictions(self):
        # get predictions
        preds, conf_int = self.model.predict(n_periods=self.delta, return_conf_int=True)

        # create dfs
        self.predictions = pd.DataFrame(data={self.key: preds}, index=self.index)
        self.predictions_log = self.predictions.diff() * 100

    def save_model(self):
        subfolder = self.output_directory / "model_params"
        Path(subfolder).mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, subfolder / f'{self.key}_model.pkl')

    def load_model(self):
        try:
            subfolder = self.output_directory / "model_params"
            # subfolder = 'C:/Users/emilie.allen/PycharmProjects/long - term - gdp - model/src/model_params'
            # self.model = joblib.load(f'C:/Users/emilie.allen/PycharmProjects/long-term-gdp-model/src/model_params/{self.key}_model.pkl')
            self.model = joblib.load(subfolder/f'{self.key}_model.pkl')

        except Exception as e:
            f"No model found - run train_predict_pipeline() instead"
            raise e

    def train_predict_pipeline(self):
        self.model_train()
        self.get_predictions()

    def predict_pipeline(self):
        self.load_model()
        self.get_predictions()
