from config import *
from data_preparation import get_prophet_dataset

import gc
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

from itertools import product
from os.path import exists
from prophet import Prophet


# Training and testing Prophet models
def train_test_prophet(dataframes_dict: dict):
    model_metrics_df = None
    for object_name in OBJECTS:
        print("Trying Prophet model with the object %s" % object_name)
        model = Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=True, growth='logistic')

        # Get dataset according to the object
        target = None
        if object_name in GENERATION_OBJECTS:
            target = GEN_TARGET_COLUMN
        elif object_name in CONSUMPTION_OBJECTS:
            target = CON_TARGET_COLUMN
        else:
            raise ValueError('Unknown target!')
        dataset = get_prophet_dataset(dataframes_dict[object_name].copy(), TESTING_INTERVALS, target)

        for column in dataset.df_train.columns:
            if column not in ['y', 'ds', 'cap', 'floor']:
                model.add_regressor(column)

        model.fit(dataset.df_train)

        # Test the model with training data (in-sample)
        train_result_df = model.predict(dataset.df_train)
        train_result_df = pd.merge(train_result_df[['ds', 'yhat']], dataset.df_train, on='ds')

        # Test the model with testing data (out of sample)
        test_results_df = model.predict(dataset.df_test)
        test_results_df = pd.merge(test_results_df[['ds', 'yhat']], dataset.df_test, on='ds')

        y_train_real = train_result_df['y'].tolist()
        y_train_pred = train_result_df['yhat'].tolist()
        y_test_real = test_results_df['y'].tolist()
        y_test_pred = test_results_df['yhat'].tolist()

        model_dict = {
            'model': [target],
            'object': [object_name],
            'train_mae': [metrics.mean_absolute_error(y_train_real, y_train_pred)],
            'train_mse': [metrics.mean_squared_error(y_train_real, y_train_pred)],
            'train_r2': [metrics.r2_score(y_train_real, y_train_pred)],
            'train_rmse': [metrics.mean_squared_error(y_train_real, y_train_pred, squared=False)],
            'train_mape': [metrics.mean_absolute_percentage_error(y_train_real, y_train_pred)],
            'test_mae': [metrics.mean_absolute_error(y_test_real, y_test_pred)],
            'test_mse': [metrics.mean_squared_error(y_test_real, y_test_pred)],
            'test_r2': [metrics.r2_score(y_test_real, y_test_pred)],
            'test_rmse': [metrics.mean_squared_error(y_test_real, y_test_pred, squared=False)],
            'test_mape': [metrics.mean_absolute_percentage_error(y_test_real, y_test_pred)]
        }

        model_metrics_df = pd.DataFrame.from_dict(model_dict) if model_metrics_df is None \
            else pd.concat([model_metrics_df, pd.DataFrame.from_dict(model_dict)])

        # Plot the comparison with the testing dataset
        test_len = len(y_test_pred)
        x = range(0, test_len)
        plt.figure(figsize=(30, 10), dpi=160)
        plt.plot(x, y_test_real, alpha=0.5, label='Real Values')
        plt.plot(x, y_test_pred, alpha=0.5, label='Predicted Values')

        # Line to illustrate the end of different seasons
        for line in range(1, 4):
            plt.axvline(x=line * test_len / 4)

        plt.title("Comparison (Model Prophet %s)" % (object_name))
        plt.legend(loc="upper left", fontsize="x-large")
        plt.savefig(PLOT_FILES_LOC + "model_prophet_%s.png" % (object_name))

        # Cleanup plotting environment after each loop
        plt.close('all')
        plt.cla()
        plt.clf()

    model_metrics_df.to_csv(PROPHET_RESULTS_FILE, mode='a', index=False, header=not (exists(PROPHET_RESULTS_FILE)))
