import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import tensorflow as tf
import time

from itertools import product
from os.path import exists
from pvlib.location import Location

from data_preparation import *
from models import *


# Data files
_DATA_FILES_LOC = "../Data-Files/"
_WEATHER_DATA = _DATA_FILES_LOC + "weather_data.csv"
_SOLAR_5K_DATA = _DATA_FILES_LOC + "solar_5k_generation.csv"
_SOLAR_30K_DATA = _DATA_FILES_LOC + "solar_30k_generation.csv"
_CONSUMPTION_DATA = _DATA_FILES_LOC + "consumption_data.csv"
_DATETIME_COLS = ['datetime_unix', 'datetime_local']

_RESULTS_FILE = 'results.csv'

# Location of a Solar Power plants
_LOCATION = Location(54.925976, 25.371658, tz='Europe/Vilnius')

# Target columns
_GEN_TARGET_COLUMN = 'generation'
_CON_TARGET_COLUMN = 'consumption'

# Testing intervals for dividing data into training and testing
_DATE_FORMAT = '%Y-%m-%d'
_TESTING_INTERVALS = [
    (pd.to_datetime('2021-03-20', format=_DATE_FORMAT), pd.to_datetime('2021-03-22', format=_DATE_FORMAT)),
    (pd.to_datetime('2021-07-10', format=_DATE_FORMAT), pd.to_datetime('2021-07-12', format=_DATE_FORMAT)),
    (pd.to_datetime('2021-10-28', format=_DATE_FORMAT), pd.to_datetime('2021-10-30', format=_DATE_FORMAT)),
    (pd.to_datetime('2022-02-26', format=_DATE_FORMAT), pd.to_datetime('2022-02-28', format=_DATE_FORMAT))
]

_GENERATION_OBJECTS = ["solar_30k", "solar_5k"]
_CONSUMPTION_OBJECTS = ["1", "5"]

# Parameter tuning
_OBJECTS = ['solar_30k', 'solar_5k', '1', '5']
_OPTIMIZERS = ['Adam', 'RMSProp']
_LOSS_FUNCTIONS = ['MAE', 'MSE', 'MAPE']
_LEARNING_RATES = [0.001, 0.0001]
_ARCHITECTURES = ['FEEDFORWARD', 'LSTM', 'GRU']
_SEQUENCE_LENGHTS = [12, 24, 36]
_NUMBER_OF_LAYERS = [2, 4]

# 4 Different training steps, model will be trained for 400 (50 + 50 + 100 + 200) epochs in total
# _NUMBER_OF_EPOCHS = [50, 50, 100, 200]
_NUMBER_OF_EPOCHS = [2, 2, 4, 8]


if __name__ == '__main__':
    weather_df = prepare_weather_data(pd.read_csv(_WEATHER_DATA, parse_dates=_DATETIME_COLS), _LOCATION)
    consumption_df = pd.read_csv(_CONSUMPTION_DATA, parse_dates=['datetime_local'])

    # Dictionary for storing dataframes of the objects
    dataframes_dict = {
        "solar_30k": pd.read_csv(_SOLAR_30K_DATA, parse_dates=_DATETIME_COLS),
        "solar_5k": pd.read_csv(_SOLAR_5K_DATA, parse_dates=_DATETIME_COLS),
        "1": consumption_df.loc[consumption_df['object_id'] == 1].drop('object_id', axis=1),
        "5": consumption_df.loc[consumption_df['object_id'] == 5].drop('object_id', axis=1)
    }

    # Dictionary for info about fetched data
    fetch_data_dict = {
        'Data type': ["Weather"],
        'First timestamp': [weather_df['datetime_local'].min()],
        'Last timestamp': [weather_df['datetime_local'].max()],
        'Number of rows': [len(weather_df.index)]
    }

    # Dictionary for info about prepared data
    prep_data_dict = {
        'Object': [],
        'First timestamp': [],
        'Last timestamp': [],
        'Number of rows': []
    }

    for obj, dataframe in dataframes_dict.items():
        fetch_data_dict['Data type'].append(obj)
        fetch_data_dict['First timestamp'].append(dataframe['datetime_local'].min())
        fetch_data_dict['Last timestamp'].append(dataframe['datetime_local'].max())
        fetch_data_dict['Number of rows'].append(len(dataframe.index))

        if obj in _GENERATION_OBJECTS:
            dataframes_dict[obj] = generate_previous_values(dataframe, 'datetime_local', _GEN_TARGET_COLUMN)
            dataframes_dict[obj] = dataframes_dict[obj].drop('datetime_unix', axis=1)

        elif obj in _CONSUMPTION_OBJECTS:
            dataframes_dict[obj] = generate_previous_values(dataframe, 'datetime_local', _CON_TARGET_COLUMN)

        else:
            raise ValueError('Unknown target!')

        dataframes_dict[obj] = pd.merge(weather_df, dataframes_dict[obj], on='datetime_local')

        prep_data_dict['Object'].append(obj)
        prep_data_dict['First timestamp'].append(dataframes_dict[obj]['datetime_local'].min())
        prep_data_dict['Last timestamp'].append(dataframes_dict[obj]['datetime_local'].max())
        prep_data_dict['Number of rows'].append(len(dataframes_dict[obj].index))

    print("Following data fetched:")
    print(pd.DataFrame.from_dict(fetch_data_dict).to_string(), '\n')

    print("Following data prepared:")
    print(pd.DataFrame.from_dict(prep_data_dict).to_string(), '\n')

    params_product = product(_OBJECTS, _OPTIMIZERS, _LOSS_FUNCTIONS, _LEARNING_RATES,
                             _ARCHITECTURES, _SEQUENCE_LENGHTS, _NUMBER_OF_LAYERS)
    for idx, parameters in enumerate(params_product, start=1):
        (object_name, optimizer_str, loss_str, lr, architecture, sequence_len, number_of_layers) = parameters
        print("Trying model %d with the following parameters:" % idx)
        print(parameters)

        # Code below makes sure that the Feed-Forward models aren't trained with different sequence lengths
        if architecture == 'FEEDFORWARD':

            # Skip the training process as the Feed-Forward model doesn't have a sequence length
            if sequence_len != _SEQUENCE_LENGHTS[0]:
                print("Model %d is already trained. Skipping..." % idx)
                continue

            # Set the sequence_len to 1 for Feed-Forward models
            else:
                sequence_len = 1

        # Get dataset according to the object
        dataset = None
        if object_name in _GENERATION_OBJECTS:
            dataset = \
                get_datasets(dataframes_dict[object_name].copy(), _TESTING_INTERVALS, sequence_len, _GEN_TARGET_COLUMN)
        elif object_name in _CONSUMPTION_OBJECTS:
            dataset = \
                get_datasets(dataframes_dict[object_name].copy(), _TESTING_INTERVALS, sequence_len, _CON_TARGET_COLUMN)
        else:
            raise ValueError('Unknown target!')

        # Construct the tensorflow sequential model according to the current parameters
        model = construct_model(architecture, sequence_len, dataset.num_features, number_of_layers)

        # Set the optimizer
        optimizer = None
        if optimizer_str == 'Adam':
            optimizer = tf.keras.optimizers.Adam(lr)
        elif optimizer_str == 'RMSProp':
            optimizer = tf.keras.optimizers.RMSprop(lr)
        else:
            raise ValueError('Unknown optimizer!')

        # Set the loss function
        loss = None
        if loss_str == 'MAE':
            loss = tf.keras.losses.MeanAbsoluteError()
        elif loss_str == 'MSE':
            loss = tf.keras.losses.MeanSquaredError()
        elif loss_str == 'MAPE':
            loss = tf.keras.losses.MeanAbsolutePercentageError()
        else:
            raise ValueError('Unknown loss function!')

        model.compile(optimizer=optimizer, loss=loss)
        model.summary()

        model_metrics_df = None

        total_epochs = 0
        for num_of_epochs in _NUMBER_OF_EPOCHS:
            total_epochs += num_of_epochs

            model.fit(dataset.x_train, dataset.y_train, batch_size=256, epochs=num_of_epochs)

            # Test the model with training data (in-sample)
            y_train_pred = model.predict(dataset.x_train).reshape(-1)
            y_train_real = dataset.y_train.reshape(-1)

            # Test the model with testing data (out of sample)
            y_test_pred = model.predict(dataset.x_test).reshape(-1)
            y_test_real = dataset.y_test.reshape(-1)

            # Model results dictionary
            model_dict = {
                'id': [idx],
                'number_of_epochs': [total_epochs],
                'model': ['Generation' if object_name in _GENERATION_OBJECTS else 'Consumption'],
                'architecture': [architecture],
                'object': [object_name],
                'optimizer': [optimizer_str],
                'loss': [loss_str],
                'learning_rate': [lr],
                'sequence_len': [sequence_len],
                'number_of_layers': [number_of_layers],
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

            # Model results with the same parameters but different number of epochs are store in the same dataframe
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

            plt.title("Comparison (Model ID %d, epochs - %d)" % (idx, total_epochs))
            plt.legend(loc="upper left", fontsize="x-large")
            plt.savefig("plots/model_%d_%d.png" % (idx, total_epochs))

            # Cleanup plotting environment after each loop
            plt.close('all')
            plt.cla()
            plt.clf()

        model_metrics_df.to_csv(_RESULTS_FILE, mode='a', index=False, header=not(exists(_RESULTS_FILE)))

        # Free up memory
        del dataset, model, model_dict, model_metrics_df, loss, optimizer
        del y_train_real, y_train_pred, y_test_pred, y_test_real, total_epochs, test_len
        gc.collect()
