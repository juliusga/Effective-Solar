import tensorflow as tf

_NUMBER_OF_UNITS = 64


def construct_lstm(num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(_NUMBER_OF_UNITS, input_shape=(num_timesteps, num_features), return_sequences=True)
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS, return_sequences=True))
        model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS, return_sequences=True))

    model.add(tf.keras.layers.LSTM(_NUMBER_OF_UNITS))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_gru(num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.GRU(_NUMBER_OF_UNITS, input_shape=(num_timesteps, num_features), return_sequences=True)
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS, return_sequences=True))
        model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS, return_sequences=True))

    model.add(tf.keras.layers.GRU(_NUMBER_OF_UNITS))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_feed_forward(num_features: int, num_layers: int) -> tf.keras.Sequential:
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, input_shape=(None, num_features), activation='relu'),
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'),
        tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'),
    ])
    if num_layers == 4:
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
        model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))

    model.add(tf.keras.layers.Dense(_NUMBER_OF_UNITS, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation=tf.keras.layers.LeakyReLU(0.2)))
    return model


def construct_model(architecture: str, num_timesteps: int, num_features: int, num_layers: int) -> tf.keras.Sequential:
    if architecture == 'FEEDFORWARD':
        return construct_feed_forward(num_features, num_layers)

    elif architecture == 'LSTM':
        return construct_lstm(num_timesteps, num_features, num_layers)

    elif architecture == 'GRU':
        return construct_gru(num_timesteps, num_features, num_layers)

    else:
        raise ValueError('Unknown architecture!')
