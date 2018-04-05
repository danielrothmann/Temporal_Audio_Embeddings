from keras.layers import Input, LSTM, RepeatVector, Dense, Flatten
from keras.models import Model, Sequential


def prepare_autoencoder(timesteps,
                        input_dim,
                        latent_dim,
                        optimizer_type,
                        loss_type) -> (Model, Model):
    """Prepares sequential autoencoder model

        Args:
            :param timesteps: The number of timesteps in input sequence
            :param input_dim: The dimensions of the input
            :param latent_dim: The latent dimensionality of LSTM
            :param optimizer_type: The type of optimizer to use
            :param loss_type: The type of loss to use

        Returns:
            Autoencoder model, Encoder model
    """

    inputs = Input(shape=(timesteps, input_dim))
    encoded = LSTM(int(input_dim / 2), activation="relu", return_sequences=True)(inputs)
    encoded = LSTM(latent_dim, activation="relu", return_sequences=False)(encoded)
    decoded = RepeatVector(timesteps)(encoded)
    decoded = LSTM(int(input_dim / 2), activation="relu", return_sequences=True)(decoded)
    decoded = LSTM(input_dim, return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)

    autoencoder.compile(optimizer=optimizer_type, loss=loss_type,  metrics=['acc'])
    print(autoencoder.summary())

    return autoencoder, encoder


def prepare_processor(input_dim,
                      output_dim,
                      output_timesteps,
                      hidden_dim,
                      dense_depth,
                      lstm_depth,
                      optimizer_type,
                      loss_type) -> Model:
    """Prepares the processing LSTM decoder network

            Args:
                :param input_dim: The dimensions of the input
                :param output_dim: The dimensions of the output
                :param output_timesteps: The number of timesteps to predict for the output
                :param hidden_dim: The dimensionality of internal network layers
                :param dense_depth: The number of dense hidden layers
                :param lstm_depth: The number of LSTM hidden layers
                :param optimizer_type: The type of optimizer to use (eg. "rmsprop")
                :param loss_type: The type of loss to use (eg. "mse")

            Returns:
                Processing decoder model
        """

    model = Sequential()
    model.add(Dense(input_dim, input_shape=(input_dim,), activation="tanh"))

    for i in range(dense_depth):
        model.add(Dense(hidden_dim, activation="tanh"))

    model.add(RepeatVector(output_timesteps))

    for i in range(lstm_depth):
        model.add(LSTM(hidden_dim, return_sequences=True))

    model.add(LSTM(output_dim, return_sequences=True))
    model.add(Flatten())

    model.compile(optimizer=optimizer_type, loss=loss_type, metrics=['acc'])
    print(model.summary())

    return model
