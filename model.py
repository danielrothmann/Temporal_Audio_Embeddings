from keras.layers import Input, LSTM, RepeatVector
from keras.models import Model


def prepare_model(timesteps, input_dim, latent_dim, optimizer_type, loss_type) -> (Model, Model):
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

