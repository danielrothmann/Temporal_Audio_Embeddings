from keras.layers import Input, LSTM, RepeatVector, Dense, Flatten, Conv1D, GlobalAveragePooling1D, Lambda, Add, Reshape, Activation, AlphaDropout
from keras.models import Model, Sequential
from keras.activations import relu
import numpy as np

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
                      hidden_dim,
                      dense_depth,
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
    model.add(Dense(input_dim, input_shape=(input_dim,), activation="selu", kernel_initializer='lecun_normal'))

    model.add(AlphaDropout(0.2))

    for i in range(dense_depth):
        model.add(Dense(hidden_dim, activation="selu", kernel_initializer='lecun_normal'))
        model.add(AlphaDropout(0.2))

    model.add(Dense(output_dim, activation="softmax"))

    model.compile(optimizer=optimizer_type, loss=loss_type, metrics=['acc'])
    print(model.summary())

    return model


def prepare_fftnet(sample_input_dim, aux_input_dim, depth, channels=256, optimizer_type='rmsprop', loss_type='mse') -> Model:
    sample_inputs = Input(shape=(sample_input_dim, channels), name="Sample_Input")
    # aux_inputs = Input(shape=(aux_input_dim,), name="Auxiliary_Input")

    fftnet_layer = sample_inputs
    fft_input_dim = sample_input_dim

    for i in range(depth):
        fftnet_layer, fft_input_dim = apply_fftnet_layer(fftnet_layer, fft_input_dim, channels)

    dense = Dense(channels, activation='softmax', name='Output')(fftnet_layer)

    model = Model(sample_inputs, dense)
    model.compile(optimizer=optimizer_type, loss=loss_type,  metrics=['acc'])
    print(model.summary())

    return model

def apply_fftnet_layer(inputs, input_dim, channels):
    split_len = int(input_dim / 2)
    input1 = Lambda(lambda x: x[:split_len], output_shape=(split_len, channels), name='Split_L_' + str(split_len))(inputs)
    input2 = Lambda(lambda x: x[split_len:], output_shape=(split_len, channels), name='Split_R_' + str(split_len))(inputs)
    conv1 = Conv1D(channels, 1, name='Conv1D_L_' + str(split_len))(input1)
    conv2 = Conv1D(channels, 1, name='Conv1D_R_' + str(split_len))(input2)
    merged = Add(name='Add_' + str(split_len))([conv1, conv2])
    merged = Activation(activation='relu', name='ReLu_' + str(split_len))(merged)
    conv3 = Conv1D(channels, 1, activation='relu', name='Conv1D_LR_' + str(split_len))(merged)
    return conv3, split_len
