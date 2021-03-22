import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Flatten, Reshape
from tensorflow import keras
import tensorflow.keras.backend as K


class Sampling(Layer):
    def call(self, inputs, **kwargs):
        mean, log_var = inputs
        return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean


class Encoder(Layer):
    def __init__(self, hidden_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.flatten = Flatten()
        self.dense1 = Dense(150, activation='selu')
        self.dense2 = Dense(100, activation='selu')
        self.codings_mean = Dense(self.hidden_dim)
        self.codings_log_var = Dense(self.hidden_dim)
        self.sampling = Sampling()

    def call(self, inputs, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        mean = self.codings_mean(x)
        log_var = self.codings_log_var(x)
        codings = self.sampling(mean, log_var)
        return mean, log_var, codings


class Decoder(Layer):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.dense1 = Dense(100, activation='selu')
        self.dense2 = Dense(150, activation='selu')
        self.dense3 = Dense(28*28, activation='sigmoid')
        self.ouput = Reshape([28, 28])

    def call(self, inputs, **kwargs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        output = self.ouput(x)
        return output


class VAE(keras.Model):
    def __init__(self, hidden_dim, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = Encoder(hidden_dim)
        self.decoder = Decoder()

    def call(self, inputs, **kwargs):
        x = self.encoder(inputs)
        output = self.decoder(x)
        return output


class LatentLoss(keras.losses.Loss):
    def __init__(self, encoder, **kwargs):
        super(LatentLoss, self).__init__(**kwargs)
        self.encoder = encoder

    def call(self, **kwargs):
        latent_loss = -0.5 * K.sum(
            1 + self.encoder.codings_log_var - K.exp(self.encoder.codings_log_var) - K.square(self.encoder.codings_mean),
            axis=-1
        )
        return K.mean(latent_loss) / 784


def rounded_accuracy(y_true, y_pred):
    return keras.metrics.binary_accuracy(tf.round(y_true), tf.round(y_pred))


if __name__ == '__main__':
    vae = VAE(hidden_dim=10)
    latent_loss = LatentLoss(vae.encoder)
    vae.add_loss(latent_loss)
    vae.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=[rounded_accuracy])







